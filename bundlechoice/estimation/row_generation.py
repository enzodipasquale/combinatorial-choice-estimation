"""
Row generation solver for modular bundle choice estimation (v2).
This module will be used by BundleChoice to estimate parameters using row generation.
Future solvers can be added to this folder as well.
"""
import numpy as np
from numpy.typing import NDArray
from datetime import datetime
from typing import Tuple, List, Optional, Any, Dict, Callable
import logging
import gurobipy as gp
from gurobipy import GRB
from bundlechoice.utils import get_logger, suppress_output
from .base import BaseEstimationSolver
logger = get_logger(__name__)

# Ensure root logger is configured for INFO level output
logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s][%(process)d][%(name)s] %(message)s')


class RowGenerationSolver(BaseEstimationSolver):
    """
    Implements the row generation algorithm for parameter estimation in modular bundle choice models.

    This solver is designed for use with the v2 BundleChoice API and its managers. It supports distributed computation via MPI and Gurobi for solving the master problem.
    """
    def __init__(
                self, 
                comm_manager, 
                dimensions_cfg, 
                row_generation_cfg, 
                data_manager, 
                feature_manager, 
                subproblem_manager,
                theta_init=None):
        """
        Initialize the RowGenerationSolver.

        Args:
            comm_manager: Communication manager for MPI operations
            dimensions_cfg: DimensionsConfig instance
            row_generation_cfg: RowGenerationConfig instance
            data_manager: DataManager instance
            feature_manager: FeatureManager instance
            subproblem_manager: SubproblemManager instance
            theta_init: Optional initial theta for warm start
        """
        super().__init__(
            comm_manager=comm_manager,
            dimensions_cfg=dimensions_cfg,
            data_manager=data_manager,
            feature_manager=feature_manager,
            subproblem_manager=subproblem_manager
        )
        
        self.row_generation_cfg = row_generation_cfg
        self.master_model = None
        self.master_variables = None
        self.theta_val = None
        self.theta_hat = None
        self.slack_counter = None
        self.theta_init = theta_init

    def _setup_gurobi_model_params(self):
        """
        Create and set up Gurobi model with parameters from configuration.
        
        Returns:
            Gurobi model instance with configured parameters.
        """    
        # Default values for parameters not specified in config
        defaults = {
            'Method': 0,
            'LPWarmStart': 2,
            'OutputFlag': 0
        }
        
        with suppress_output():
            model = gp.Model()
            
            # Merge defaults with user settings (user settings take precedence)
            params = {**defaults, **self.row_generation_cfg.gurobi_settings}
            
            # Set all parameters
            for param_name, value in params.items():
                if value is not None:
                    model.setParam(param_name, value)
        return model

    def _initialize_master_problem(self):
        """
        Create and configure the master problem (Gurobi model).
        
        Returns:
            tuple: (master_model, master_variables, theta_val)
        """
        obs_features = self.get_obs_features()
        if self.is_root():
            self.master_model = self._setup_gurobi_model_params()    
            theta = self.master_model.addMVar(self.num_features, obj= - obs_features, ub=self.row_generation_cfg.theta_ubs, name='parameter')
            if self.row_generation_cfg.theta_lbs is not None:
                theta.lb = self.row_generation_cfg.theta_lbs
            
            # Apply warm start if provided
            if self.theta_init is not None:
                for k in range(self.num_features):
                    theta[k].Start = self.theta_init[k]
            
            u = self.master_model.addMVar(self.num_simuls * self.num_agents, obj=1, name='utility')
            
            # errors = self.input_data["errors"].reshape(-1, self.num_items)
            # self.master_model.addConstrs((
            #     u[si] >=
            #     errors[si] @ self.input_data["obs_bundle"][si % self.num_agents] +
            #     self.agents_obs_features[si % self.num_agents, :] @ theta
            #     for si in range(self.num_simuls * self.num_agents)
            # ))
            self.master_model.optimize()
            logger.info("Master Initialized")
            self.master_variables = (theta, u)
            self.theta_val = theta.X
            self.log_parameter()
        else:
            self.theta_val = np.empty(self.num_features, dtype=np.float64)
        
        self.theta_val = self.comm_manager.broadcast_array(self.theta_val, root=0)
    


    def _master_iteration(self, local_pricing_results, timing_dict):
        """
        Perform one iteration of the master problem in the row generation algorithm.

        Args:
            pricing_results (list of np.ndarray): List of bundle selection matrices from pricing subproblems.
            timing_dict (dict): Dictionary to store timing information for this iteration.

        Returns:
            bool: Whether the stopping criterion is met.
        """
        t_mpi_gather_start = datetime.now()
        x_sim = self.feature_manager.compute_gathered_features(local_pricing_results)
        errors_sim = self.feature_manager.compute_gathered_errors(local_pricing_results)
        timing_dict['mpi_gather'] = (datetime.now() - t_mpi_gather_start).total_seconds()
        
        stop = False
        if self.is_root():
            t_master_prep_start = datetime.now()
            theta, u = self.master_variables
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                u_sim = x_sim @ theta.X + errors_sim
            u_master = u.X

            violations = np.where(~np.isclose(u_master, u_sim, rtol = 1e-6, atol = 1e-6) * (u_master > u_sim))[0]
            if len(violations) > 0:
                logger.warning(
                    "Possible failure of demand oracle at agents ids: %s, "
                    "u_sim: %s, u_master: %s",
                    violations, u_sim[violations], u_master[violations]
                )

            self.log_parameter()
            logger.info(f"ObjVal: {self.master_model.ObjVal}")
            max_reduced_cost = np.max(u_sim - u_master)
            logger.info("Reduced cost: %s", max_reduced_cost)
            if max_reduced_cost < self.row_generation_cfg.tolerance_optimality:
                stop = True
            rows_to_add = np.where(u_sim > u_master * (1 + self.row_generation_cfg.tol_row_generation) + self.row_generation_cfg.tolerance_optimality)[0]
            logger.info("New constraints: %d", len(rows_to_add))
            timing_dict['master_prep'] = (datetime.now() - t_master_prep_start).total_seconds()
            
            t_master_update_start = datetime.now()
            self.master_model.addConstr(u[rows_to_add]  >= errors_sim[rows_to_add] + x_sim[rows_to_add] @ theta)
            self._enforce_slack_counter()
            logger.info("Number of constraints: %d", self.master_model.NumConstrs)
            timing_dict['master_update'] = (datetime.now() - t_master_update_start).total_seconds()
            
            t_master_optimize_start = datetime.now()
            self.master_model.optimize()
            timing_dict['master_optimize'] = (datetime.now() - t_master_optimize_start).total_seconds()
            
            theta_val = theta.X
            self.row_generation_cfg.tol_row_generation *= self.row_generation_cfg.row_generation_decay
        else:
            theta_val = None
            stop = False
            timing_dict['master_prep'] = 0.0
            timing_dict['master_update'] = 0.0
            timing_dict['master_optimize'] = 0.0
        
        # Broadcast theta and stop flag together (single broadcast reduces latency)
        t_mpi_broadcast_start = datetime.now()
        self.theta_val, stop = self.comm_manager.broadcast_from_root((theta_val, stop), root=0)
        timing_dict['mpi_broadcast'] = (datetime.now() - t_mpi_broadcast_start).total_seconds()
        
        return stop

    def solve(self, callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> NDArray[np.float64]:
        """
        Run the row generation algorithm to estimate model parameters.

        Args:
            callback: Optional callback function called after each iteration.
                     Signature: callback(info: dict) where info contains:
                     - 'iteration': Current iteration number (int)
                     - 'theta': Current parameter estimate (np.ndarray)
                     - 'objective': Current objective value (float)
                     - 'pricing_time': Time spent solving subproblems in seconds (float)
                     - 'master_time': Time spent on master problem in seconds (float)
        
        Returns:
            np.ndarray: Estimated parameter vector.
        """
        logger.info("=== ROW GENERATION ===")
        tic = datetime.now()
        
        t_init = datetime.now()
        self.subproblem_manager.initialize_local()
        self._initialize_master_problem()        
        self.slack_counter = {}
        init_time = (datetime.now() - t_init).total_seconds()
        
        logger.info("Starting row generation loop.")
        iteration = 0
        
        # Detailed timing tracking
        timing_breakdown = {
            'pricing': [],
            'mpi_gather': [],
            'master_prep': [],
            'master_update': [],
            'master_optimize': [],
            'mpi_broadcast': [],
            'callback': []
        }
        
        while iteration < self.row_generation_cfg.max_iters:
            logger.info(f"ITERATION {iteration + 1}")
            iter_timing = {}
            
            # Pricing phase
            t_pricing = datetime.now()
            local_pricing_results = self.subproblem_manager.solve_local(self.theta_val)
            iter_timing['pricing'] = (datetime.now() - t_pricing).total_seconds()
            
            # Master iteration (with internal timing)
            stop = self._master_iteration(local_pricing_results, iter_timing) 
            
            # Store timing breakdown
            for key in timing_breakdown.keys():
                if key in iter_timing:
                    timing_breakdown[key].append(iter_timing[key])
            
            # Callback phase
            if callback:
                t_callback = datetime.now()
                if self.is_root():
                    callback({
                        'iteration': iteration + 1,
                        'theta': self.theta_val.copy() if self.theta_val is not None else None,
                        'objective': self.master_model.ObjVal if hasattr(self.master_model, 'ObjVal') else None,
                        'pricing_time': iter_timing['pricing'],
                        'master_time': sum([iter_timing.get(k, 0) for k in ['mpi_gather', 'master_prep', 'master_update', 'master_optimize', 'mpi_broadcast']]),
                    })
                timing_breakdown['callback'].append((datetime.now() - t_callback).total_seconds())
            
            if stop and iteration >= self.row_generation_cfg.min_iters:
                if self.is_root():
                    elapsed = (datetime.now() - tic).total_seconds()
                    logger.info("Row generation ended after %d iterations in %.2f seconds.", iteration + 1, elapsed)
                    logger.info(f"ObjVal: {self.master_model.ObjVal}")
                    self._log_timing_summary(init_time, elapsed, iteration + 1, timing_breakdown)
                break
            iteration += 1
        
        # Log timing even if max iterations reached
        if iteration >= self.row_generation_cfg.max_iters and self.is_root():
            elapsed = (datetime.now() - tic).total_seconds()
            logger.info("Row generation reached max iterations (%d) in %.2f seconds.", iteration, elapsed)
            self._log_timing_summary(init_time, elapsed, iteration, timing_breakdown)
        
        self.theta_hat = self.theta_val
        return self.theta_hat

    def _enforce_slack_counter(self):
        """
        Update the slack counter for master problem constraints and remove those that have been slack for too long.

        Returns:
            int: Number of constraints removed.
        """
        if self.row_generation_cfg.max_slack_counter < float('inf'):
            to_remove = []
            for constr in self.master_model.getConstrs():
                if constr.Slack < -1e-6:
                    # Only add to counter when constraint is actually slack
                    if constr not in self.slack_counter:
                        self.slack_counter[constr] = 0
                    self.slack_counter[constr] += 1
                    if self.slack_counter[constr] >= self.row_generation_cfg.max_slack_counter:
                        to_remove.append(constr)
                if constr.Pi > 1e-6:
                    self.slack_counter.pop(constr, None)
            # Remove all constraints that exceeded the slack counter limit
            for constr in to_remove:
                self.master_model.remove(constr)
                self.slack_counter.pop(constr, None)
            num_removed = len(to_remove)
            logger.info("Removed constraints: %d", num_removed)
            return num_removed
        else:
            return 0



    def _log_timing_summary(self, init_time, total_time, num_iterations, timing_breakdown):
        """
        Log comprehensive timing summary showing bottlenecks.
        
        Args:
            init_time: Time spent on initialization
            total_time: Total elapsed time
            num_iterations: Number of iterations completed
            timing_breakdown: Dictionary of timing lists for each component
        """
        logger.info("=" * 70)
        logger.info("TIMING SUMMARY - Row Generation Performance Analysis")
        logger.info("=" * 70)
        logger.info(f"Total iterations: {num_iterations}")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Initialization time: {init_time:.3f}s ({100*init_time/total_time:.1f}%)")
        logger.info("-" * 70)
        
        # Calculate totals and percentages for each component
        component_stats = []
        total_accounted = init_time
        
        for component, times in timing_breakdown.items():
            if len(times) > 0:
                total = np.sum(times)
                mean = np.mean(times)
                std = np.std(times)
                min_t = np.min(times)
                max_t = np.max(times)
                pct = 100 * total / total_time
                total_accounted += total
                component_stats.append({
                    'name': component,
                    'total': total,
                    'mean': mean,
                    'std': std,
                    'min': min_t,
                    'max': max_t,
                    'pct': pct
                })
        
        # Sort by total time (descending) to show bottlenecks first
        component_stats.sort(key=lambda x: x['total'], reverse=True)
        
        logger.info("Component breakdown (sorted by total time):")
        for stat in component_stats:
            logger.info(
                f"  {stat['name']:16s}: {stat['total']:7.2f}s ({stat['pct']:5.1f}%) | "
                f"avg: {stat['mean']:.3f}s ± {stat['std']:.3f}s | "
                f"range: [{stat['min']:.3f}s, {stat['max']:.3f}s]"
            )
        
        logger.info("-" * 70)
        
        # Combined metrics
        total_pricing = np.sum(timing_breakdown.get('pricing', [0]))
        total_mpi = np.sum(timing_breakdown.get('mpi_gather', [0])) + np.sum(timing_breakdown.get('mpi_broadcast', [0]))
        total_master_work = (np.sum(timing_breakdown.get('master_prep', [0])) + 
                            np.sum(timing_breakdown.get('master_update', [0])) + 
                            np.sum(timing_breakdown.get('master_optimize', [0])))
        
        logger.info("Aggregated metrics:")
        logger.info(f"  Total pricing time:        {total_pricing:7.2f}s ({100*total_pricing/total_time:5.1f}%)")
        logger.info(f"  Total MPI communication:   {total_mpi:7.2f}s ({100*total_mpi/total_time:5.1f}%)")
        logger.info(f"  Total master problem work: {total_master_work:7.2f}s ({100*total_master_work/total_time:5.1f}%)")
        
        if num_iterations > 0:
            logger.info(f"  Avg time per iteration:    {total_time/num_iterations:.3f}s")
        
        unaccounted = total_time - total_accounted
        if abs(unaccounted) > 0.01:
            logger.info(f"  Unaccounted time:          {unaccounted:7.2f}s ({100*unaccounted/total_time:5.1f}%)")
        
        logger.info("=" * 70)

    def log_parameter(self):
        feature_ids = self.row_generation_cfg.parameters_to_log
        precision = 3
        if feature_ids is not None:
            logger.info("Parameters: %s", np.round(self.theta_val[feature_ids], precision))
        else:
            logger.info("Parameters: %s", np.round(self.theta_val, precision))
