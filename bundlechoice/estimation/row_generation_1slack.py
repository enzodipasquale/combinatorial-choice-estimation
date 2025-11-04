"""
Row generation solver with 1slack formulation for modular bundle choice estimation (v2).
This module implements a simplified row generation approach with a single scalar utility variable.
"""
import numpy as np
from datetime import datetime
from typing import Tuple, List, Optional, Any, Dict
import logging
import gurobipy as gp
from gurobipy import GRB
from bundlechoice.utils import get_logger, suppress_output
from .base import BaseEstimationSolver
logger = get_logger(__name__)

# Ensure root logger is configured for INFO level output
logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s][%(process)d][%(name)s] %(message)s')


class RowGeneration1SlackSolver(BaseEstimationSolver):
    """
    Implements the row generation algorithm with 1slack formulation for parameter estimation in modular bundle choice models.

    This solver uses a single scalar utility variable instead of one per simulation/agent pair,
    with constraints of the form: u >= sum(si) errors_si + sum(si) sum(k) x_si,k * theta_k

    This solver is designed for use with the v2 BundleChoice API and its managers. It supports distributed computation via MPI and Gurobi for solving the master problem.
    """
    def __init__(
                self, 
                comm_manager, 
                dimensions_cfg, 
                row_generation_cfg, 
                data_manager, 
                feature_manager, 
                subproblem_manager):
        """
        Initialize the RowGeneration1SlackSolver.

        Args:
            comm_manager: Communication manager for MPI operations
            dimensions_cfg: DimensionsConfig instance
            row_generation_cfg: RowGenerationConfig instance
            data_manager: DataManager instance
            feature_manager: FeatureManager instance
            subproblem_manager: SubproblemManager instance
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
        self.timing_stats = None  # Store detailed timing statistics

    def _setup_gurobi_model_params(self):
        """
        Create and set up Gurobi model with parameters from configuration.
        
        Returns:
            Gurobi model instance with configured parameters.
        """    
        with suppress_output():
            model = gp.Model()
            Method = self.row_generation_cfg.gurobi_settings.get("Method", 0)
            model.setParam('Method', Method)
            Threads = self.row_generation_cfg.gurobi_settings.get("Threads")
            if Threads is not None:
                model.setParam('Threads', Threads)
            LPWarmStart = self.row_generation_cfg.gurobi_settings.get("LPWarmStart", 2)
            model.setParam('LPWarmStart', LPWarmStart)
            OutputFlag = self.row_generation_cfg.gurobi_settings.get("OutputFlag", 0)
            model.setParam('OutputFlag', OutputFlag)
        return model

    def _initialize_master_problem(self):
        """
        Create and configure the master problem (Gurobi model) with 1slack formulation.
        
        Returns:
            tuple: (master_model, master_variables, theta_val)
        """
        obs_features = self.get_obs_features()
        if self.is_root():
            self.master_model = self._setup_gurobi_model_params()    
            theta = self.master_model.addMVar(self.num_features, obj=-obs_features, ub=self.row_generation_cfg.theta_ubs, name='parameter')
            if self.row_generation_cfg.theta_lbs is not None:
                theta.lb = self.row_generation_cfg.theta_lbs
            u_bar = self.master_model.addVar(obj=1, name='utility')  # Single scalar utility variable
            
            self.master_model.optimize()
            logger.info("Master Initialized (1slack formulation)")
            self.master_variables = (theta, u_bar)
            self.theta_val = theta.X
            self.log_parameter()
        else:
            self.theta_val = np.empty(self.num_features, dtype=np.float64)

        self.theta_val = self.comm_manager.broadcast_array(self.theta_val, root=0)

    def _master_iteration(self, optimal_bundles, timing_dict):
        """
        Perform one iteration of the master problem in the row generation algorithm with 1slack formulation.

        Args:
            pricing_results (list of np.ndarray): List of bundle selection matrices from pricing subproblems.
            timing_dict (dict): Dictionary to store timing information for this iteration.

        Returns:
            bool: Whether the stopping criterion is met.
        """
        t_mpi_gather_start = datetime.now()
        x_sim = self.feature_manager.compute_gathered_features(optimal_bundles)
        errors_sim = self.feature_manager.compute_gathered_errors(optimal_bundles)
        timing_dict['mpi_gather'] = (datetime.now() - t_mpi_gather_start).total_seconds()
        stop = False
        
        if self.is_root():
            t_master_prep_start = datetime.now()
            theta, u_bar = self.master_variables
            u_sim = (x_sim @ theta.X).sum() + errors_sim.sum()
            u_master = u_bar.X  # Single scalar value

            self.log_parameter()
            logger.info(f"ObjVal: {self.master_model.ObjVal}")
            reduced_cost = u_sim - u_master
            logger.info("Reduced cost: %s", reduced_cost)
            
            if reduced_cost < self.row_generation_cfg.tolerance_optimality:
                stop = True
                timing_dict['master_prep'] = (datetime.now() - t_master_prep_start).total_seconds()
                timing_dict['master_update'] = 0.0
                timing_dict['master_optimize'] = 0.0
            else:          
                # Only add constraint if there's a violation (like standard formulation)
                if u_sim > u_master * (1 + self.row_generation_cfg.tol_row_generation) + self.row_generation_cfg.tolerance_optimality:
                    timing_dict['master_prep'] = (datetime.now() - t_master_prep_start).total_seconds()
                    t_master_update_start = datetime.now()
                    agents_utilities = (x_sim @ theta).sum() + errors_sim.sum()
                    self.master_model.addConstr(u_bar >= agents_utilities)
                    self._enforce_slack_counter()
                    logger.info("Number of constraints: %d", self.master_model.NumConstrs)
                    timing_dict['master_update'] = (datetime.now() - t_master_update_start).total_seconds()
                    t_master_optimize_start = datetime.now()
                    self.master_model.optimize()
                    timing_dict['master_optimize'] = (datetime.now() - t_master_optimize_start).total_seconds()
                    self.row_generation_cfg.tol_row_generation *= self.row_generation_cfg.row_generation_decay
                else:
                    timing_dict['master_prep'] = (datetime.now() - t_master_prep_start).total_seconds()
                    timing_dict['master_update'] = 0.0
                    timing_dict['master_optimize'] = 0.0
            
            # Get theta values for broadcasting
            theta_val = theta.X
        else:
            stop = False
            theta_val = None
            timing_dict['master_prep'] = 0.0
            timing_dict['master_update'] = 0.0
            timing_dict['master_optimize'] = 0.0
            
        # Broadcast theta and stop flag together (single broadcast reduces latency)
        t_mpi_broadcast_start = datetime.now()
        self.theta_val, stop = self.comm_manager.broadcast_from_root((theta_val, stop), root=0)
        timing_dict['mpi_broadcast'] = (datetime.now() - t_mpi_broadcast_start).total_seconds()
        return stop

    def solve(self):
        """
        Run the row generation algorithm with 1slack formulation to estimate model parameters.

        Returns:
            tuple: (theta_val)
                - theta_val (np.ndarray): Estimated parameter vector.
        """
        logger.info("=== ROW GENERATION 1SLACK ===")
        tic = datetime.now()
        self.subproblem_manager.initialize_local()
        self._initialize_master_problem()        
        self.slack_counter = {}
        logger.info("Starting row generation loop (1slack formulation).")
        iteration = 0
        
        # Detailed timing tracking
        timing_breakdown = {
            'pricing': [],
            'mpi_gather': [],
            'master_prep': [],
            'master_update': [],
            'master_optimize': [],
            'mpi_broadcast': []
        }
        
        while iteration < self.row_generation_cfg.max_iters:
            logger.info(f"ITERATION {iteration + 1}")
            iter_timing = {}
            
            # Pricing phase
            t_pricing = datetime.now()
            optimal_bundles = self.subproblem_manager.solve_local(self.theta_val)
            iter_timing['pricing'] = (datetime.now() - t_pricing).total_seconds()
            
            # Master iteration (with internal timing)
            t_master_start = datetime.now()
            stop = self._master_iteration(optimal_bundles, iter_timing)
            iter_timing['master_total'] = (datetime.now() - t_master_start).total_seconds()
            
            # Store timing breakdown
            for key in timing_breakdown.keys():
                if key in iter_timing:
                    timing_breakdown[key].append(iter_timing[key])
            
            if stop and iteration >= self.row_generation_cfg.min_iters:
                elapsed = (datetime.now() - tic).total_seconds()
                if self.is_root():
                    logger.info("Row generation ended after %d iterations in %.2f seconds.", iteration + 1, elapsed)
                    logger.info(f"ObjVal: {self.master_model.ObjVal}")
                
                # Store timing statistics
                if self.is_root():
                    total_pricing = np.sum(timing_breakdown.get('pricing', [0]))
                    total_master = (np.sum(timing_breakdown.get('master_prep', [0])) + 
                                  np.sum(timing_breakdown.get('master_update', [0])) + 
                                  np.sum(timing_breakdown.get('master_optimize', [0])))
                    total_mpi = (np.sum(timing_breakdown.get('mpi_gather', [0])) + 
                                np.sum(timing_breakdown.get('mpi_broadcast', [0])))
                    
                    self.timing_stats = {
                        'total_time': elapsed,
                        'num_iterations': iteration + 1,
                        'pricing_time': total_pricing,
                        'master_time': total_master,
                        'mpi_time': total_mpi,
                        'pricing_time_pct': 100 * total_pricing / elapsed if elapsed > 0 else 0,
                        'master_time_pct': 100 * total_master / elapsed if elapsed > 0 else 0,
                        'mpi_time_pct': 100 * total_mpi / elapsed if elapsed > 0 else 0,
                    }
                else:
                    self.timing_stats = None
                break
            iteration += 1
        
        elapsed = (datetime.now() - tic).total_seconds()
        if iteration >= self.row_generation_cfg.max_iters:
            if self.is_root():
                logger.info("Row generation reached max iterations (%d) in %.2f seconds.", iteration, elapsed)
            
            # Store timing statistics
            if self.is_root():
                total_pricing = np.sum(timing_breakdown.get('pricing', [0]))
                total_master = (np.sum(timing_breakdown.get('master_prep', [0])) + 
                              np.sum(timing_breakdown.get('master_update', [0])) + 
                              np.sum(timing_breakdown.get('master_optimize', [0])))
                total_mpi = (np.sum(timing_breakdown.get('mpi_gather', [0])) + 
                            np.sum(timing_breakdown.get('mpi_broadcast', [0])))
                
                self.timing_stats = {
                    'total_time': elapsed,
                    'num_iterations': iteration,
                    'pricing_time': total_pricing,
                    'master_time': total_master,
                    'mpi_time': total_mpi,
                    'pricing_time_pct': 100 * total_pricing / elapsed if elapsed > 0 else 0,
                    'master_time_pct': 100 * total_master / elapsed if elapsed > 0 else 0,
                    'mpi_time_pct': 100 * total_mpi / elapsed if elapsed > 0 else 0,
                }
            else:
                self.timing_stats = None
        
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

    def log_parameter(self):
        feature_ids = self.row_generation_cfg.parameters_to_log
        precision = 3
        if feature_ids is not None:
            logger.info("Parameters: %s", np.round(self.theta_val[feature_ids], precision))
        else:
            logger.info("Parameters: %s", np.round(self.theta_val, precision))
