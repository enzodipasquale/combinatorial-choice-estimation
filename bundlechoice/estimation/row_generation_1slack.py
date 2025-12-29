"""
Row generation solver with 1slack formulation for modular bundle choice estimation (v2).
This module implements a simplified row generation approach with a single scalar utility variable.
"""
import numpy as np
from datetime import datetime
from typing import Tuple, List, Optional, Any, Dict
from numpy.typing import NDArray
import logging
import gurobipy as gp
from gurobipy import GRB
from bundlechoice.utils import get_logger, suppress_output
from .base import BaseEstimationManager
logger = get_logger(__name__)

# Ensure root logger is configured for INFO level output
logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s][%(process)d][%(name)s] %(message)s')


class RowGeneration1SlackManager(BaseEstimationManager):
    """
    Implements the row generation algorithm with 1slack formulation for parameter estimation in modular bundle choice models.

    This solver uses a single scalar utility variable instead of one per simulation/agent pair,
    with constraints of the form: u >= sum(si) errors_si + sum(si) sum(k) x_si,k * theta_k

    This solver is designed for use with the v2 BundleChoice API and its managers. It supports distributed computation via MPI and Gurobi for solving the master problem.
    """
    def __init__(
                self,
                comm_manager: Any,
                dimensions_cfg: Any,
                row_generation_cfg: Any,
                data_manager: Any,
                feature_manager: Any,
                subproblem_manager: Any
                ) -> None:
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

    def _setup_gurobi_model_params(self) -> Any:
        """Create and set up Gurobi model with parameters from configuration."""    
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

    def _initialize_master_problem(self) -> None:
        """Create and configure master problem (Gurobi model) with 1slack formulation."""
        obs_features = self.get_obs_features()
        if self.is_root():
            self.master_model = self._setup_gurobi_model_params()    
            theta = self.master_model.addMVar(self.num_features, obj=-obs_features, ub=self.row_generation_cfg.theta_ubs, name='parameter')
            if self.row_generation_cfg.theta_lbs is not None:
                theta.lb = self.row_generation_cfg.theta_lbs
            else:
                # Default: non-negativity constraints (lb=0 for all variables)
                theta.lb = 0.0
            u_bar = self.master_model.addVar(obj=1, name='utility')  # Single scalar utility variable
            
            self.master_model.optimize()
            logger.info("Master Initialized (1slack formulation)")
            self.master_variables = (theta, u_bar)
            self.theta_val = theta.X
            self.log_parameter()
        else:
            self.theta_val = np.empty(self.num_features, dtype=np.float64)

        self.theta_val = self.comm_manager.broadcast_array(self.theta_val, root=0)

    def _master_iteration(self, optimal_bundles: NDArray[np.float64], 
                         timing_dict: Dict[str, float]) -> bool:
        """Perform one iteration of master problem (1slack). Returns True if stopping criterion met."""
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
            # Pre-allocate array for buffer-based broadcast (must match root's shape/dtype)
            theta_val = np.empty(self.num_features, dtype=np.float64)
            timing_dict['master_prep'] = 0.0
            timing_dict['master_update'] = 0.0
            timing_dict['master_optimize'] = 0.0
            
        # Broadcast theta and stop flag using buffer-based operations (no pickle)
        t_mpi_broadcast_start = datetime.now()
        self.theta_val, stop = self.comm_manager.broadcast_array_with_flag(theta_val, stop, root=0)
        timing_dict['mpi_broadcast'] = (datetime.now() - t_mpi_broadcast_start).total_seconds()
        return stop

    def solve(self) -> NDArray[np.float64]:
        """Run row generation with 1slack formulation. Returns estimated parameter vector."""
        if self.is_root():
            print("=" * 70)
            print("ROW GENERATION (1SLACK)")
            print("=" * 70)
            print()  # Blank line after header
            print(f"  Problem: {self.dimensions_cfg.num_agents} agents × {self.dimensions_cfg.num_items} items, {self.num_features} features")
            if self.dimensions_cfg.num_simuls > 1:
                print(f"  Simulations: {self.dimensions_cfg.num_simuls}")
            print(f"  Max iterations: {self.row_generation_cfg.max_iters if self.row_generation_cfg.max_iters != float('inf') else '∞'}")
            print(f"  Min iterations: {self.row_generation_cfg.min_iters}")
            print(f"  Optimality tolerance: {self.row_generation_cfg.tolerance_optimality}")
            if self.row_generation_cfg.max_slack_counter < float('inf'):
                print(f"  Max slack counter: {self.row_generation_cfg.max_slack_counter}")
            if self.row_generation_cfg.tol_row_generation > 0:
                print(f"  Row generation tolerance: {self.row_generation_cfg.tol_row_generation}")
            if self.row_generation_cfg.row_generation_decay > 0:
                print(f"  Tolerance decay: {self.row_generation_cfg.row_generation_decay}")
            print()  # Blank line before starting
            print("  Starting row generation algorithm (1slack formulation)...")
            print()  # Blank line before iterations
        tic = datetime.now()
        self.subproblem_manager.initialize_local()
        self._initialize_master_problem()        
        self.slack_counter = {}
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
                init_time = 0.0  # 1slack doesn't track init time separately
                if self.is_root():
                    logger.info("Row generation ended after %d iterations in %.2f seconds.", iteration + 1, elapsed)
                    obj_val = self.master_model.ObjVal if hasattr(self.master_model, 'ObjVal') else None
                    self._log_timing_summary(init_time, elapsed, iteration + 1, timing_breakdown, obj_val, self.theta_val)
                
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
        init_time = 0.0  # 1slack doesn't track init time separately
        if iteration >= self.row_generation_cfg.max_iters:
            if self.is_root():
                logger.info("Row generation reached max iterations (%d) in %.2f seconds.", iteration, elapsed)
                obj_val = self.master_model.ObjVal if hasattr(self.master_model, 'ObjVal') else None
                self._log_timing_summary(init_time, elapsed, iteration, timing_breakdown, obj_val, self.theta_val)
            
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

    def _enforce_slack_counter(self) -> int:
        """Update slack counter and remove constraints that have been slack too long. Returns number removed."""
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

    def _log_timing_summary(self, init_time: float, total_time: float, 
                           num_iterations: int, timing_breakdown: Dict[str, List[float]],
                           obj_val: Optional[float] = None, theta: Optional[NDArray[np.float64]] = None) -> None:
        """Log comprehensive timing summary showing bottlenecks."""
        # Use print for statistics to avoid logging prefix clutter
        if self.is_root():
            print("=" * 70)
            print("ROW GENERATION SUMMARY (1SLACK)")
            print("=" * 70)
            
            # Show solution results
            if obj_val is not None:
                print(f"Objective value at solution: {obj_val:.6f}")
            if theta is not None:
                # For high-dimensional theta, show compact representation
                if len(theta) <= 10:
                    # Show all values if small
                    print(f"Theta at solution: {np.array2string(theta, precision=6, suppress_small=True)}")
                else:
                    # Show summary for high-dimensional theta
                    print(f"Theta at solution (dim={len(theta)}):")
                    print(f"  First 5: {np.array2string(theta[:5], precision=6, suppress_small=True)}")
                    print(f"  Last 5:  {np.array2string(theta[-5:], precision=6, suppress_small=True)}")
                    print(f"  Min: {theta.min():.6f}, Max: {theta.max():.6f}, Mean: {theta.mean():.6f}")
            
            print(f"Total iterations: {num_iterations}")
            print(f"Total time: {total_time:.2f}s")
            print()
            print("Timing Statistics:")
            
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
            
            print("  Component breakdown (sorted by total time):")
            for stat in component_stats:
                print(
                    f"  {stat['name']:16s}: {stat['total']:7.2f}s ({stat['pct']:5.1f}%) | "
                    f"avg: {stat['mean']:.3f}s ± {stat['std']:.3f}s | "
                    f"range: [{stat['min']:.3f}s, {stat['max']:.3f}s]"
                )
            
            unaccounted = total_time - total_accounted
            if abs(unaccounted) > 0.01:
                print(f"  Unaccounted time:          {unaccounted:7.2f}s ({100*unaccounted/total_time:5.1f}%)")
            
            print()  # Blank line to separate from next section

    def log_parameter(self) -> None:
        """Log current parameter values (if parameters_to_log is set in config)."""
        feature_ids = self.row_generation_cfg.parameters_to_log
        precision = 3
        if feature_ids is not None:
            logger.info("Parameters: %s", np.round(self.theta_val[feature_ids], precision))
        else:
            logger.info("Parameters: %s", np.round(self.theta_val, precision))
