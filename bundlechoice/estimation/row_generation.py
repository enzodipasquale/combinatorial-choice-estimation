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
import sys
import gurobipy as gp
from gurobipy import GRB
from mpi4py import MPI
from bundlechoice.utils import get_logger, suppress_output
from .base import BaseEstimationManager
from .result import EstimationResult

# Try to import tracemalloc for memory profiling (optional)
try:
    import tracemalloc
    TRACEMALLOC_AVAILABLE = True
except ImportError:
    TRACEMALLOC_AVAILABLE = False
    tracemalloc = None

logger = get_logger(__name__)

# Ensure root logger is configured for INFO level output
logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s][%(process)d][%(name)s] %(message)s')


class RowGenerationManager(BaseEstimationManager):
    """
    Implements the row generation algorithm for parameter estimation in modular bundle choice models.

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
        Initialize the RowGenerationManager.

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
        self.timing_stats = None  # Store detailed timing statistics
        self.theta_val = None
        self.theta_hat = None
        self.slack_counter = None

    def _setup_gurobi_model_params(self) -> Any:
        """Create and set up Gurobi model with parameters from configuration."""    
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

    def _initialize_master_problem(self, initial_constraints: Optional[Dict[str, NDArray]] = None) -> None:
        """
        Create and configure the master problem (Gurobi model).
        
        Args:
            initial_constraints: Optional dict with keys 'indices' and 'bundles' for warm-starting
        """
        obs_features = self.get_obs_features()
        if self.is_root():
            self.master_model = self._setup_gurobi_model_params()    
            theta = self.master_model.addMVar(self.num_features, obj= - obs_features, ub=self.row_generation_cfg.theta_ubs, name='parameter')
            if self.row_generation_cfg.theta_lbs is not None:
                # Set bounds per variable, None means unbounded (don't set)
                for k in range(self.num_features):
                    if k < len(self.row_generation_cfg.theta_lbs) and self.row_generation_cfg.theta_lbs[k] is not None:
                        theta[k].lb = float(self.row_generation_cfg.theta_lbs[k])
            else:
                # Default: non-negativity constraints (lb=0 for all variables)
                theta.lb = 0.0
            
            # Set start value if theta_init was provided
            if hasattr(self, '_theta_init_for_start') and self._theta_init_for_start is not None:
                theta.Start = self._theta_init_for_start

            u = self.master_model.addMVar(self.num_simuls * self.num_agents, obj=1, name='utility')
            
            # Add initial constraints if provided (warm-starting)
            if initial_constraints is not None and len(initial_constraints.get('indices', [])) > 0:
                indices = initial_constraints['indices']
                bundles = initial_constraints['bundles']
                for i, idx in enumerate(indices):
                    agent_id = idx % self.num_agents
                    sim_id = idx // self.num_agents
                    bundle = bundles[i]
                    
                    # Compute features and errors
                    features = self.feature_manager.features_oracle(agent_id, bundle, self.input_data)
                    error = (self.input_data["errors"][sim_id, agent_id] * bundle).sum()
                    
                    # Encode bundle as binary string
                    # Use hash of bundle for constraint name (avoids Gurobi 255 char limit)
                    bundle_hash = hash(bundle.tobytes())
                    constr_name = f"rowgen_{idx}_b{abs(bundle_hash) % 1000000000}"
                    
                    # Add constraint
                    self.master_model.addConstr(u[idx] >= error + features @ theta, name=constr_name)
                
                logger.info("Added %d initial constraints for warm-start", len(indices))

            self.master_model.optimize()
            logger.info("Master Initialized")
            self.master_variables = (theta, u)
            if self.master_model.Status == GRB.OPTIMAL:
                self.theta_val = theta.X
            else:
                logger.warning("Master problem not optimal at initialization, status=%s", self.master_model.Status)
                self.theta_val = np.zeros(self.num_features, dtype=np.float64)
            self.log_parameter()
        else:
            self.theta_val = np.empty(self.num_features, dtype=np.float64)
        
        self.theta_val = self.comm_manager.broadcast_array(self.theta_val, root=0)
    


    def _check_early_convergence(self, local_pricing_results: NDArray[np.float64]) -> bool:
        """
        Check convergence using Allreduce before full gather (Phase 2 optimization).
        Returns True if should stop early (converged).
        """
        from mpi4py import MPI
        
        # Compute features and errors locally (no gather yet)
        features_local = self.feature_manager.compute_rank_features(local_pricing_results)
        errors_local = (self.data_manager.local_data["errors"] * local_pricing_results).sum(1)
        
        # Compute local u_sim using local features/errors and broadcasted theta
        if len(features_local) > 0 and len(errors_local) > 0:
            u_sim_local = features_local @ self.theta_val + errors_local
            max_u_sim_local = np.max(u_sim_local) if len(u_sim_local) > 0 else -np.inf
        else:
            max_u_sim_local = -np.inf
        
        # Get global max u_sim using Allreduce
        max_u_sim_global = self.comm_manager.comm.allreduce(max_u_sim_local, op=MPI.MAX)
        
        # Broadcast u_master bounds from root (needed for convergence check)
        if self.is_root():
            if self.master_model is None:
                return False
            _, u = self.master_variables
            u_master = u.X
            min_u_master = np.min(u_master) if len(u_master) > 0 else -np.inf
        else:
            min_u_master = None
        
        # Broadcast min_u_master to all ranks
        min_u_master_buffer = np.array([min_u_master], dtype=np.float64) if self.is_root() else np.empty(1, dtype=np.float64)
        self.comm_manager.comm.Bcast(min_u_master_buffer, root=0)
        min_u_master = min_u_master_buffer[0]
        
        # Compute upper bound for max_reduced_cost: max(u_sim) - min(u_master) >= max(u_sim - u_master)
        # If this upper bound < tolerance, then we're definitely converged
        max_reduced_cost_upper_bound = max_u_sim_global - min_u_master
        
        # Check if we should stop early (only if not in suboptimal mode)
        suboptimal_mode = getattr(self.subproblem_manager, '_suboptimal_mode', False)
        should_stop = (max_reduced_cost_upper_bound < self.row_generation_cfg.tolerance_optimality) and (not suboptimal_mode)
        
        return should_stop

    def _master_iteration(self, local_pricing_results: NDArray[np.float64], 
                         timing_dict: Dict[str, float]) -> bool:
        """Perform one iteration of master problem. Returns True if stopping criterion met."""
        # Phase 2 optimization: Early convergence check using Allreduce (optional, fails silently)
        should_stop_early = False
        try:
            should_stop_early = self._check_early_convergence(local_pricing_results)
        except Exception:
            # If early check fails, continue with normal flow
            pass
        
        if should_stop_early:
            # Early convergence detected - skip full gather
            if self.is_root():
                theta, u = self.master_variables
                theta_val = theta.X
            else:
                theta_val = None
            if not self.is_root() and self.theta_val is None:
                self.theta_val = np.empty(self.num_features, dtype=np.float64)
            self.theta_val, _ = self.comm_manager.broadcast_array_with_flag(
                theta_val if self.is_root() else self.theta_val, 
                True, root=0
            )
            timing_dict['mpi_gather'] = 0.0
            timing_dict['master_prep'] = 0.0
            timing_dict['master_update'] = 0.0
            timing_dict['master_optimize'] = 0.0
            return True
        
        # COMBINED GATHER OPTIMIZATION: Gather bundles once, compute features/errors on root
        # This avoids 2 additional MPI gather operations
        t_gather_bundles_start = datetime.now()
        bundles_sim = self.comm_manager.concatenate_array_at_root_fast(local_pricing_results, root=0)
        gather_bundles_time = (datetime.now() - t_gather_bundles_start).total_seconds()
        timing_dict['gather_bundles'] = gather_bundles_time
        
        # Compute features and errors on root from gathered bundles (no additional gather)
        if self.is_root():
            t_comp_features_start = datetime.now()
            x_sim = self.feature_manager.compute_all_features_on_root(bundles_sim)
            comp_features_time = (datetime.now() - t_comp_features_start).total_seconds()
            timing_dict['compute_features'] = comp_features_time
            
            t_comp_errors_start = datetime.now()
            # Compute errors on root using input_data
            errors_input = self.data_manager.input_data["errors"]
            if errors_input.ndim == 2:
                errors_reshaped = errors_input
            elif errors_input.ndim == 3:
                errors_reshaped = errors_input.reshape(-1, self.num_items)
            else:
                raise ValueError(f"Unexpected errors shape: {errors_input.shape}")
            
            errors_sim = (errors_reshaped * bundles_sim).sum(1)
            comp_errors_time = (datetime.now() - t_comp_errors_start).total_seconds()
            timing_dict['compute_errors'] = comp_errors_time
        else:
            x_sim = None
            errors_sim = None
            timing_dict['compute_features'] = 0.0
            timing_dict['compute_errors'] = 0.0
        
        # mpi_gather is now just the bundle gather time (the actual MPI operation)
        timing_dict['mpi_gather'] = gather_bundles_time
        
        stop = False
        if self.is_root():
            t_master_prep_start = datetime.now()
            theta, u = self.master_variables
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                u_sim = x_sim @ theta.X + errors_sim
            u_master = u.X

            violations = np.where(~np.isclose(u_master, u_sim, rtol = 1e-5, atol = 1e-5) * (u_master > u_sim))[0]
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
            # Check if we're in suboptimal cuts mode (set by callback)
            suboptimal_mode = getattr(self.subproblem_manager, '_suboptimal_mode', False)
            if max_reduced_cost < self.row_generation_cfg.tolerance_optimality:
                if not suboptimal_mode:
                    stop = True
                else:
                    logger.info("Reduced cost below tolerance, but suboptimal cuts mode active - continuing")
            rows_to_add = np.where(u_sim > u_master * (1 + self.row_generation_cfg.tol_row_generation) + self.row_generation_cfg.tolerance_optimality)[0]
            logger.info("New constraints: %d", len(rows_to_add))
            timing_dict['master_prep'] = (datetime.now() - t_master_prep_start).total_seconds()
            
            t_master_update_start = datetime.now()
            # Add constraints with names encoding index and bundle (hash for long bundles)
            if len(rows_to_add) > 0 and bundles_sim is not None:
                for idx in rows_to_add:
                    # Use hash of bundle for constraint name (avoids Gurobi 255 char limit)
                    bundle_hash = hash(bundles_sim[idx].tobytes())
                    constr_name = f"rowgen_{idx}_b{abs(bundle_hash) % 1000000000}"  # Use abs and mod to keep reasonable length
                    self.master_model.addConstr(u[idx] >= errors_sim[idx] + x_sim[idx] @ theta, name=constr_name)
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
        # Non-root ranks must pre-allocate theta_val for buffer-based broadcast
        if not self.is_root() and self.theta_val is None:
            self.theta_val = np.empty(self.num_features, dtype=np.float64)
        self.theta_val, stop = self.comm_manager.broadcast_array_with_flag(
            theta_val if self.is_root() else self.theta_val, 
            stop, root=0
        )
        timing_dict['mpi_broadcast'] = (datetime.now() - t_mpi_broadcast_start).total_seconds()
        
        return stop

    def solve(self, callback: Optional[Callable[[Dict[str, Any]], None]] = None,
              theta_init: Optional[NDArray[np.float64]] = None) -> EstimationResult:
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
            theta_init: Optional initial parameter vector. If None, uses default initialization.
        
        Returns:
            EstimationResult: Result object containing theta_hat and diagnostics.
        """
        if self.is_root():
            print("=" * 70)
            print("ROW GENERATION")
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
            print("  Starting row generation algorithm...")
            print()  # Blank line before iterations
        tic = datetime.now()
        
        t_init = datetime.now()
        
        self.subproblem_manager.initialize_local()
        
        # Initialize with theta_init if provided
        initial_constraints = None
        if theta_init is not None:
            if self.is_root():
                logger.info("Initializing with provided theta (warm start)")
                # Handle both EstimationResult and numpy array
                if hasattr(theta_init, 'theta_hat'):
                    theta_init_array = theta_init.theta_hat
                else:
                    theta_init_array = theta_init
                self.theta_val = np.asarray(theta_init_array, dtype=np.float64).copy()
            else:
                self.theta_val = np.empty(self.num_features, dtype=np.float64)
            
            
            self.theta_val = self.comm_manager.broadcast_array(self.theta_val, root=0)
            
            
            # Solve subproblems at initial theta to get initial constraints
            
            local_pricing_results = self.subproblem_manager.solve_local(self.theta_val)
            
            
            # Gather bundles - all processes must participate
            
            bundles_sim = self.comm_manager.concatenate_array_at_root_fast(local_pricing_results, root=0)
            
            
            if self.is_root() and bundles_sim is not None and len(bundles_sim) > 0:
                indices = np.arange(self.num_simuls * self.num_agents, dtype=np.int64)
                initial_constraints = {
                    'indices': indices,
                    'bundles': bundles_sim.astype(np.float64)
                }
                logger.info("Pre-computed %d initial constraints from theta_init", len(indices))
        
        # Store theta_init for Gurobi start value (extract array if EstimationResult)
        if theta_init is not None:
            if hasattr(theta_init, 'theta_hat'):
                self._theta_init_for_start = theta_init.theta_hat
            else:
                self._theta_init_for_start = theta_init
        else:
            self._theta_init_for_start = None
        
        self._initialize_master_problem(initial_constraints=initial_constraints)
        
        self.slack_counter = {}
        init_time = (datetime.now() - t_init).total_seconds()
        iteration = 0
        
        # Timing tracking (production - minimal overhead)
        timing_breakdown = {
            'pricing': [],
            'mpi_gather': [],
            'gather_bundles': [],
            'compute_features': [],
            'compute_errors': [],
            'master_prep': [],
            'master_update': [],
            'master_optimize': [],
            'mpi_broadcast': [],
            'callback': [],
        }
        
        # Rank distribution verification (once at start) - all ranks must participate
        from mpi4py import MPI
        num_local_agents_all = self.comm_manager.comm.allgather(
            self.data_manager.num_local_agents if self.data_manager else 0
        )
        if self.is_root():
            num_local_agents_array = np.array(num_local_agents_all)
            logger.info("=" * 70)
            logger.info("RANK DISTRIBUTION VERIFICATION")
            logger.info("=" * 70)
            logger.info(f"Total ranks: {len(num_local_agents_array)}")
            logger.info(f"Agents per rank: min={num_local_agents_array.min()}, "
                       f"max={num_local_agents_array.max()}, "
                       f"mean={num_local_agents_array.mean():.2f}, "
                       f"std={num_local_agents_array.std():.2f}")
            imbalance = num_local_agents_array.max() - num_local_agents_array.min()
            if imbalance > 1:
                logger.warning(f"Load imbalance detected: {imbalance} agents difference")
            else:
                logger.info("Load distribution is balanced")
            logger.info("=" * 70)
        
        while iteration < self.row_generation_cfg.max_iters:
            logger.info(f"ITERATION {iteration + 1}")
            iter_timing = {}
            
            # Subproblem callback (if configured)
            if self.row_generation_cfg.subproblem_callback is not None:
                master_model = self.master_model if self.is_root() else None
                self.row_generation_cfg.subproblem_callback(
                    iteration, 
                    self.subproblem_manager, 
                    master_model
                )
            
            # Pricing phase
            t_pricing = datetime.now()
            local_pricing_results = self.subproblem_manager.solve_local(self.theta_val)
            iter_timing['pricing'] = (datetime.now() - t_pricing).total_seconds()
            
            # Master iteration
            stop = self._master_iteration(local_pricing_results, iter_timing) 
            
            # Store timing breakdown (including enhanced diagnostics)
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
            
            # Print periodic timing summary every 10 iterations
            if self.is_root() and (iteration + 1) % 10 == 0:
                import sys
                elapsed = (datetime.now() - tic).total_seconds()
                obj_val = self.master_model.ObjVal if hasattr(self.master_model, 'ObjVal') else None
                print(f"\n=== INTERMEDIATE TIMING (Iteration {iteration + 1}) ===", flush=True)
                sys.stdout.flush()
                self._log_timing_summary(init_time, elapsed, iteration + 1, timing_breakdown, obj_val, self.theta_val)
                sys.stdout.flush()
            
            if stop and iteration >= self.row_generation_cfg.min_iters:
                elapsed = (datetime.now() - tic).total_seconds()
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
                        'init_time': init_time,
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
        
        # Log timing even if max iterations reached
        elapsed = (datetime.now() - tic).total_seconds()
        if iteration >= self.row_generation_cfg.max_iters and self.is_root():
            logger.info("Row generation reached max iterations (%d) in %.2f seconds.", iteration, elapsed)
            obj_val = self.master_model.ObjVal if hasattr(self.master_model, 'ObjVal') else None
            self._log_timing_summary(init_time, elapsed, iteration, timing_breakdown, obj_val, self.theta_val)
        
        # Store timing statistics for access later
        if self.is_root():
            total_pricing = np.sum(timing_breakdown.get('pricing', [0]))
            total_master = (np.sum(timing_breakdown.get('master_prep', [0])) + 
                          np.sum(timing_breakdown.get('master_update', [0])) + 
                          np.sum(timing_breakdown.get('master_optimize', [0])))
            total_mpi = (np.sum(timing_breakdown.get('mpi_gather', [0])) + 
                        np.sum(timing_breakdown.get('mpi_broadcast', [0])))
            
            self.timing_stats = {
                'total_time': elapsed,
                'init_time': init_time,
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
        
        self.theta_hat = self.theta_val.copy()
        
        # Create result object
        if self.is_root():
            obj_val = self.master_model.ObjVal if hasattr(self.master_model, 'ObjVal') else None
            converged = iteration < self.row_generation_cfg.max_iters
            result = EstimationResult(
                theta_hat=self.theta_hat.copy(),
                converged=converged,
                num_iterations=iteration + 1 if converged else iteration,
                final_objective=obj_val,
                timing=self.timing_stats,
                iteration_history=None,
                warnings=[],
                metadata={}
            )
        else:
            # Non-root ranks: theta_val is already broadcast, use it for consistency
            result = EstimationResult(
                theta_hat=self.theta_val.copy(),
                converged=iteration < self.row_generation_cfg.max_iters,
                num_iterations=iteration + 1 if iteration < self.row_generation_cfg.max_iters else iteration,
                final_objective=None,
                timing=None,
                iteration_history=None,
                warnings=[],
                metadata={}
            )
        
        return result

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
        import sys
        if self.is_root():
            print("=" * 70)
            print("ROW GENERATION SUMMARY")
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
            # Filter out non-time entries (sizes, bandwidth, etc.)
            time_only_keys = {'pricing', 'mpi_gather', 'master_prep', 'master_update', 
                            'master_optimize', 'mpi_broadcast', 'callback', 
                            'gather_bundles', 'gather_features', 'gather_errors',
                            'compute_features', 'compute_errors'}
            
            component_stats = []
            total_accounted = init_time
            
            for component, times in timing_breakdown.items():
                # Only process time-based metrics
                if component not in time_only_keys:
                    continue
                if len(times) > 0:
                    total = np.sum(times)
                    mean = np.mean(times)
                    std = np.std(times)
                    min_t = np.min(times)
                    max_t = np.max(times)
                    pct = 100 * total / total_time if total_time > 0 else 0
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

    def add_constraints(self, indices: NDArray[np.int64], bundles: NDArray[np.float64]) -> None:
        """
        Add constraints to the initialized master problem.
        
        Args:
            indices: Array of u variable indices (agent-simulation pairs)
            bundles: Array of bundles, shape (len(indices), num_items)
        """
        if not self.is_root() or self.master_model is None:
            return
        
        if len(indices) == 0:
            return
        
        theta, u = self.master_variables
        
        # Compute features and errors for each constraint
        for i, idx in enumerate(indices):
            agent_id = idx % self.num_agents
            sim_id = idx // self.num_agents
            bundle = bundles[i]
            
            # Compute features and errors
            features = self.feature_manager.features_oracle(agent_id, bundle, self.input_data)
            error = (self.input_data["errors"][sim_id, agent_id] * bundle).sum()
            
            # Encode bundle as binary string
            # Use hash of bundle for constraint name (avoids Gurobi 255 char limit)
            bundle_hash = hash(bundle.tobytes())
            constr_name = f"rowgen_{idx}_b{abs(bundle_hash) % 1000000000}"
            
            # Add constraint
            self.master_model.addConstr(u[idx] >= error + features @ theta, name=constr_name)
        
        logger.info("Added %d constraints to master problem", len(indices))

    def get_constraints(self) -> Optional[Dict[str, NDArray]]:
        """
        Extract constraints from the Gurobi model by parsing constraint names.
        
        Returns:
            Dict with keys 'indices', 'bundles' containing numpy arrays,
            or None if not on root process or model not initialized.
        """
        if not self.is_root() or self.master_model is None:
            return None
        
        indices = []
        bundles = []
        
        # Extract constraints by parsing names (handle both old format with bundle_binary and new format with hash)
        for constr in self.master_model.getConstrs():
            if constr.ConstrName and constr.ConstrName.startswith("rowgen_"):
                try:
                    # Parse: "rowgen_{idx}_bundle_{binary_string}" (old) or "rowgen_{idx}_b{hash}" (new)
                    if "_bundle_" in constr.ConstrName:
                        # Old format: extract bundle from binary string
                        parts = constr.ConstrName.split("_bundle_")
                        if len(parts) == 2:
                            idx = int(parts[0].split("_")[1])  # Extract idx from "rowgen_{idx}"
                            bundle_binary = parts[1]  # Binary string
                            # Convert binary string back to bundle array
                            bundle = np.array([int(b) for b in bundle_binary], dtype=np.float64)
                            # Verify bundle has correct length
                            if len(bundle) == self.num_items:
                                indices.append(idx)
                                bundles.append(bundle)
                    elif constr.ConstrName.startswith("rowgen_") and "_b" in constr.ConstrName:
                        # New format: extract bundle from constraint expression (u[idx] >= error + features @ theta)
                        # Parse idx from name: "rowgen_{idx}_b{hash}"
                        name_parts = constr.ConstrName.split("_")
                        if len(name_parts) >= 2:
                            idx = int(name_parts[1])
                            # Extract bundle from constraint: get the error term which encodes the bundle
                            # This is complex, so for now we skip bundles for hash-based names
                            # The constraint still works, we just can't extract the bundle for analysis
                            indices.append(idx)
                            # Use zeros as placeholder (bundle info lost in hash-based naming)
                            bundles.append(np.zeros(self.num_items, dtype=np.float64))
                except (ValueError, IndexError):
                    continue
        
        if len(indices) == 0:
            return {'indices': np.array([], dtype=np.int64),
                    'bundles': np.array([], dtype=np.float64).reshape(0, self.num_items)}
        
        return {
            'indices': np.array(indices, dtype=np.int64),
            'bundles': np.array(bundles, dtype=np.float64)
        }

    def get_binding_constraints(self, tolerance: float = 1e-6) -> Optional[Dict[str, NDArray]]:
        """
        Extract only binding constraints (slack ≈ 0) from the Gurobi model.
        
        Args:
            tolerance: Tolerance for considering a constraint binding (default: 1e-6)
        
        Returns:
            Dict with keys 'indices', 'bundles' containing numpy arrays,
            or None if not on root process or model not initialized.
        """
        if not self.is_root() or self.master_model is None:
            return None
        
        indices = []
        bundles = []
        
        # Extract only binding constraints by parsing names and checking slack
        for constr in self.master_model.getConstrs():
            if constr.ConstrName and constr.ConstrName.startswith("rowgen_"):
                # Check if constraint is binding (slack close to zero)
                if abs(constr.Slack) <= tolerance:
                    try:
                        # Parse: "rowgen_{idx}_bundle_{binary_string}" (old) or "rowgen_{idx}_b{hash}" (new)
                        if "_bundle_" in constr.ConstrName:
                            # Old format: extract bundle from binary string
                            parts = constr.ConstrName.split("_bundle_")
                            if len(parts) == 2:
                                idx = int(parts[0].split("_")[1])  # Extract idx from "rowgen_{idx}"
                                bundle_binary = parts[1]  # Binary string
                                # Convert binary string back to bundle array
                                bundle = np.array([int(b) for b in bundle_binary], dtype=np.float64)
                                # Verify bundle has correct length
                                if len(bundle) == self.num_items:
                                    indices.append(idx)
                                    bundles.append(bundle)
                        elif constr.ConstrName.startswith("rowgen_") and "_b" in constr.ConstrName:
                            # New format: extract idx only (bundle info lost in hash-based naming)
                            name_parts = constr.ConstrName.split("_")
                            if len(name_parts) >= 2:
                                idx = int(name_parts[1])
                                indices.append(idx)
                                # Use zeros as placeholder (bundle info lost in hash-based naming)
                                bundles.append(np.zeros(self.num_items, dtype=np.float64))
                    except (ValueError, IndexError):
                        continue
        
        if len(indices) == 0:
            return {'indices': np.array([], dtype=np.int64),
                    'bundles': np.array([], dtype=np.float64).reshape(0, self.num_items)}
        
        return {
            'indices': np.array(indices, dtype=np.int64),
            'bundles': np.array(bundles, dtype=np.float64)
        }

