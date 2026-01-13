"""
Row generation solver for modular bundle choice estimation (v2).
This module will be used by BundleChoice to estimate parameters using row generation.
Future solvers can be added to this folder as well.
"""
import time
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Any, Dict, Callable
import gurobipy as gp
from gurobipy import GRB
from bundlechoice.utils import get_logger, suppress_output, make_timing_stats
from .base import BaseEstimationManager
from .result import EstimationResult
logger = get_logger(__name__)


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
                oracles_manager: Any,
                subproblem_manager: Any
    ) -> None:
        """
        Initialize the RowGenerationManager.

        Args:
            comm_manager: Communication manager for MPI operations
            dimensions_cfg: DimensionsConfig instance
            row_generation_cfg: RowGenerationConfig instance
            data_manager: DataManager instance
            oracles_manager: OraclesManager instance
            subproblem_manager: SubproblemManager instance
        """
        super().__init__(
            comm_manager=comm_manager,
            dimensions_cfg=dimensions_cfg,
            data_manager=data_manager,
            oracles_manager=oracles_manager,
            subproblem_manager=subproblem_manager
        )
        
        self.row_generation_cfg = row_generation_cfg
        self.master_model = None
        self.master_variables = None
        self.timing_stats = None  # Store detailed timing statistics
        self.theta_val = None
        self.theta_hat = None
        self.slack_counter = None
        self.constraint_info = {}  # Map constraint objects to (idx, bundle) tuples

    def _check_bounds_hit(self, tolerance: float = 1e-6) -> Dict[str, Any]:
        """
        Check if any theta variable is at its bounds.
        
        Returns:
            Dict with 'hit_lower', 'hit_upper' (lists of indices), and 'any_hit' (bool)
        """
        if not self.comm_manager.is_root() or self.master_model is None:
            return {'hit_lower': [], 'hit_upper': [], 'any_hit': False}
        
        theta = self.master_variables[0]
        hit_lower, hit_upper = [], []
        
        for k in range(self.dimensions_cfg.num_features):
            val = theta[k].X
            lb, ub = theta[k].LB, theta[k].UB
            if lb > -GRB.INFINITY and abs(val - lb) < tolerance:
                hit_lower.append(k)
            if ub < GRB.INFINITY and abs(val - ub) < tolerance:
                hit_upper.append(k)
        
        return {'hit_lower': hit_lower, 'hit_upper': hit_upper, 'any_hit': bool(hit_lower or hit_upper)}

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
        # Compute observed features (weighted if agent_weights provided)
        if hasattr(self, '_agent_weights') and self._agent_weights is not None:
            # Weighted observed features: sum_i w_i * x_i
            # agents_obs_features has shape (num_agents * num_simulations, num_features)
            # so tile weights to match
            if self.comm_manager.is_root():
                weights_tiled = np.tile(self._agent_weights, self.dimensions_cfg.num_simulations)
                obs_features = (weights_tiled[:, None] * self.agents_obs_features).sum(0)
            else:
                obs_features = None
        else:
            obs_features = self.get_obs_features()
        
        if self.comm_manager.is_root():
            # Clear constraint info when creating a new model (old constraint objects become invalid)
            self.constraint_info.clear()
            self.master_model = self._setup_gurobi_model_params()    
            theta = self.master_model.addMVar(self.dimensions_cfg.num_features, obj= - obs_features, ub=self.row_generation_cfg.theta_ubs, name='parameter')
            if self.row_generation_cfg.theta_lbs is not None:
                # Set bounds per variable, None means unbounded (don't set)
                for k in range(self.dimensions_cfg.num_features):
                    if k < len(self.row_generation_cfg.theta_lbs) and self.row_generation_cfg.theta_lbs[k] is not None:
                        theta[k].lb = float(self.row_generation_cfg.theta_lbs[k])
            else:
                # Default: non-negativity constraints (lb=0 for all variables)
                theta.lb = 0.0
            
            # Set start value if theta_init was provided
            if hasattr(self, '_theta_init_for_start') and self._theta_init_for_start is not None:
                theta.Start = self._theta_init_for_start

            # Utility variables with (optionally weighted) objective coefficients
            if hasattr(self, '_agent_weights') and self._agent_weights is not None:
                # Weighted objective: w_i for each agent i (repeated for each simulation)
                u_obj = np.tile(self._agent_weights, self.dimensions_cfg.num_simulations)
                u = self.master_model.addMVar(self.dimensions_cfg.num_simulations * self.dimensions_cfg.num_agents, obj=u_obj, name='utility')
            else:
                u = self.master_model.addMVar(self.dimensions_cfg.num_simulations * self.dimensions_cfg.num_agents, obj=1, name='utility')
            
            # Add initial constraints if provided (warm-starting)
            if initial_constraints is not None and len(initial_constraints.get('indices', [])) > 0:
                indices = initial_constraints['indices']
                bundles = initial_constraints['bundles']
                errors = self.data_manager.input_data["errors"]
                has_sim_dim = errors.ndim == 3
                
                for i, idx in enumerate(indices):
                    agent_id = idx % self.dimensions_cfg.num_agents
                    sim_id = idx // self.dimensions_cfg.num_agents
                    bundle = bundles[i]
                    
                    features = self.oracles_manager.features_oracle(agent_id, bundle, self.data_manager.input_data)
                    if has_sim_dim:
                        error = (errors[sim_id, agent_id] * bundle).sum()
                    else:
                        error = (errors[agent_id] * bundle).sum()
                    
                    constr = self.master_model.addConstr(u[idx] >= error + features @ theta)
                    self.constraint_info[constr] = (idx, bundle.copy())
                
                logger.info("Added %d initial constraints for warm-start", len(indices))

            # Call master init callback if configured (for custom constraints)
            if self.row_generation_cfg.master_init_callback is not None:
                self.row_generation_cfg.master_init_callback(self.master_model, theta, u)

            self.master_model.optimize()
            logger.info("Master Initialized")
            self.master_variables = (theta, u)
            if self.master_model.Status == GRB.OPTIMAL:
                self.theta_val = theta.X
            else:
                logger.warning("Master problem not optimal at initialization, status=%s", self.master_model.Status)
                self.theta_val = np.zeros(self.dimensions_cfg.num_features, dtype=np.float64)
            self.log_parameter()
        else:
            self.theta_val = np.empty(self.dimensions_cfg.num_features, dtype=np.float64)
        
        self.theta_val = self.comm_manager.broadcast_array(self.theta_val, root=0)
    


    def _master_iteration(self, local_pricing_results: NDArray[np.float64]) -> bool:
        """
        Perform one iteration of master problem. Returns True if stopping criterion met.
        
        Uses distributed violation detection: each rank computes local violations,
        then only violation data is gathered to root (efficient when V << N).
        """
        from mpi4py import MPI
        
        # Compute local features and errors (no gather yet)
        features_local = self.oracles_manager.compute_rank_features(local_pricing_results)
        errors_local = self.oracles_manager.compute_rank_errors(local_pricing_results)
        
        # Get global indices and per-rank counts (stored during scatter)
        global_indices_local = self.data_manager.local_data["global_indices"]
        all_counts = self.data_manager.local_data["agent_counts"]
        
        # Scatter u_master from root to all ranks using consistent counts
        if self.comm_manager.is_root():
            theta, u = self.master_variables
            u_master_all = u.X
            theta_current = theta.X
        else:
            u_master_all = np.empty(self.dimensions_cfg.num_simulations * self.dimensions_cfg.num_agents, dtype=np.float64)
            theta_current = np.empty(self.dimensions_cfg.num_features, dtype=np.float64)
        
        u_master_local = self.comm_manager.scatter_array(u_master_all, counts=all_counts, root=0, dtype=np.float64)
        theta_current = self.comm_manager.broadcast_array(theta_current, root=0)
        
        # Compute local utilities
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            u_sim_local = features_local @ theta_current + errors_local
        
        # Local violation check (demand oracle consistency)
        local_violations = np.where(
            ~np.isclose(u_master_local, u_sim_local, rtol=1e-5, atol=1e-5) & (u_master_local > u_sim_local)
        )[0]
        if len(local_violations) > 0:
            logger.warning(
                "Rank %d: Possible failure of demand oracle at local ids: %s",
                self.comm_manager.rank, local_violations[:10]  # Limit output
            )
        
        # Compute max reduced cost via Allreduce
        reduced_costs_local = u_sim_local - u_master_local
        local_max_rc = reduced_costs_local.max() if len(reduced_costs_local) > 0 else -np.inf
        max_reduced_cost = self.comm_manager.comm.allreduce(local_max_rc, op=MPI.MAX)
        
        # Find local rows to add (violations that exceed threshold)
        tol_opt = self.row_generation_cfg.tolerance_optimality
        tol_rg = self.row_generation_cfg.tol_row_generation
        local_rows_to_add = np.where(
            u_sim_local > u_master_local * (1 + tol_rg) + tol_opt
        )[0]
        
        # Gather only violation data (efficient when V << N)
        viol_global_ids = global_indices_local[local_rows_to_add]
        viol_bundles = local_pricing_results[local_rows_to_add]
        viol_features = features_local[local_rows_to_add]
        viol_errors = errors_local[local_rows_to_add]
        
        all_viol_ids = self.comm_manager.concatenate_array_at_root_fast(viol_global_ids, root=0)
        all_viol_bundles = self.comm_manager.concatenate_array_at_root_fast(viol_bundles, root=0)
        all_viol_features = self.comm_manager.concatenate_array_at_root_fast(viol_features, root=0)
        all_viol_errors = self.comm_manager.concatenate_array_at_root_fast(viol_errors, root=0)
        
        stop = False
        if self.comm_manager.is_root():
            self.log_parameter()
            logger.info(f"ObjVal: {self.master_model.ObjVal}")
            logger.info("Reduced cost: %s", max_reduced_cost)
            
            # Check if we're in suboptimal cuts mode (set by callback)
            suboptimal_mode = getattr(self.subproblem_manager, '_suboptimal_mode', False)
            if max_reduced_cost < tol_opt:
                if not suboptimal_mode:
                    stop = True
                else:
                    logger.info("Reduced cost below tolerance, but suboptimal cuts mode active - continuing")
            
            num_new = len(all_viol_ids) if all_viol_ids is not None else 0
            logger.info("New constraints: %d", num_new)
            
            # Add constraints using gathered violation data
            if num_new > 0:
                for i in range(num_new):
                    idx = int(all_viol_ids[i])
                    constr = self.master_model.addConstr(
                        u[idx] >= all_viol_errors[i] + all_viol_features[i] @ theta
                    )
                    self.constraint_info[constr] = (idx, all_viol_bundles[i].copy())
            
            self._enforce_slack_counter()
            logger.info("Number of constraints: %d", self.master_model.NumConstrs)
            
            self.master_model.optimize()
            
            theta_val = theta.X
            self.row_generation_cfg.tol_row_generation *= self.row_generation_cfg.row_generation_decay
        else:
            theta_val = None
            stop = False
        
        # Broadcast theta and stop flag using buffer-based method
        if self.comm_manager.is_root():
            theta_to_broadcast = theta_val
        else:
            theta_to_broadcast = np.empty(self.dimensions_cfg.num_features, dtype=np.float64)
        self.theta_val, stop = self.comm_manager.broadcast_array_with_flag(theta_to_broadcast, stop, root=0)
        
        return stop

    def solve(self, callback: Optional[Callable[[Dict[str, Any]], None]] = None,
              theta_init: Optional[NDArray[np.float64]] = None,
              agent_weights: Optional[NDArray[np.float64]] = None,
              initial_constraints: Optional[Dict[str, NDArray]] = None) -> EstimationResult:
        """
        Run the row generation algorithm to estimate model parameters.

        Args:
            callback: Optional callback function called after each iteration.
            theta_init: Optional initial parameter vector for warm-start.
            agent_weights: Optional per-agent weights for Bayesian bootstrap. Shape (num_agents,).
            initial_constraints: Optional dict with 'indices' and 'bundles' for warm-starting.
                               Use get_constraints() from a previous solve to pass here.
        
        Returns:
            EstimationResult: Result object containing theta_hat and diagnostics.
        """
        if self.comm_manager.is_root():
            lines = ["=" * 70, "ROW GENERATION", "=" * 70, ""]
            lines.append(f"  Problem: {self.dimensions_cfg.num_agents} agents × {self.dimensions_cfg.num_items} items, {self.dimensions_cfg.num_features} features")
            if self.dimensions_cfg.num_simulations > 1:
                lines.append(f"  Simulations: {self.dimensions_cfg.num_simulations}")
            lines.append(f"  Max iterations: {self.row_generation_cfg.max_iters if self.row_generation_cfg.max_iters != float('inf') else '∞'}")
            lines.append(f"  Min iterations: {self.row_generation_cfg.min_iters}")
            lines.append(f"  Optimality tolerance: {self.row_generation_cfg.tolerance_optimality}")
            if self.row_generation_cfg.max_slack_counter < float('inf'):
                lines.append(f"  Max slack counter: {self.row_generation_cfg.max_slack_counter}")
            if self.row_generation_cfg.tol_row_generation > 0:
                lines.append(f"  Row generation tolerance: {self.row_generation_cfg.tol_row_generation}")
            if self.row_generation_cfg.row_generation_decay > 0:
                lines.append(f"  Tolerance decay: {self.row_generation_cfg.row_generation_decay}")
            lines.append("")
            lines.append("  Starting row generation algorithm...")
            if agent_weights is not None:
                lines.append("  Using agent weights (Bayesian bootstrap)")
            lines.append("")
            logger.info("\n".join(lines))
        
        # Store agent weights (broadcast if needed)
        if agent_weights is not None:
            self._agent_weights = self.comm_manager.broadcast_array(
                np.asarray(agent_weights, dtype=np.float64) if self.comm_manager.is_root() else np.empty(self.dimensions_cfg.num_agents),
                root=0
            )
        else:
            self._agent_weights = None
        
        tic = time.perf_counter()
        self.subproblem_manager.initialize_local()
        
        # Use provided initial_constraints or compute from theta_init
        if initial_constraints is not None:
            if self.comm_manager.is_root():
                n_init = len(initial_constraints.get('indices', []))
                logger.info("Using %d provided initial constraints (warm start)", n_init)
        elif theta_init is not None:
            if self.comm_manager.is_root():
                logger.info("Initializing with provided theta (warm start)")
                if hasattr(theta_init, 'theta_hat'):
                    theta_init_array = theta_init.theta_hat
                else:
                    theta_init_array = theta_init
                self.theta_val = np.asarray(theta_init_array, dtype=np.float64).copy()
            else:
                self.theta_val = np.empty(self.dimensions_cfg.num_features, dtype=np.float64)
            self.theta_val = self.comm_manager.broadcast_array(self.theta_val, root=0)
            
            local_pricing_results = self.subproblem_manager.solve_local(self.theta_val)
            bundles_sim = self.comm_manager.concatenate_array_at_root_fast(local_pricing_results, root=0)
            
            if self.comm_manager.is_root() and bundles_sim is not None and len(bundles_sim) > 0:
                indices = np.arange(self.dimensions_cfg.num_simulations * self.dimensions_cfg.num_agents, dtype=np.int64)
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
        iteration = 0
        
        # Track per-iteration times
        pricing_times = []
        master_times = []
        
        while iteration < self.row_generation_cfg.max_iters:
            logger.info(f"ITERATION {iteration + 1}")
            
            # Subproblem callback (if configured)
            if self.row_generation_cfg.subproblem_callback is not None:
                master_model = self.master_model if self.comm_manager.is_root() else None
                self.row_generation_cfg.subproblem_callback(
                    iteration, 
                    self.subproblem_manager, 
                    master_model
                )
            
            # Pricing phase
            t0 = time.perf_counter()
            local_pricing_results = self.subproblem_manager.solve_local(self.theta_val)
            pricing_time = time.perf_counter() - t0
            pricing_times.append(pricing_time)
            
            # Master iteration
            t1 = time.perf_counter()
            stop = self._master_iteration(local_pricing_results)
            master_time = time.perf_counter() - t1
            master_times.append(master_time)
            
            # Callback
            if callback and self.comm_manager.is_root():
                callback({
                    'iteration': iteration + 1,
                    'theta': self.theta_val.copy() if self.theta_val is not None else None,
                    'objective': self.master_model.ObjVal if hasattr(self.master_model, 'ObjVal') else None,
                    'pricing_time': pricing_time,
                    'master_time': master_time,
                })
            
            if stop and iteration >= self.row_generation_cfg.min_iters:
                break
            iteration += 1
        
        elapsed = time.perf_counter() - tic
        num_iters = iteration + 1
        converged = iteration < self.row_generation_cfg.max_iters
        
        # Check bounds
        bounds_info = self._check_bounds_hit()
        warnings_list = []
        if self.comm_manager.is_root() and bounds_info['any_hit']:
            if bounds_info['hit_lower']:
                msg = f"Theta hit LOWER bound at indices: {bounds_info['hit_lower']}"
                logger.warning(msg)
                warnings_list.append(msg)
            if bounds_info['hit_upper']:
                msg = f"Theta hit UPPER bound at indices: {bounds_info['hit_upper']}"
                logger.warning(msg)
                warnings_list.append(msg)
        
        if self.comm_manager.is_root():
            msg = "ended" if converged else "reached max iterations"
            logger.info(f"Row generation {msg} after {num_iters} iterations in {elapsed:.2f} seconds.")
            obj_val = self.master_model.ObjVal if hasattr(self.master_model, 'ObjVal') else None
            self.timing_stats = make_timing_stats(elapsed, num_iters, pricing_times, master_times)
            self._log_timing_summary(self.timing_stats, obj_val, self.theta_val, header="ROW GENERATION SUMMARY")
        else:
            obj_val = None
            self.timing_stats = None
        
        self.theta_hat = self.theta_val.copy()
        result = self._create_result(self.theta_hat, converged, num_iters, obj_val)
        result.warnings.extend(warnings_list)
        result.metadata['bounds_hit'] = bounds_info
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
                # Also remove from constraint_info (RowGenerationManager specific)
                if hasattr(self, 'constraint_info'):
                    self.constraint_info.pop(constr, None)
            num_removed = len(to_remove)
            if num_removed > 0:
                logger.info("Removed %d slack constraints", num_removed)
            return num_removed
        return 0


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
            features = self.oracles_manager.features_oracle(agent_id, bundle, self.input_data)
            error = (self.input_data["errors"][sim_id, agent_id] * bundle).sum()
            
            # Add constraint (no name needed, store info in mapping)
            constr = self.master_model.addConstr(u[idx] >= error + features @ theta)
            self.constraint_info[constr] = (idx, bundle.copy())
        
        logger.info("Added %d constraints to master problem", len(indices))

    def get_constraints(self) -> Optional[Dict[str, NDArray]]:
        """
        Extract constraints from the Gurobi model using constraint info mapping.
        
        Returns:
            Dict with keys 'indices', 'bundles' containing numpy arrays,
            or None if not on root process or model not initialized.
        """
        if not self.comm_manager.is_root() or self.master_model is None:
            return None
        
        indices = []
        bundles = []
        
        # Use constraint_info directly (faster and more reliable)
        for constr, (idx, bundle) in self.constraint_info.items():
            indices.append(idx)
            bundles.append(bundle)
        
        logger.debug("get_constraints: extracted %d constraints from constraint_info", len(indices))
        
        if len(indices) == 0:
            return {'indices': np.array([], dtype=np.int64),
                    'bundles': np.array([], dtype=np.float64).reshape(0, self.dimensions_cfg.num_items)}
        
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
        if not self.comm_manager.is_root() or self.master_model is None:
            return None
        
        indices = []
        bundles = []
        
        # Extract only binding constraints from mapping
        for constr in self.master_model.getConstrs():
            if constr in self.constraint_info:
                # Check if constraint is binding (slack close to zero)
                if abs(constr.Slack) <= tolerance:
                    idx, bundle = self.constraint_info[constr]
                    indices.append(idx)
                    bundles.append(bundle)
        
        if len(indices) == 0:
            return {'indices': np.array([], dtype=np.int64),
                    'bundles': np.array([], dtype=np.float64).reshape(0, self.dimensions_cfg.num_items)}
        
        return {
            'indices': np.array(indices, dtype=np.int64),
            'bundles': np.array(bundles, dtype=np.float64)
        }

    def strip_slack_constraints(self, tolerance: float = 1e-6) -> int:
        """
        Remove non-binding (slack) constraints from the model in-place.
        Keeps only constraints with slack ≈ 0.
        
        Args:
            tolerance: Tolerance for considering a constraint binding
        
        Returns:
            Number of constraints removed
        """
        if not self.comm_manager.is_root() or self.master_model is None:
            return 0
        
        to_remove = []
        for constr in self.master_model.getConstrs():
            if constr in self.constraint_info:
                if abs(constr.Slack) > tolerance:
                    to_remove.append(constr)
        
        for constr in to_remove:
            self.master_model.remove(constr)
            self.constraint_info.pop(constr, None)
            self.slack_counter.pop(constr, None)
        
        if len(to_remove) > 0:
            self.master_model.update()
            logger.info("Stripped %d slack constraints, %d binding remain", 
                       len(to_remove), self.master_model.NumConstrs)
        
        return len(to_remove)

    def update_objective_for_weights(self, agent_weights: NDArray[np.float64]) -> None:
        """
        Update objective coefficients for new agent weights without rebuilding the model.
        This enables true Gurobi warm-start by reusing the LP basis.
        
        Args:
            agent_weights: New agent weights, shape (num_agents,)
        """
        if not self.comm_manager.is_root() or self.master_model is None:
            return
        
        theta, u = self.master_variables
        
        # Update theta objective: -sum_i w_i * x_obs_i
        # agents_obs_features has shape (num_agents * num_simulations, num_features)
        # so tile weights to match
        weights_tiled = np.tile(agent_weights, self.dimensions_cfg.num_simulations)
        obs_features = (weights_tiled[:, None] * self.agents_obs_features).sum(0)
        for k in range(self.dimensions_cfg.num_features):
            theta[k].Obj = -obs_features[k]
        
        # Update u objective: w_i for each agent (repeated for simulations)
        u_obj = np.tile(agent_weights, self.dimensions_cfg.num_simulations)
        for i in range(len(u_obj)):
            u[i].Obj = u_obj[i]
        
        self.master_model.update()

    def solve_reuse_model(self, agent_weights: NDArray[np.float64],
                          strip_slack: bool = False) -> EstimationResult:
        """
        Solve using existing model with updated weights (true Gurobi warm-start).
        
        This method reuses the existing Gurobi model and LP basis, only updating
        the objective coefficients. This is much faster than rebuilding the model.
        
        Args:
            agent_weights: Per-agent weights for the new solve
            strip_slack: If True, remove non-binding constraints before solving
        
        Returns:
            EstimationResult with theta_hat and diagnostics
        """
        # Store and broadcast weights
        self._agent_weights = self.comm_manager.broadcast_array(
            np.asarray(agent_weights, dtype=np.float64) if self.comm_manager.is_root() else np.empty(self.dimensions_cfg.num_agents),
            root=0
        )
        
        tic = time.perf_counter()
        self.subproblem_manager.initialize_local()
        
        if self.comm_manager.is_root():
            if self.master_model is None:
                raise RuntimeError("No existing model to reuse. Call solve() first.")
            
            # Optionally strip slack constraints
            if strip_slack:
                self.strip_slack_constraints()
            
            # Update objective coefficients in-place
            self.update_objective_for_weights(self._agent_weights)
            
            # Reset LP basis so Gurobi actually re-solves with new objective
            # reset(0) discards solution; this forces Gurobi to re-optimize
            self.master_model.reset(0)
            self.master_model.optimize()
            
            theta, u = self.master_variables
            if self.master_model.Status == GRB.OPTIMAL:
                self.theta_val = theta.X
            else:
                self.theta_val = np.zeros(self.dimensions_cfg.num_features, dtype=np.float64)
        else:
            self.theta_val = np.empty(self.dimensions_cfg.num_features, dtype=np.float64)
        
        self.theta_val = self.comm_manager.broadcast_array(self.theta_val, root=0)
        
        # Run row generation iterations
        iteration = 0
        while iteration < self.row_generation_cfg.max_iters:
            local_pricing_results = self.subproblem_manager.solve_local(self.theta_val)
            stop = self._master_iteration(local_pricing_results)
            if stop and iteration >= self.row_generation_cfg.min_iters:
                break
            iteration += 1
        
        toc = time.perf_counter()
        
        # Check bounds
        bounds_info = self._check_bounds_hit()
        warnings_list = []
        if self.comm_manager.is_root() and bounds_info['any_hit']:
            if bounds_info['hit_lower']:
                msg = f"Theta hit LOWER bound at indices: {bounds_info['hit_lower']}"
                logger.warning(msg)
                warnings_list.append(msg)
            if bounds_info['hit_upper']:
                msg = f"Theta hit UPPER bound at indices: {bounds_info['hit_upper']}"
                logger.warning(msg)
                warnings_list.append(msg)
        
        # Build result
        if self.comm_manager.is_root():
            result = EstimationResult(
                theta_hat=self.theta_val.copy(),
                converged=iteration < self.row_generation_cfg.max_iters,
                num_iterations=iteration + 1,
                final_objective=self.master_model.ObjVal if self.master_model.Status == GRB.OPTIMAL else float('inf'),
                timing={'total': toc - tic},
            )
            result.warnings.extend(warnings_list)
            result.metadata['bounds_hit'] = bounds_info
            return result
        return EstimationResult(
            theta_hat=self.theta_val.copy(),
            converged=True,
            num_iterations=iteration + 1,
            metadata={'bounds_hit': {'hit_lower': [], 'hit_upper': [], 'any_hit': False}},
        )