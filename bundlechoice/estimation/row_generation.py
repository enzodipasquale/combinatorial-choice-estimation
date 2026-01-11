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
        self.constraint_info = {}  # Map constraint objects to (idx, bundle) tuples

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
            if self.is_root():
                obs_features = (self._agent_weights[:, None] * self.agents_obs_features).sum(0)
            else:
                obs_features = None
        else:
            obs_features = self.get_obs_features()
        
        if self.is_root():
            # Clear constraint info when creating a new model (old constraint objects become invalid)
            self.constraint_info.clear()
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

            # Utility variables with (optionally weighted) objective coefficients
            if hasattr(self, '_agent_weights') and self._agent_weights is not None:
                # Weighted objective: w_i for each agent i (repeated for each simulation)
                u_obj = np.tile(self._agent_weights, self.num_simulations)
                u = self.master_model.addMVar(self.num_simulations * self.num_agents, obj=u_obj, name='utility')
            else:
                u = self.master_model.addMVar(self.num_simulations * self.num_agents, obj=1, name='utility')
            
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
                    
                    # Add constraint (no name needed, store info in mapping)
                    constr = self.master_model.addConstr(u[idx] >= error + features @ theta)
                    self.constraint_info[constr] = (idx, bundle.copy())
                
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
    


    def _master_iteration(self, local_pricing_results: NDArray[np.float64]) -> bool:
        """Perform one iteration of master problem. Returns True if stopping criterion met."""
        # Gather bundles
        bundles_sim = self.comm_manager.concatenate_array_at_root_fast(local_pricing_results, root=0)
        
        # Compute and gather features
        features_local = self.feature_manager.compute_rank_features(local_pricing_results)
        x_sim = self.comm_manager.concatenate_array_at_root_fast(features_local, root=0)
        
        # Compute and gather errors
        errors_local = (self.data_manager.local_data["errors"] * local_pricing_results).sum(1)
        errors_sim = self.comm_manager.concatenate_array_at_root_fast(errors_local, root=0)
        
        stop = False
        if self.is_root():
            theta, u = self.master_variables
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                u_sim = x_sim @ theta.X + errors_sim
            u_master = u.X

            violations = np.where(~np.isclose(u_master, u_sim, rtol=1e-5, atol=1e-5) * (u_master > u_sim))[0]
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
            
            # Add constraints
            if len(rows_to_add) > 0 and bundles_sim is not None:
                for idx in rows_to_add:
                    constr = self.master_model.addConstr(u[idx] >= errors_sim[idx] + x_sim[idx] @ theta)
                    self.constraint_info[constr] = (idx, bundles_sim[idx].copy())
            self._enforce_slack_counter()
            logger.info("Number of constraints: %d", self.master_model.NumConstrs)
            
            self.master_model.optimize()
            
            theta_val = theta.X
            self.row_generation_cfg.tol_row_generation *= self.row_generation_cfg.row_generation_decay
        else:
            theta_val = None
            stop = False
        
        # Broadcast theta and stop flag
        self.theta_val, stop = self.comm_manager.broadcast_from_root((theta_val, stop), root=0)
        
        return stop

    def solve(self, callback: Optional[Callable[[Dict[str, Any]], None]] = None,
              theta_init: Optional[NDArray[np.float64]] = None,
              agent_weights: Optional[NDArray[np.float64]] = None) -> EstimationResult:
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
            agent_weights: Optional per-agent weights for Bayesian bootstrap. Shape (num_agents,).
                          If None, uniform weights (1.0) are used.
        
        Returns:
            EstimationResult: Result object containing theta_hat and diagnostics.
        """
        if self.is_root():
            print("=" * 70)
            print("ROW GENERATION")
            print("=" * 70)
            print()  # Blank line after header
            print(f"  Problem: {self.dimensions_cfg.num_agents} agents × {self.dimensions_cfg.num_items} items, {self.num_features} features")
            if self.dimensions_cfg.num_simulations > 1:
                print(f"  Simulations: {self.dimensions_cfg.num_simulations}")
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
            if agent_weights is not None:
                print("  Using agent weights (Bayesian bootstrap)")
            print()  # Blank line before iterations
        
        # Store agent weights (broadcast if needed)
        if agent_weights is not None:
            self._agent_weights = self.comm_manager.broadcast_array(
                np.asarray(agent_weights, dtype=np.float64) if self.is_root() else np.empty(self.num_agents),
                root=0
            )
        else:
            self._agent_weights = None
        
        tic = time.perf_counter()
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
                indices = np.arange(self.num_simulations * self.num_agents, dtype=np.int64)
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
                master_model = self.master_model if self.is_root() else None
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
            if callback and self.is_root():
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
        
        if self.is_root():
            converged = iteration < self.row_generation_cfg.max_iters
            msg = "ended" if converged else "reached max iterations"
            logger.info(f"Row generation {msg} after {num_iters} iterations in {elapsed:.2f} seconds.")
            obj_val = self.master_model.ObjVal if hasattr(self.master_model, 'ObjVal') else None
            self.timing_stats = make_timing_stats(elapsed, num_iters, pricing_times, master_times)
            self._log_timing_summary(self.timing_stats, obj_val, self.theta_val, header="ROW GENERATION SUMMARY")
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
            features = self.feature_manager.features_oracle(agent_id, bundle, self.input_data)
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
        if not self.is_root() or self.master_model is None:
            return None
        
        indices = []
        bundles = []
        
        # Extract constraints from mapping
        for constr in self.master_model.getConstrs():
            if constr in self.constraint_info:
                idx, bundle = self.constraint_info[constr]
                indices.append(idx)
                bundles.append(bundle)
        
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
                    'bundles': np.array([], dtype=np.float64).reshape(0, self.num_items)}
        
        return {
            'indices': np.array(indices, dtype=np.int64),
            'bundles': np.array(bundles, dtype=np.float64)
        }

