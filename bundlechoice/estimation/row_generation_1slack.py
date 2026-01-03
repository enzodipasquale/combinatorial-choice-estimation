"""
Row generation solver with 1slack formulation for modular bundle choice estimation (v2).
This module implements a simplified row generation approach with a single scalar utility variable.
"""
import time
import numpy as np
from typing import Optional, Any
from numpy.typing import NDArray
import gurobipy as gp
from gurobipy import GRB
from bundlechoice.utils import get_logger, suppress_output, make_timing_stats
from .base import BaseEstimationManager
from .result import EstimationResult
logger = get_logger(__name__)


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
        Initialize the RowGeneration1SlackManager.

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
        self.timing_stats = None

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
                theta.lb = 0.0
            u_bar = self.master_model.addVar(obj=1, name='utility')
            
            self.master_model.optimize()
            logger.info("Master Initialized (1slack formulation)")
            self.master_variables = (theta, u_bar)
            self.theta_val = theta.X
            self.log_parameter()
        else:
            self.theta_val = np.empty(self.num_features, dtype=np.float64)

        self.theta_val = self.comm_manager.broadcast_array(self.theta_val, root=0)

    def _master_iteration(self, optimal_bundles: NDArray[np.float64]) -> bool:
        """Perform one iteration of master problem (1slack). Returns True if stopping criterion met."""
        x_sim = self.feature_manager.compute_gathered_features(optimal_bundles)
        errors_sim = self.feature_manager.compute_gathered_errors(optimal_bundles)
        
        stop = False
        if self.is_root():
            theta, u_bar = self.master_variables
            u_sim = (x_sim @ theta.X).sum() + errors_sim.sum()
            u_master = u_bar.X

            self.log_parameter()
            logger.info(f"ObjVal: {self.master_model.ObjVal}")
            reduced_cost = u_sim - u_master
            logger.info("Reduced cost: %s", reduced_cost)
            
            if reduced_cost < self.row_generation_cfg.tolerance_optimality:
                stop = True
            else:
                if u_sim > u_master * (1 + self.row_generation_cfg.tol_row_generation) + self.row_generation_cfg.tolerance_optimality:
                    agents_utilities = (x_sim @ theta).sum() + errors_sim.sum()
                    self.master_model.addConstr(u_bar >= agents_utilities)
                    self._enforce_slack_counter()
                    logger.info("Number of constraints: %d", self.master_model.NumConstrs)
                    self.master_model.optimize()
                    self.row_generation_cfg.tol_row_generation *= self.row_generation_cfg.row_generation_decay
            
            theta_val = theta.X
        else:
            theta_val = np.empty(self.num_features, dtype=np.float64)
            
        # Broadcast theta and stop flag
        self.theta_val, stop = self.comm_manager.broadcast_array_with_flag(theta_val, stop, root=0)
        
        return stop

    def solve(self) -> EstimationResult:
        """Run row generation with 1slack formulation. Returns EstimationResult with theta_hat and diagnostics."""
        if self.is_root():
            print("=" * 70)
            print("ROW GENERATION (1SLACK)")
            print("=" * 70)
            print()
            print(f"  Problem: {self.dimensions_cfg.num_agents} agents × {self.dimensions_cfg.num_items} items, {self.num_features} features")
            if self.dimensions_cfg.num_simulations > 1:
                print(f"  Simulations: {self.dimensions_cfg.num_simulations}")
            print(f"  Max iterations: {self.row_generation_cfg.max_iters if self.row_generation_cfg.max_iters != float('inf') else '∞'}")
            print(f"  Min iterations: {self.row_generation_cfg.min_iters}")
            print(f"  Optimality tolerance: {self.row_generation_cfg.tolerance_optimality}")
            if self.row_generation_cfg.max_slack_counter < float('inf'):
                print(f"  Max slack counter: {self.row_generation_cfg.max_slack_counter}")
            print()
            print("  Starting row generation algorithm (1slack formulation)...")
            print()
        
        tic = time.perf_counter()
        self.subproblem_manager.initialize_local()
        self._initialize_master_problem()        
        self.slack_counter = {}
        iteration = 0
        
        # Simple timing: only track pricing time
        total_pricing = 0.0
        
        while iteration < self.row_generation_cfg.max_iters:
            logger.info(f"ITERATION {iteration + 1}")
            
            # Pricing phase
            t0 = time.perf_counter()
            optimal_bundles = self.subproblem_manager.solve_local(self.theta_val)
            total_pricing += time.perf_counter() - t0
            
            # Master iteration
            stop = self._master_iteration(optimal_bundles)
            
            if stop and iteration >= self.row_generation_cfg.min_iters:
                break
            iteration += 1
        
        elapsed = time.perf_counter() - tic
        num_iters = iteration + 1
        
        if self.is_root():
            converged = iteration < self.row_generation_cfg.max_iters
            msg = "ended" if converged else "reached max iterations"
            logger.info(f"Row generation (1slack) {msg} after {num_iters} iterations in {elapsed:.2f} seconds.")
            obj_val = self.master_model.ObjVal if hasattr(self.master_model, 'ObjVal') else None
            self.timing_stats = make_timing_stats(elapsed, num_iters, total_pricing)
            self._log_timing_summary(self.timing_stats, obj_val, self.theta_val, header="ROW GENERATION (1-SLACK) SUMMARY")
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
                num_iterations=num_iters,
                final_objective=obj_val,
                timing=self.timing_stats,
                iteration_history=None,
                warnings=[],
                metadata={}
            )
        else:
            result = EstimationResult(
                theta_hat=self.theta_val.copy(),
                converged=iteration < self.row_generation_cfg.max_iters,
                num_iterations=num_iters,
                final_objective=None,
                timing=None,
                iteration_history=None,
                warnings=[],
                metadata={}
            )
        
        return result

    # _enforce_slack_counter, _log_timing_summary, log_parameter inherited from BaseEstimationManager
