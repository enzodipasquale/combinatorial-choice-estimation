"""
Base estimation solver for bundle choice estimation.

Provides common functionality for row generation, ellipsoid, and other solvers.
"""

import numpy as np
from typing import Any, Optional, Tuple, Dict
from numpy.typing import NDArray
from bundlechoice.base import HasDimensions, HasData, HasComm
from bundlechoice.utils import get_logger, extract_theta

logger = get_logger(__name__)


# ============================================================================
# Base Estimation Solver
# ============================================================================

class BaseEstimationManager(HasDimensions, HasData, HasComm):
    """Base class for estimation managers (row generation, ellipsoid, etc.)."""
    
    # Subclasses should set this for their row_generation config
    row_generation_cfg: Any = None
    master_model: Any = None
    slack_counter: Optional[Dict] = None
    theta_val: Optional[NDArray[np.float64]] = None
    
    def __init__(self, comm_manager: Any, dimensions_cfg: Any, data_manager: Any,
                 feature_manager: Any, subproblem_manager: Any) -> None:
        """Initialize base estimation solver."""
        self.comm_manager = comm_manager
        self.dimensions_cfg = dimensions_cfg
        self.data_manager = data_manager
        self.feature_manager = feature_manager
        self.subproblem_manager = subproblem_manager

        self.agents_obs_features = self.get_agents_obs_features()
        self.obs_features = self.agents_obs_features.sum(0) if self.agents_obs_features is not None else None

    # ============================================================================
    # Observed Features
    # ============================================================================

    def get_obs_features(self) -> Optional[NDArray[np.float64]]:
        """Compute aggregate observed features (rank 0 only)."""
        local_bundles = self.local_data.get("obs_bundles")
        agents_obs_features = self.feature_manager.compute_gathered_features(local_bundles)
        if self.is_root():
            return agents_obs_features.sum(0)
        return None

    def get_agents_obs_features(self) -> Optional[NDArray[np.float64]]:
        """Compute per-agent observed features (rank 0 only)."""
        local_bundles = self.local_data.get("obs_bundles")
        agents_obs_features = self.feature_manager.compute_gathered_features(local_bundles)
        return agents_obs_features if self.is_root() else None

    # ============================================================================
    # Objective & Gradient
    # ============================================================================

    def compute_obj_and_gradient(self, theta: NDArray[np.float64]) -> Tuple[Optional[float], Optional[NDArray[np.float64]]]:
        """Compute objective and gradient in one call (avoids duplicate subproblem solves)."""
        B_local = self.subproblem_manager.solve_local(theta)
        agents_features = self.feature_manager.compute_gathered_features(B_local)
        utilities = self.feature_manager.compute_gathered_utilities(B_local, theta)
        
        if self.is_root():
            obj_value = utilities.sum() - (self.obs_features @ theta).sum()
            gradient = (agents_features.sum(0) - self.obs_features) / self.num_agents
            return obj_value, gradient
        return None, None

    def objective(self, theta: NDArray[np.float64]) -> Optional[float]:
        """Compute objective function value."""
        theta = extract_theta(theta)
        if theta.ndim == 0:
            raise ValueError(f"theta must be 1D array, got scalar or 0D array")
        B_local = self.subproblem_manager.solve_local(theta)
        utilities = self.feature_manager.compute_gathered_utilities(B_local, theta)
        if self.is_root():
            return utilities.sum() - (self.obs_features @ theta).sum()
        return None
    
    def obj_gradient(self, theta: NDArray[np.float64]) -> Optional[NDArray[np.float64]]:
        """Compute objective gradient."""
        B_local = self.subproblem_manager.solve_local(theta)
        agents_features = self.feature_manager.compute_gathered_features(B_local)
        if self.is_root():
            return (agents_features.sum(0) - self.obs_features) / self.num_agents
        return None

    # ============================================================================
    # Abstract Solve Method
    # ============================================================================

    def solve(self):
        """Main solve method (implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement the solve method")

    # ============================================================================
    # Shared Row Generation Methods
    # ============================================================================

    def _enforce_slack_counter(self) -> int:
        """Update slack counter and remove constraints that have been slack too long. Returns number removed."""
        if self.row_generation_cfg is None or self.master_model is None:
            return 0
        if self.row_generation_cfg.max_slack_counter < float('inf'):
            if self.slack_counter is None:
                self.slack_counter = {}
            to_remove = []
            for constr in self.master_model.getConstrs():
                if constr.Slack < -1e-6:
                    if constr not in self.slack_counter:
                        self.slack_counter[constr] = 0
                    self.slack_counter[constr] += 1
                    if self.slack_counter[constr] >= self.row_generation_cfg.max_slack_counter:
                        to_remove.append(constr)
                if constr.Pi > 1e-6:
                    self.slack_counter.pop(constr, None)
            for constr in to_remove:
                self.master_model.remove(constr)
                self.slack_counter.pop(constr, None)
            num_removed = len(to_remove)
            logger.info("Removed constraints: %d", num_removed)
            return num_removed
        return 0

    def _log_timing_summary(self, timing_stats: Dict[str, Any],
                           obj_val: Optional[float] = None, 
                           theta: Optional[NDArray[np.float64]] = None,
                           header: str = "ESTIMATION SUMMARY") -> None:
        """Log timing summary from timing_stats dict (created by make_timing_stats)."""
        if not self.is_root():
            return
        
        total_time = timing_stats.get('total_time', 0.0)
        num_iters = timing_stats.get('num_iterations', 0)
        time_per_iter = timing_stats.get('time_per_iter', total_time / num_iters if num_iters > 0 else 0)
        pricing = timing_stats.get('pricing_time', 0.0)
        other = timing_stats.get('other_time', total_time - pricing)
        pricing_pct = timing_stats.get('pricing_pct', 100 * pricing / total_time if total_time > 0 else 0)
        other_pct = timing_stats.get('other_pct', 100 * other / total_time if total_time > 0 else 0)
            
        print("=" * 70)
        print(header)
        print("=" * 70)
        
        if obj_val is not None:
            print(f"Objective value: {obj_val:.6f}")
        if theta is not None:
            if len(theta) <= 10:
                print(f"Theta: {np.array2string(theta, precision=6, suppress_small=True)}")
            else:
                print(f"Theta (dim={len(theta)}):")
                print(f"  First 5: {np.array2string(theta[:5], precision=6, suppress_small=True)}")
                print(f"  Last 5:  {np.array2string(theta[-5:], precision=6, suppress_small=True)}")
                print(f"  Min: {theta.min():.6f}, Max: {theta.max():.6f}, Mean: {theta.mean():.6f}")
        
        print(f"Iterations: {num_iters}")
        print(f"Total time: {total_time:.2f}s ({time_per_iter:.2f}s/iter)")
        print()
        print("Timing (root process):")
        print(f"  Pricing (subproblems): {pricing:7.2f}s ({pricing_pct:5.1f}%)")
        print(f"  Other (master + sync): {other:7.2f}s ({other_pct:5.1f}%)")
        print()

    def log_parameter(self) -> None:
        """Log current parameter values (if parameters_to_log is set in config)."""
        if self.row_generation_cfg is None or self.theta_val is None:
            return
        feature_ids = self.row_generation_cfg.parameters_to_log
        precision = 3
        if feature_ids is not None:
            logger.info("Parameters: %s", np.round(self.theta_val[feature_ids], precision))
        else:
            logger.info("Parameters: %s", np.round(self.theta_val, precision))


# Alias for backward compatibility
BaseEstimationSolver = BaseEstimationManager
