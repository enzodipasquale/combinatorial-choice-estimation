"""
Base estimation solver for bundle choice estimation.

Provides common functionality for row generation, ellipsoid, and other solvers.
"""

import numpy as np
from typing import Any, Optional, Tuple, Dict, List
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

    def _log_timing_summary(self, init_time: float, total_time: float, 
                           num_iterations: int, timing_breakdown: Dict[str, List[float]],
                           obj_val: Optional[float] = None, theta: Optional[NDArray[np.float64]] = None,
                           header_suffix: str = "") -> None:
        """Log comprehensive timing summary showing bottlenecks."""
        if self.is_root():
            print("=" * 70)
            print(f"ROW GENERATION SUMMARY{header_suffix}")
            print("=" * 70)
            
            if obj_val is not None:
                print(f"Objective value at solution: {obj_val:.6f}")
            if theta is not None:
                if len(theta) <= 10:
                    print(f"Theta at solution: {np.array2string(theta, precision=6, suppress_small=True)}")
                else:
                    print(f"Theta at solution (dim={len(theta)}):")
                    print(f"  First 5: {np.array2string(theta[:5], precision=6, suppress_small=True)}")
                    print(f"  Last 5:  {np.array2string(theta[-5:], precision=6, suppress_small=True)}")
                    print(f"  Min: {theta.min():.6f}, Max: {theta.max():.6f}, Mean: {theta.mean():.6f}")
            
            print(f"Total iterations: {num_iterations}")
            print(f"Total time: {total_time:.2f}s")
            print()
            print("Timing Statistics:")
            
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
                        'name': component, 'total': total, 'mean': mean,
                        'std': std, 'min': min_t, 'max': max_t, 'pct': pct
                    })
            
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
