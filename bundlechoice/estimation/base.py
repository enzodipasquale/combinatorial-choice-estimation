"""
Base estimation solver for bundle choice estimation.

Provides common functionality for row generation, ellipsoid, and other solvers.
"""

import numpy as np
from typing import Optional, Tuple, Dict, List, Any, TYPE_CHECKING
from numpy.typing import NDArray
from bundlechoice.base import HasDimensions, HasData, HasComm
from bundlechoice.utils import get_logger, extract_theta
from .result import EstimationResult

if TYPE_CHECKING:
    from bundlechoice.comm_manager import CommManager
    from bundlechoice.config import DimensionsConfig, RowGenerationConfig
    from bundlechoice.data_manager import DataManager
    from bundlechoice.feature_manager import FeatureManager
    from bundlechoice.subproblems.subproblem_manager import SubproblemManager

logger = get_logger(__name__)


class BaseEstimationManager(HasDimensions, HasData, HasComm):
    """Base class for estimation managers (row generation, ellipsoid, etc.)."""
    
    row_generation_cfg: Optional['RowGenerationConfig'] = None
    master_model: Any = None
    slack_counter: Optional[Dict] = None
    theta_val: Optional[NDArray[np.float64]] = None
    timing_stats: Optional[Dict[str, Any]] = None
    
    def __init__(
        self,
        comm_manager: 'CommManager',
        dimensions_cfg: 'DimensionsConfig',
        data_manager: 'DataManager',
        feature_manager: 'FeatureManager',
        subproblem_manager: 'SubproblemManager',
    ) -> None:
        """Initialize base estimation solver."""
        self.comm_manager = comm_manager
        self.dimensions_cfg = dimensions_cfg
        self.data_manager = data_manager
        self.feature_manager = feature_manager
        self.subproblem_manager = subproblem_manager

        self.agents_obs_features = self.get_agents_obs_features()
        self.obs_features = self.agents_obs_features.sum(0) if self.agents_obs_features is not None else None

    # ========================================================================
    # Observed Features
    # ========================================================================

    def get_obs_features(self) -> Optional[NDArray[np.float64]]:
        """Compute aggregate observed features (rank 0 only)."""
        local_bundles = self.local_data.get("obs_bundles")
        agents_obs_features = self.feature_manager.compute_gathered_features(local_bundles)
        return agents_obs_features.sum(0) if self.is_root() else None

    def get_agents_obs_features(self) -> Optional[NDArray[np.float64]]:
        """Compute per-agent observed features (rank 0 only)."""
        local_bundles = self.local_data.get("obs_bundles")
        agents_obs_features = self.feature_manager.compute_gathered_features(local_bundles)
        return agents_obs_features if self.is_root() else None

    # ========================================================================
    # Objective & Gradient
    # ========================================================================

    def compute_obj_and_gradient(self, theta: NDArray[np.float64]) -> Tuple[Optional[float], Optional[NDArray[np.float64]]]:
        """Compute objective and gradient in one call."""
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
            raise ValueError("theta must be 1D array, got scalar or 0D array")
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

    # ========================================================================
    # Abstract Solve Method
    # ========================================================================

    def solve(self) -> EstimationResult:
        """Main solve method (implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement the solve method")

    # ========================================================================
    # Result Creation (Shared)
    # ========================================================================

    def _create_result(
        self,
        theta: NDArray[np.float64],
        converged: bool,
        num_iterations: int,
        final_objective: Optional[float] = None,
        warnings: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EstimationResult:
        """Create EstimationResult with consistent handling for root/non-root ranks."""
        if self.is_root():
            return EstimationResult(
                theta_hat=theta.copy(),
                converged=converged,
                num_iterations=num_iterations,
                final_objective=final_objective,
                timing=self.timing_stats,
                iteration_history=None,
                warnings=warnings or [],
                metadata=metadata or {},
            )
        return EstimationResult(
            theta_hat=theta.copy(),
            converged=converged,
            num_iterations=num_iterations,
            final_objective=None,
            timing=None,
            iteration_history=None,
            warnings=[],
            metadata={},
        )

    # ========================================================================
    # Slack Counter Management
    # ========================================================================

    def _enforce_slack_counter(self) -> int:
        """Remove constraints that have been slack too long. Returns number removed."""
        if self.row_generation_cfg is None or self.master_model is None:
            return 0
        if self.row_generation_cfg.max_slack_counter >= float('inf'):
            return 0
        if self.slack_counter is None:
            self.slack_counter = {}
        
        to_remove = []
        for constr in self.master_model.getConstrs():
            if constr.Slack < -1e-6:
                self.slack_counter[constr] = self.slack_counter.get(constr, 0) + 1
                if self.slack_counter[constr] >= self.row_generation_cfg.max_slack_counter:
                    to_remove.append(constr)
            if constr.Pi > 1e-6:
                self.slack_counter.pop(constr, None)
        
        for constr in to_remove:
            self.master_model.remove(constr)
            self.slack_counter.pop(constr, None)
        
        if to_remove:
            logger.info("Removed %d slack constraints", len(to_remove))
        return len(to_remove)

    # ========================================================================
    # Logging Utilities
    # ========================================================================

    def _log_timing_summary(
        self,
        timing_stats: Dict[str, Any],
        obj_val: Optional[float] = None,
        theta: Optional[NDArray[np.float64]] = None,
        header: str = "ESTIMATION SUMMARY",
    ) -> None:
        """Log timing summary (rank 0 only)."""
        if not self.is_root():
            return
        
        total_time = timing_stats.get('total_time', 0.0)
        num_iters = timing_stats.get('num_iterations', 0)
        time_per_iter = timing_stats.get('time_per_iter', total_time / num_iters if num_iters > 0 else 0)
        pricing = timing_stats.get('pricing_time', 0.0)
        master = timing_stats.get('master_time', 0.0)
        other = timing_stats.get('other_time', total_time - pricing)
        pricing_pct = timing_stats.get('pricing_pct', 100 * pricing / total_time if total_time > 0 else 0)
        master_pct = timing_stats.get('master_pct', 100 * master / total_time if total_time > 0 else 0)
        other_pct = timing_stats.get('other_pct', 100 * other / total_time if total_time > 0 else 0)
        
        lines = ["=" * 70, header, "=" * 70]
        
        if obj_val is not None:
            lines.append(f"Objective value: {obj_val:.6f}")
        if theta is not None:
            if len(theta) <= 10:
                lines.append(f"Theta: {np.array2string(theta, precision=6, suppress_small=True)}")
            else:
                lines.append(f"Theta (dim={len(theta)}):")
                lines.append(f"  First 5: {np.array2string(theta[:5], precision=6, suppress_small=True)}")
                lines.append(f"  Last 5:  {np.array2string(theta[-5:], precision=6, suppress_small=True)}")
                lines.append(f"  Min: {theta.min():.6f}, Max: {theta.max():.6f}, Mean: {theta.mean():.6f}")
        
        lines.append(f"Iterations: {num_iters}")
        lines.append(f"Total time: {total_time:.2f}s ({time_per_iter:.2f}s/iter avg)")
        lines.append("")
        
        pricing_per = timing_stats.get('pricing_per_iter')
        master_per = timing_stats.get('master_per_iter')
        has_per_iter = pricing_per or master_per
        
        timing_header = "Timing:                  Total           [min  /  avg  /  max  per iter]" if has_per_iter else "Timing:"
        lines.append(timing_header)
        pricing_detail = f"  [{pricing_per['min']:.3f}s / {pricing_per['avg']:.3f}s / {pricing_per['max']:.3f}s]" if pricing_per else ""
        lines.append(f"  Pricing (subproblems): {pricing:5.2f}s ({pricing_pct:5.1f}%){pricing_detail}")
        if master > 0:
            master_detail = f"  [{master_per['min']:.3f}s / {master_per['avg']:.3f}s / {master_per['max']:.3f}s]" if master_per else ""
            lines.append(f"  Master problem:        {master:5.2f}s ({master_pct:5.1f}%){master_detail}")
        lines.append(f"  Other (sync/overhead): {other:5.2f}s ({other_pct:5.1f}%)")
        lines.append("")
        
        logger.info("\n".join(lines))

    def log_parameter(self) -> None:
        """Log current parameter values (if parameters_to_log is set)."""
        if self.row_generation_cfg is None or self.theta_val is None:
            return
        feature_ids = self.row_generation_cfg.parameters_to_log
        precision = 3
        if feature_ids is not None:
            logger.info("Parameters: %s", np.round(self.theta_val[feature_ids], precision))
        else:
            logger.info("Parameters: %s", np.round(self.theta_val, precision))


# Backward compatibility alias
BaseEstimationSolver = BaseEstimationManager
