"""
Base estimation solver for bundle choice estimation.

Provides common functionality for row generation, ellipsoid, and other solvers.
"""

import numpy as np
from typing import Any, Optional, Tuple
from numpy.typing import NDArray
from bundlechoice.base import HasDimensions, HasData, HasComm
from bundlechoice.utils import get_logger

logger = get_logger(__name__)


# ============================================================================
# Base Estimation Solver
# ============================================================================

class BaseEstimationSolver(HasDimensions, HasData, HasComm):
    """Base class for estimation solvers (row generation, ellipsoid, etc.)."""
    
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

    def solve(self) -> NDArray[np.float64]:
        """Main solve method (implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement the solve method") 