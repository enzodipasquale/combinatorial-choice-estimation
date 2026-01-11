"""
Standard errors computation for bundle choice estimation.

Provides multiple methods:
- Sandwich SE: Full A^{-1} B A^{-1} estimator
- B-inverse SE: Simplified B^{-1} estimator (no finite differences)
- Bootstrap SE: Resample agents with replacement
- Subsampling SE: Subsample without replacement
- Bayesian Bootstrap SE: Reweight agents (avoids rare-item problem with FE)
"""

from typing import Optional
import numpy as np
from numpy.typing import NDArray

from bundlechoice.base import HasDimensions, HasData, HasComm
from bundlechoice.config import DimensionsConfig, StandardErrorsConfig
from bundlechoice.comm_manager import CommManager
from bundlechoice.data_manager import DataManager
from bundlechoice.feature_manager import FeatureManager
from bundlechoice.subproblems.subproblem_manager import SubproblemManager

from .result import StandardErrorsResult
from .sandwich import SandwichMixin
from .resampling import ResamplingMixin


class StandardErrorsManager(SandwichMixin, ResamplingMixin, HasDimensions, HasData, HasComm):
    """
    Computes standard errors for bundle choice estimation.
    
    Methods:
        compute(): Full sandwich estimator (A^{-1} B A^{-1})
        compute_B_inverse(): B^{-1} only (faster, no finite differences)
        compute_bootstrap(): Standard bootstrap (resample agents)
        compute_subsampling(): Subsampling (subsample without replacement)
        compute_bayesian_bootstrap(): Bayesian bootstrap (reweight agents)
    """
    
    def __init__(
        self,
        comm_manager: CommManager,
        dimensions_cfg: DimensionsConfig,
        data_manager: DataManager,
        feature_manager: FeatureManager,
        subproblem_manager: SubproblemManager,
        se_cfg: StandardErrorsConfig,
    ):
        self.comm_manager = comm_manager
        self.dimensions_cfg = dimensions_cfg
        self.data_manager = data_manager
        self.feature_manager = feature_manager
        self.subproblem_manager = subproblem_manager
        self.se_cfg = se_cfg
        
        # Caches
        self._obs_features: Optional[NDArray[np.float64]] = None
        self._mean_obs_full: Optional[NDArray[np.float64]] = None
        self._mean_obs_subset: Optional[dict] = None
    
    def clear_cache(self) -> None:
        """Clear cached values. Call if underlying data changes."""
        self._obs_features = None
        self._mean_obs_full = None
        self._mean_obs_subset = None


__all__ = ["StandardErrorsManager", "StandardErrorsResult"]
