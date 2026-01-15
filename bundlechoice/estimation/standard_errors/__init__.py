from typing import Optional
import numpy as np
from numpy.typing import NDArray
from bundlechoice.config import DimensionsConfig, StandardErrorsConfig
from bundlechoice.comm_manager import CommManager
from bundlechoice.data_manager import DataManager
from bundlechoice.oracles_manager import OraclesManager
from bundlechoice.subproblems.subproblem_manager import SubproblemManager
from .result import StandardErrorsResult
from .sandwich import SandwichMixin
from .resampling import ResamplingMixin

class StandardErrorsManager(SandwichMixin, ResamplingMixin):

    def __init__(self, comm_manager, dimensions_cfg, data_manager, oracles_manager, subproblem_manager, se_cfg):
        self.comm_manager = comm_manager
        self.dimensions_cfg = dimensions_cfg
        self.data_manager = data_manager
        self.oracles_manager = oracles_manager
        self.subproblem_manager = subproblem_manager
        self.se_cfg = se_cfg
        self._obs_features: Optional[NDArray[np.float64]] = None
        self._mean_obs_full: Optional[NDArray[np.float64]] = None
        self._mean_obs_subset: Optional[dict] = None

    def clear_cache(self):
        self._obs_features = None
        self._mean_obs_full = None
        self._mean_obs_subset = None
__all__ = ['StandardErrorsManager', 'StandardErrorsResult']