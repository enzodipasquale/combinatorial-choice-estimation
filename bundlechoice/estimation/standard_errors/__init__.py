import numpy as np
from .result import StandardErrorsResult
from .sandwich import SandwichMixin
from .resampling import ResamplingMixin

class StandardErrorsManager(SandwichMixin, ResamplingMixin):

    def __init__(self, comm_manager, config, data_manager, oracles_manager, subproblem_manager):
        self.comm_manager = comm_manager
        self.config = config
        self.data_manager = data_manager
        self.oracles_manager = oracles_manager
        self.subproblem_manager = subproblem_manager
        self.se_cfg = config.standard_errors
        self.dim = config.dimensions
        self._obs_features = None
        self._mean_obs_full = None
        self._mean_obs_subset = None

    def clear_cache(self):
        self._obs_features = None
        self._mean_obs_full = None
        self._mean_obs_subset = None

__all__ = ['StandardErrorsManager', 'StandardErrorsResult']
