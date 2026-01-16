import numpy as np
from functools import lru_cache
from numpy.typing import NDArray
from bundlechoice.utils import get_logger
from bundlechoice.config import DimensionsConfig
from bundlechoice.data_manager import DataManager
logger = get_logger(__name__)

class OraclesManager:

    def __init__(self, dimensions_cfg, comm_manager, data_manager):
        self.dimensions_cfg = dimensions_cfg
        self.comm_manager = comm_manager
        self.data_manager = data_manager

        self._features_oracle = None
        self._error_oracle = None

        self._features_oracle_vectorized = None
        self._error_oracle_vectorized = None
        self._features_oracle_takes_data = True
        self._error_oracle_takes_data = False
        self._modular_local_errors = None

    @property
    def _features_at_obs_bundles_at_root(self):
        version = getattr(self.data_manager, '_local_data_version', 0)
        return self._compute_features_at_obs_bundles_at_root(version)

    @lru_cache(maxsize=1)
    def _compute_features_at_obs_bundles_at_root(self, _version):
        local_obs_features = self.features_oracle(self.data_manager.local_obs_bundles)
        return self.comm_manager.sum_row_andReduce(local_obs_features)

    def _check_vectorized_oracle_support(self, oracle):
        try:
            test_bundles = np.zeros((self.data_manager.num_local_agent, self.dimensions_cfg.num_items), dtype=bool)
            ids = np.arange(self.data_manager.num_local_agent)
            if oracle.__code__.co_argcount == 2:
                test_features = oracle(ids, test_bundles)
            else: 
                test_features = oracle(ids, test_bundles, self.data_manager.local_data)
            assert test_features.shape == (self.data_manager.num_local_agent, 
                                            self.dimensions_cfg.num_features)
            return True
        except:
            return False

    def _check_oracle_takes_data(self, oracle):
        if oracle.__code__.co_argcount == 3:
            return True
        else:
            return False

    def set_features_oracle(self, _features_oracle):
        self._features_oracle = _features_oracle
        self._features_oracle_vectorized = self._check_vectorized_oracle_support(_features_oracle)
        self._features_oracle_takes_data = self._check_oracle_takes_data(_features_oracle)

    def set_error_oracle(self, _error_oracle):
        self._error_oracle = _error_oracle
        self._error_oracle_vectorized = self._check_vectorized_oracle_support(_error_oracle)
        self._error_oracle_takes_data = self._check_oracle_takes_data(_error_oracle)
    
    def features_oracle(self, bundles, ids=None):
        ids = np.arange(self.data_manager.num_local_agent) if ids is None else ids
        data_arg = (self.data_manager.local_data,) if self._features_oracle_takes_data else ()
        if self._features_oracle_vectorized:
            return self._features_oracle(bundles, ids, *data_arg)
        else:
            return np.stack([self._features_oracle(bundles[id], id, *data_arg) for id in ids])

    def error_oracle(self, bundles, ids=None):
        ids = np.arange(self.data_manager.num_local_agent) if ids is None else ids
        data_arg = (self.data_manager.local_data,) if self._error_oracle_takes_data else ()
        if self._error_oracle_vectorized:
            return self._error_oracle(bundles, ids, *data_arg)
        else:
            return np.stack([self._error_oracle(bundles[id], id, *data_arg) for id in ids])

    def utility_oracle(self, bundles, theta, ids = None):
        return self.features_oracle(bundles, ids) @ theta + self.error_oracle(bundles, ids)
 
    def features_oracle_individual(self, bundle, id):
        data_arg = (self.data_manager.local_data,) if self._features_oracle_takes_data else ()
        if self._features_oracle_vectorized:
            bundle, id = bundle[None, :], np.atleast_1d(id)
        ids_arg = np.atleast_1d(id) if self._features_oracle_vectorized else id
        return self._features_oracle(bundle, ids_arg, *data_arg)

    def error_oracle_individual(self, bundle, id):
        data_arg = (self.data_manager.local_data,) if self._error_oracle_takes_data else ()
        if self._error_oracle_vectorized:
            bundle, ids_arg = bundle[None, :], np.atleast_1d(id)            
        return self._error_oracle(bundle, ids_arg, *data_arg)

    def utility_oracle_individual(self, bundle, theta, id):
        vals = self.features_oracle_individual(bundle, id) @ theta + self.error_oracle_individual(bundle, id)
        return vals.ravel()[0] if np.ndim(id) == 0 else vals


    def build_local_modular_error_oracle(self, seed=42, items_correlation_matrix=None):
        np.random.seed(seed + self.comm_manager.rank)
        self._modular_local_errors = np.random.normal(0, 1, (self.data_manager.num_local_agent, 
                                                                self.dimensions_cfg.num_items))
        if items_correlation_matrix is not None:
            L = np.linalg.cholesky(items_correlation_matrix)
            self._modular_local_errors = self._modular_local_errors @ L
        self._error_oracle = lambda bundles, ids: (self._modular_local_errors * bundles).sum(-1)
        self._error_oracle_vectorized = True
        self._error_oracle_takes_data = False
        return self._error_oracle

    def build_quadratic_features_from_data(self):
        qinfo = self.data_manager.quadratic_data_info
        def features_oracle(bundles, ids, data):
            feats = []
            if qinfo.modular_agent:
                modular = data['agent_data']['modular'][ids]
                feats.append(np.einsum('ijk,ij->ik', modular, bundles))
            if qinfo.modular_item:
                modular = data['item_data']['modular']
                feats.append(np.einsum('jk,ij->ik', modular, bundles))
            if qinfo.quadratic_agent:
                quadratic = data['agent_data']['quadratic'][ids]
                feats.append(np.einsum('ijlk,ij,il->ik', quadratic, bundles, bundles))
            if qinfo.quadratic_item:
                quadratic = data['item_data']['quadratic']
                feats.append(np.einsum('jlk,ij,il->ik', quadratic, bundles, bundles))
            return np.concatenate(feats, axis=-1)
        self._features_oracle = features_oracle
        self._features_oracle_vectorized = True
        self._features_oracle_takes_data = True
        return self._features_oracle

