import numpy as np
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
        self._features_oracle_takes_data = None
        self._error_oracle_takes_data = None
        self._modular_local_errors = None




    def _check_vectorized_oracle_support(self, oracle):
        try:
            test_bundles = np.zeros((self.data_manager.num_local_agent, self.dimensions_cfg.num_items), dtype=bool)
            if oracle.__code__.co_argcount == 2:
                test_features = oracle(np.arange(self.data_manager.num_local_agent), test_bundles)
            else: 
                test_features = oracle(np.arange(self.data_manager.num_local_agent), test_bundles, self.data_manager.local_data)
            assert test_features.shape == (self.data_manager.num_local_agent, self.dimensions_cfg.num_features)
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
    
    def features_oracle(self, bundles, local_id=None):
        if local_id is None:
            local_id = self.data_manager.local_id
        if self._features_oracle_vectorized:
            return self._features_oracle(bundles, local_id, self.data_manager.local_data)
        else:
            return np.stack([self._features_oracle(bundles[id], id, self.data_manager.local_data) for id in local_id])

    def error_oracle(self, bundles, local_id=None):
        if local_id is None:
            local_id = self.data_manager.local_id
        if self._error_oracle_vectorized:
            return self._error_oracle(bundles, local_id)
        else:
            return np.stack([self._error_oracle(bundles[id], 
                    self.data_manager.local_id[id]) for id in local_id], )

    def build_local_modular_error_oracle(self, seed=42, items_correlation_matrix=None):
        np.random.seed(seed + self.comm_manager.rank)
        self._modular_local_errors = np.random.normal(0, 1, (self.data_manager.num_local_agent, self.dimensions_cfg.num_items))
        if items_correlation_matrix is not None:
            L = np.linalg.cholesky(items_correlation_matrix)
            self._modular_local_errors = self._modular_local_errors @ L
        self._error_oracle = lambda bundles, local_id: self._modular_local_errors[local_id] @ bundles
        self._error_oracle_vectorized = True
        self._error_oracle_takes_data = False
        return self._error_oracle

    def utilities_oracle(self, bundles, theta, local_id = None):
        return self.features_oracle(bundles, local_id) @ theta + self.error_oracle(bundles, local_id)
 

    def build_quadratic_features_from_data(self):
        ma, qa, mi, qi = self.data_manager.quadratic_features_flags()
        def features_oracle(bundles, local_id, data):
            feats = []
            if ma:
                modular = data['agent_data']['modular'][local_id]
                feats.append(np.einsum('jk,j->k', modular, bundles))
            if mi:
                modular = data['item_data']['modular']
                feats.append(np.einsum('jk,j->k', modular, bundles))
            if qa:
                quadratic = data['agent_data']['quadratic'][local_id]
                feats.append(np.einsum('jlk,j,l->k', quadratic, bundles, bundles))
            if qi:
                quadratic = data['item_data']['quadratic']
                feats.append(np.einsum('jlk,j,l->k', quadratic, bundles, bundles))
            return np.concatenate(feats)
        self._features_oracle = features_oracle
        self._features_oracle_vectorized = True
        self._features_oracle_takes_data = True
        return self._features_oracle

      