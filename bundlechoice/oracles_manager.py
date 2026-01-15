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
        self.local_ids = np.arange(self.dimensions_cfg.num_local_agents)
        self._features_oracle_takes_data = None
        self._error_oracle_takes_data = None




    def _check_vectorized_oracle_support(self, oracle):
        try:
            test_bundles = np.zeros((self.dimensions_cfg.num_local_agents, self.dimensions_cfg.num_items), dtype=bool)
            if len(oracle.func_code.co_argcount) == 3:
                test_features = oracle(np.arange(self.dimensions_cfg.num_local_agents), test_bundles)
            else:
                test_features = oracle(np.arange(self.dimensions_cfg.num_local_agents), test_bundles, self.data_manager.local_data)
            assert test_features.shape == (self.dimensions_cfg.num_local_agents, self.dimensions_cfg.num_features)
            return True
        except:
            return False
    def _check_oracle_takes_data(self, oracle):
        if len(oracle.func_code.co_argcount) == 3:
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
    
    def features_oracle(self, bundles, local_ids=None):
        if local_ids is None:
            local_ids = self.local_ids
        if self._features_oracle_vectorized:
            return self._features_oracle(bundles, local_ids, self.data_manager.local_data)
        else:
            return np.stack([self._features_oracle(bundles[id], id, self.data_manager.local_data) for id in local_ids])

    def error_oracle(self, bundles, local_ids=None):
        if local_ids is None:
            local_ids = self.local_ids
        if self._error_oracle_vectorized:
            return self._error_oracle(bundles, local_ids)
        else:
            return np.stack([self._error_oracle(bundles[id], 
                    self.data_manager.local_agent_ids[id]) for id in local_ids], )

    def build_local_modular_error_oracle(self, seed=42, items_correlation_matrix=None):
        np.random.seed(seed + self.comm_manager.rank)
        local_errors = np.random.normal(0, 1, (self.dimensions_cfg.num_local_agents, self.dimensions_cfg.num_items))
        if items_correlation_matrix is not None:
            L = np.linalg.cholesky(items_correlation_matrix)
            local_errors = local_errors @ L
        self._error_oracle = lambda bundles, local_ids: local_errors[local_ids] @ bundles
        self._error_oracle_vectorized = True
        self._error_oracle_takes_data = False
        return self._error_oracle

    def utilities_oracle(self, local_ids, bundles, theta):
        return self.features_oracle(local_ids, bundles) @ theta + self.error_oracle(local_ids, bundles)
 

    def build_quadratic_features_from_data(self):
        ma, qa, mi, qi = self.data_manager.quadratic_features_flags()
        def features_oracle(bundles, local_ids, data):
            feats = []
            if ma:
                modular = data['agent_data']['modular'][local_ids]
                feats.append(np.einsum('jk,j->k', modular, bundles))
            if qa:
                quadratic = data['agent_data']['quadratic'][local_ids]
                feats.append(np.einsum('jlk,j,l->k', quadratic, bundles, bundles))
            if mi:
                modular = data['item_data']['modular']
                feats.append(np.einsum('jk,j->k', modular, bundles))
            if qi:
                quadratic = data['item_data']['quadratic']
                feats.append(np.einsum('jlk,j,l->k', quadratic, bundles, bundles))
            return np.concatenate(feats)
        self._features_oracle = features_oracle
        self._features_oracle_vectorized = True
        self._features_oracle_takes_data = True
        return self._features_oracle

      