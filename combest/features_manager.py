import numpy as np
from functools import lru_cache
from combest.utils import get_logger
logger = get_logger(__name__)

class FeaturesManager:

    def __init__(self, dimensions_cfg, comm_manager, data_manager):
        self.dimensions_cfg = dimensions_cfg
        self.comm_manager = comm_manager
        self.data_manager = data_manager

        self._covariates_oracle = None
        self._error_oracle = None

        self._covariates_oracle_vectorized = None
        self._error_oracle_vectorized = None
        self._covariates_oracle_takes_data = True
        self._error_oracle_takes_data = False
        self.local_modular_errors = None


    def _check_vectorized_oracle_support(self, oracle):
        try:
            test_bundles = np.zeros((self.comm_manager.num_local_agent, self.dimensions_cfg.n_items), dtype=bool)
            ids = np.arange(self.comm_manager.num_local_agent)
            if oracle.__code__.co_argcount == 2:
                test_features = oracle(test_bundles, ids)
            else:
                test_features = oracle(test_bundles, ids, self.data_manager.local_data)
            assert test_features.shape == (self.comm_manager.num_local_agent, self.dimensions_cfg.n_covariates)
            return True
        except (ValueError, IndexError, AssertionError) as e:
            logger.debug(f"Vectorized oracle not supported: {e}")
            return False

    def _check_oracle_takes_data(self, oracle):
        return oracle.__code__.co_argcount == 3

    def set_covariates_oracle(self, _covariates_oracle):
        self._covariates_oracle = _covariates_oracle
        self._covariates_oracle_vectorized = self._check_vectorized_oracle_support(_covariates_oracle)
        self._covariates_oracle_takes_data = self._check_oracle_takes_data(_covariates_oracle)

    def set_error_oracle(self, _error_oracle):
        self._error_oracle = _error_oracle
        self._error_oracle_vectorized = self._check_vectorized_oracle_support(_error_oracle)
        self._error_oracle_takes_data = self._check_oracle_takes_data(_error_oracle)
    
    def covariates_oracle(self, bundles, ids=None):
        ids = self.comm_manager.local_agents_arange if ids is None else ids
        data_arg = (self.data_manager.local_data,) if self._covariates_oracle_takes_data else ()
        if self._covariates_oracle_vectorized:
            return self._covariates_oracle(bundles, ids, *data_arg)
        else:
            return np.stack([self._covariates_oracle(bundles[i], i, *data_arg) for i in ids])

    def error_oracle(self, bundles, ids=None):
        ids = self.comm_manager.local_agents_arange if ids is None else ids
        data_arg = (self.data_manager.local_data,) if self._error_oracle_takes_data else ()
        if self._error_oracle_vectorized:
            return self._error_oracle(bundles, ids, *data_arg)
        else:
            return np.stack([self._error_oracle(bundles[i], i, *data_arg) for i in ids])

    def utility_oracle(self, bundles, theta, ids = None):
        return self.covariates_oracle(bundles, ids) @ theta + self.error_oracle(bundles, ids)

    def covariates_and_errors_oracle(self, bundles, ids = None):
        features = self.covariates_oracle(bundles, ids)
        error = self.error_oracle(bundles, ids)
        return features, error

    def covariates_oracle_individual(self, bundle, idx):
        data_arg = (self.data_manager.local_data,) if self._covariates_oracle_takes_data else ()
        if self._covariates_oracle_vectorized:
            bundle, idx = bundle[None, :], np.atleast_1d(idx)
        return self._covariates_oracle(bundle, idx, *data_arg)

    def error_oracle_individual(self, bundle, idx):
        data_arg = (self.data_manager.local_data,) if self._error_oracle_takes_data else ()
        if self._error_oracle_vectorized:
            bundle, idx = bundle[None, :], np.atleast_1d(idx)
        return self._error_oracle(bundle, idx, *data_arg)

    def utility_oracle_individual(self, bundle, theta, idx):
        vals = self.covariates_oracle_individual(bundle, idx) @ theta + self.error_oracle_individual(bundle, idx)
        return vals.ravel()[0] if np.ndim(idx) == 0 else vals


    def build_local_modular_error_oracle(self, seed=42, covariance_matrix=None, sigma = 1):
        n_local = self.comm_manager.num_local_agent
        n_items = self.dimensions_cfg.n_items
        self.local_modular_errors = np.zeros((n_local, n_items))
        for i, global_id in enumerate(self.comm_manager.agent_ids):
            rng = np.random.default_rng((seed, global_id))
            self.local_modular_errors[i] = rng.normal(0, sigma, n_items)
        if covariance_matrix is not None:
            if covariance_matrix.ndim == 2:
                L = np.linalg.cholesky(covariance_matrix)
                self.local_modular_errors = self.local_modular_errors @ L
            elif covariance_matrix.ndim == 3:
                obs_ids = self.comm_manager.obs_ids
                for obs_idx in np.unique(obs_ids):
                    mask = obs_ids == obs_idx
                    L = np.linalg.cholesky(covariance_matrix[obs_idx])
                    self.local_modular_errors[mask] = self.local_modular_errors[mask] @ L
        self._error_oracle = lambda bundles, ids: (self.local_modular_errors[ids] * bundles).sum(-1)
        self._error_oracle_vectorized = True
        self._error_oracle_takes_data = False
        return self._error_oracle

    def build_quadratic_covariates_from_data(self):
        self.data_manager._validate_quadratic_data_dimensions()
        qinfo = self.data_manager.get_quadratic_data_info()
        def quadratic_covariates_oracle(bundles, ids, data):
            feats = []
            if qinfo.modular_agent:
                modular = data["id_data"]['modular'][ids]
                feats.append(np.einsum('ijk,ij->ik', modular, bundles))
            if qinfo.modular_item:
                modular = data["item_data"]['modular']
                feats.append(np.einsum('jk,ij->ik', modular, bundles))
            if qinfo.quadratic_agent:
                quadratic = data["id_data"]['quadratic'][ids]
                feats.append(np.einsum('ijlk,ij,il->ik', quadratic, bundles, bundles))
            if qinfo.quadratic_item:
                quadratic = data["item_data"]['quadratic']
                feats.append(np.einsum('jlk,ij,il->ik', quadratic, bundles, bundles))
            return np.concatenate(feats, axis=-1)
        self._covariates_oracle = quadratic_covariates_oracle
        self._covariates_oracle_vectorized = True
        self._covariates_oracle_takes_data = True
        return self._covariates_oracle

