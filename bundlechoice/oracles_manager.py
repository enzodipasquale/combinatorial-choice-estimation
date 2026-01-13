import numpy as np
from typing import Any, Callable, Optional, Dict
from numpy.typing import NDArray
from bundlechoice.utils import get_logger
from bundlechoice.config import DimensionsConfig
from bundlechoice.data_manager import DataManager
logger = get_logger(__name__)


class OraclesManager:
    """
    Manages feature and error oracles for bundle choice model.
    
    Oracles can be set in two ways:
    1. Single-agent: set_features_oracle(fn), set_error_oracle(fn)
       - fn(agent_id, bundle, data) -> features/error for one agent
    2. Vectorized (preferred): set_vectorized_features_oracle(fn), set_vectorized_error_oracle(fn)
       - fn(bundles, data) -> features_matrix/errors_vector for all local agents
    
    When using build_from_data(), both single-agent and vectorized versions are auto-generated.
    """

    def __init__(self, dimensions_cfg: DimensionsConfig, comm_manager, data_manager: DataManager) -> None:
        """Initialize OraclesManager."""
        self.dimensions_cfg = dimensions_cfg
        self.comm_manager = comm_manager
        self.data_manager = data_manager
        self._features_oracle: Optional[Callable] = None
        self._error_oracle: Optional[Callable] = None
        self._vectorized_features: Optional[Callable] = None
        self._vectorized_errors: Optional[Callable] = None

    def set_features_oracle(self, _features_oracle: Callable[[int, NDArray[np.float64], Dict[str, Any]], NDArray[np.float64]]) -> None:
        """Set user-supplied feature extraction function (single-agent)."""
        self._features_oracle = _features_oracle
        self._vectorized_features = None
        self.validate_oracle()

    def set_vectorized_features_oracle(self, vectorized_fn: Callable[[NDArray[np.float64], Dict[str, Any]], NDArray[np.float64]]) -> None:
        """Set user-supplied vectorized feature function."""
        self._vectorized_features = vectorized_fn

    def validate_oracle(self) -> None:
        """Validate that features_oracle returns shape (num_features,)."""
        if self.comm_manager is not None and self.comm_manager.rank != 0:
            return
            
        if self._features_oracle is None:
            return
        
        test_bundle = np.ones(self.dimensions_cfg.num_items)
        if self.data_manager.input_data is not None:
            test_features = self._features_oracle(0, test_bundle, self.data_manager.input_data)
        elif self.data_manager.local_data is not None:
            agent_data = self.data_manager.local_data.get("agent_data", {})
            modular = agent_data.get("modular")
            if modular is not None and modular.shape[0] > 0:
                test_features = self._features_oracle(0, test_bundle, self.data_manager.local_data)
            else:
                return
        else:
            return
        
        assert test_features.shape == (self.dimensions_cfg.num_features,), \
            f"features_oracle must return shape ({self.dimensions_cfg.num_features},), got {test_features.shape}"

    def set_error_oracle(self, _error_oracle: Callable[[int, NDArray[np.float64], Dict[str, Any]], float]) -> None:
        """Set user-supplied error oracle function (single-agent)."""
        self._error_oracle = _error_oracle
        self._vectorized_errors = None

    def set_vectorized_error_oracle(self, vectorized_fn: Callable[[NDArray[np.float64], Dict[str, Any]], NDArray[np.float64]]) -> None:
        """Set user-supplied vectorized error function."""
        self._vectorized_errors = vectorized_fn

    def error_oracle(self, agent_id: int, bundle: NDArray[np.float64], 
                    data_override: Optional[Dict[str, Any]] = None) -> float:
        """Compute error for single agent/bundle."""
        if self._error_oracle is None:
            raise RuntimeError("_error_oracle function is not set. Call build_error_oracle_from_data() first.")
        data = data_override if data_override is not None else self.data_manager.local_data
        return self._error_oracle(agent_id, bundle, data)

    def build_error_oracle_from_data(self) -> Callable[[int, NDArray[np.float64], Dict[str, Any]], float]:
        """Build modular error oracle from data['errors'] with vectorized support."""
        if self._error_oracle is not None:
            logger.info("Rebuilding error oracle (overwriting existing)")
        
        def modular_error_oracle(agent_id: int, bundle: NDArray[np.float64], data: Dict[str, Any]) -> float:
            errors = data.get("errors")
            if errors is None:
                return 0.0
            return float((errors[agent_id] * bundle).sum())
        
        def vectorized_modular_errors(bundles: NDArray[np.float64], data: Dict[str, Any]) -> NDArray[np.float64]:
            errors = data.get("errors")
            if errors is None:
                return np.zeros(len(bundles), dtype=np.float64)
            return (errors * bundles).sum(axis=1)
        
        self._error_oracle = modular_error_oracle
        self._vectorized_errors = vectorized_modular_errors
        return modular_error_oracle

    def build_local_modular_error_oracle(
        self, 
        seed: int = 42, 
        correlation_matrix: Optional[NDArray[np.float64]] = None
    ) -> None:
        """
        Build modular error oracle with locally generated IID normal errors.
        
        Instead of scattering errors from rank 0, each rank generates its own
        errors deterministically based on global agent indices.
        """
        if correlation_matrix is not None:
            correlation_matrix = self.comm_manager.broadcast_from_root(
                correlation_matrix if self.comm_manager.is_root() else None, root=0
            )
            L = np.linalg.cholesky(correlation_matrix)
        else:
            L = None
        
        total_agents = self.dimensions_cfg.num_simulations * self.dimensions_cfg.num_agents
        idx_chunks = np.array_split(np.arange(total_agents), self.comm_manager.size)
        local_global_indices = idx_chunks[self.comm_manager.rank]
        num_local = len(local_global_indices)
        
        local_errors = np.empty((num_local, self.dimensions_cfg.num_items), dtype=np.float64)
        for local_idx, global_idx in enumerate(local_global_indices):
            rng = np.random.default_rng(seed + global_idx)
            if L is not None:
                z = rng.standard_normal(self.dimensions_cfg.num_items)
                local_errors[local_idx] = L @ z
            else:
                local_errors[local_idx] = rng.standard_normal(self.dimensions_cfg.num_items)
        
        if self.data_manager.local_data is None:
            self.data_manager.local_data = {}
        self.data_manager.local_data["errors"] = local_errors
        self.data_manager.num_local_agents = num_local
        
        self.build_error_oracle_from_data()
        
        if self.comm_manager.is_root():
            corr_info = f" with {self.dimensions_cfg.num_items}x{self.dimensions_cfg.num_items} correlation matrix" if L is not None else ""
            logger.info(f"Built local modular error oracle: {total_agents} agents, seed={seed}{corr_info}")

    def features_oracle(self, agent_id: int, bundle: NDArray[np.float64], 
                       data_override: Optional[Dict[str, Any]] = None) -> NDArray[np.float64]:
        """Compute features for single agent/bundle."""
        if self._features_oracle is None:
            raise RuntimeError("_features_oracle function is not set.")
        data = data_override if data_override is not None else self.data_manager.input_data
        return self._features_oracle(agent_id, bundle, data)

    def compute_rank_features(self, local_bundles: Optional[NDArray[np.float64]]) -> NDArray[np.float64]:
        """Compute features for all local agents on this rank. Uses vectorized if available."""
        if self.data_manager.num_local_agents == 0 or local_bundles is None or len(local_bundles) == 0:
            return np.empty((0, self.dimensions_cfg.num_features), dtype=np.float64)
        
        assert self.data_manager.num_local_agents == len(local_bundles), \
            f"num_local_agents ({self.data_manager.num_local_agents}) != len(local_bundles) ({len(local_bundles)})"
        
        if self._vectorized_features is not None:
            return self._vectorized_features(local_bundles, self.data_manager.local_data)
        
        if self._features_oracle is None:
            raise RuntimeError("No features oracle set. Call set_features_oracle() or set_vectorized_features_oracle().")
        
        return np.stack([self._features_oracle(i, local_bundles[i], self.data_manager.local_data) 
                        for i in range(self.data_manager.num_local_agents)])

    def compute_gathered_features(self, local_bundles: Optional[NDArray[np.float64]]) -> Optional[NDArray[np.float64]]:
        """Compute features for all agents, gather to rank 0."""
        features_local = self.compute_rank_features(local_bundles)
        return self.comm_manager.concatenate_array_at_root_fast(features_local, root=0)

    def compute_rank_errors(self, local_bundles: Optional[NDArray[np.float64]]) -> NDArray[np.float64]:
        """Compute errors for all local agents on this rank. Uses vectorized if available."""
        if self.data_manager.num_local_agents == 0 or local_bundles is None or len(local_bundles) == 0:
            return np.empty(0, dtype=np.float64)
        
        if self._vectorized_errors is not None:
            return self._vectorized_errors(local_bundles, self.data_manager.local_data)
        
        if self._error_oracle is None:
            raise RuntimeError("No error oracle set. Call set_error_oracle(), set_vectorized_error_oracle(), or build_error_oracle_from_data().")
        
        return np.array([self._error_oracle(i, local_bundles[i], self.local_data) 
                        for i in range(self.num_local_agents)])



    def compute_gathered_utilities(self, local_bundles: NDArray[np.float64], 
                                   theta: NDArray[np.float64]) -> Optional[NDArray[np.float64]]:
        """Compute utilities for all agents, gather to rank 0."""
        features_local = self.compute_rank_features(local_bundles)
        errors_local = self.compute_rank_errors(local_bundles)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            utilities_local = features_local @ theta + errors_local
        return self.comm_manager.concatenate_array_at_root_fast(utilities_local, root=0)

    def compute_gathered_errors(self, local_bundles: NDArray[np.float64]) -> Optional[NDArray[np.float64]]:
        """Compute errors for all agents, gather to rank 0."""
        errors_local = self.compute_rank_errors(local_bundles)
        return self.comm_manager.concatenate_array_at_root_fast(errors_local, root=0)

    def build_from_data(self) -> None:
        """Build both feature and error oracles from data structure."""
        self.build_features_oracle_from_data()
        self.build_error_oracle_from_data()

    def build_features_oracle_from_data(self) -> Callable[[int, NDArray[np.float64], Dict[str, Any]], NDArray[np.float64]]:
        """Build features_oracle from data structure (modular/quadratic keys)."""
        if self._features_oracle is not None:
            logger.info("Rebuilding feature oracle (overwriting existing)")
        
        self.data_manager.verify_feature_count()
        
        if self.comm_manager.is_root():
            data_info = self.data_manager.get_data_info(self.data_manager.input_data)
            flags = (
                data_info["has_modular_agent"],
                data_info["has_quadratic_agent"],
                data_info["has_modular_item"],
                data_info["has_quadratic_item"]
            )
            
            if self.dimensions_cfg.feature_names is None:
                auto_names = self.data_manager.get_feature_names_from_data()
                if auto_names:
                    self.dimensions_cfg.feature_names = auto_names
                    logger.info(f"Auto-set {len(auto_names)} feature names from data sources")
        else:
            flags = None
        
        flags = self.comm_manager.broadcast_from_root(flags, root=0)
        has_modular_agent, has_quadratic_agent, has_modular_item, has_quadratic_item = flags

        def make_features_oracle(ma, qa, mi, qi):
            def features_oracle(agent_id: int, bundle: NDArray[np.float64], data: Dict[str, Any]) -> NDArray[np.float64]:
                feats = []
                if ma:
                    modular = data['agent_data']['modular'][agent_id]
                    feats.append(np.einsum('jk,j->k', modular, bundle))
                if qa:
                    quadratic = data['agent_data']['quadratic'][agent_id]
                    feats.append(np.einsum('jlk,j,l->k', quadratic, bundle, bundle))
                if mi:
                    modular = data['item_data']['modular']
                    feats.append(np.einsum('jk,j->k', modular, bundle))
                if qi:
                    quadratic = data['item_data']['quadratic']
                    feats.append(np.einsum('jlk,j,l->k', quadratic, bundle, bundle))
                return np.concatenate(feats)
            return features_oracle
        
        def make_vectorized_features(ma, qa, mi, qi):
            def vectorized_features(bundles: NDArray[np.float64], data: Dict[str, Any]) -> NDArray[np.float64]:
                feats = []
                if ma:
                    modular = data['agent_data']['modular']
                    feats.append(np.einsum('ijk,ij->ik', modular, bundles))
                if qa:
                    quadratic = data['agent_data']['quadratic']
                    feats.append(np.einsum('ijlk,ij,il->ik', quadratic, bundles, bundles))
                if mi:
                    modular = data['item_data']['modular']
                    feats.append(np.einsum('jk,ij->ik', modular, bundles))
                if qi:
                    quadratic = data['item_data']['quadratic']
                    feats.append(np.einsum('jlk,ij,il->ik', quadratic, bundles, bundles))
                return np.concatenate(feats, axis=1)
            return vectorized_features
        
        self._features_oracle = make_features_oracle(has_modular_agent, has_quadratic_agent, has_modular_item, has_quadratic_item)
        self._vectorized_features = make_vectorized_features(has_modular_agent, has_quadratic_agent, has_modular_item, has_quadratic_item)
        
        return self._features_oracle
