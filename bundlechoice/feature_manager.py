import numpy as np
from typing import Any, Callable, Optional, Dict, Tuple
from numpy.typing import NDArray
from bundlechoice.utils import get_logger
from bundlechoice.config import DimensionsConfig
from bundlechoice.data_manager import DataManager
from bundlechoice.base import HasDimensions, HasData, HasComm
logger = get_logger(__name__)


# ============================================================================
# FeatureManager
# ============================================================================

class FeatureManager(HasDimensions, HasComm, HasData):
    """
    Manages feature and error extraction for bundle choice model.
    
    User supplies features_oracle(agent_id, bundle, data) function.
    User can optionally supply error_oracle(agent_id, bundle, data) function.
    Supports batch/vectorized computation when oracles are built from data.
    """

    def __init__(self, dimensions_cfg: DimensionsConfig, comm_manager, data_manager: DataManager) -> None:
        """Initialize FeatureManager."""
        self.dimensions_cfg = dimensions_cfg
        self.comm_manager = comm_manager
        self.data_manager = data_manager
        self._features_oracle: Optional[Callable] = None
        self._error_oracle: Optional[Callable] = None
        # Vectorized versions (used when oracles are built from data)
        self._vectorized_features: Optional[Callable] = None
        self._vectorized_errors: Optional[Callable] = None
        self._supports_batch: Optional[bool] = None

    # ============================================================================
    # Feature Oracle Management
    # ============================================================================

    def set_oracle(self, _features_oracle: Callable[[int, NDArray[np.float64], Dict[str, Any]], NDArray[np.float64]]) -> None:
        """Set user-supplied feature extraction function."""
        self._features_oracle = _features_oracle
        self._vectorized_features = None  # Clear vectorized version
        self.validate_oracle()

    def validate_oracle(self) -> None:
        """Validate that features_oracle returns shape (num_features,)."""
        if self.comm_manager is not None and self.comm_manager.rank != 0:
            return
            
        if self._features_oracle is None:
            raise RuntimeError("_features_oracle function is not set.")
        
        test_bundle = np.ones(self.num_items)
        if self.input_data is not None:
            test_features = self._features_oracle(0, test_bundle, self.input_data)
        elif self.local_data is not None:
            agent_data = self.local_data.get("agent_data", {})
            modular = agent_data.get("modular")
            if modular is not None and modular.shape[0] > 0:
                test_features = self._features_oracle(0, test_bundle, self.local_data)
            else:
                return
        else:
            return
        
        assert test_features.shape == (self.num_features,), \
            f"features_oracle must return shape ({self.num_features},), got {test_features.shape}"

    # ============================================================================
    # Error Oracle Management
    # ============================================================================

    def set_error_oracle(self, _error_oracle: Callable[[int, NDArray[np.float64], Dict[str, Any]], float]) -> None:
        """Set user-supplied error oracle function."""
        self._error_oracle = _error_oracle
        self._vectorized_errors = None  # Clear vectorized version

    def error_oracle(self, agent_id: int, bundle: NDArray[np.float64], 
                    data_override: Optional[Dict[str, Any]] = None) -> float:
        """Compute error for single agent/bundle."""
        if self._error_oracle is None:
            raise RuntimeError("_error_oracle function is not set. Call build_error_oracle_from_data() first.")
        data = data_override if data_override is not None else self.local_data
        return self._error_oracle(agent_id, bundle, data)

    def build_error_oracle_from_data(self) -> Callable[[int, NDArray[np.float64], Dict[str, Any]], float]:
        """
        Build modular error oracle from data['errors'] with vectorized support.
        
        Returns:
            Generated error_oracle function: (agent_id, bundle, data) -> float
        """
        if self._error_oracle is not None:
            logger.info("Rebuilding error oracle (overwriting existing)")
        
        # Single-agent oracle (for compatibility with user-defined scenarios)
        def modular_error_oracle(agent_id: int, bundle: NDArray[np.float64], data: Dict[str, Any]) -> float:
            errors = data.get("errors")
            if errors is None:
                return 0.0
            return float((errors[agent_id] * bundle).sum())
        
        # Vectorized oracle (for efficient batch computation)
        def vectorized_modular_errors(bundles: NDArray[np.float64], data: Dict[str, Any]) -> NDArray[np.float64]:
            errors = data.get("errors")
            if errors is None:
                return np.zeros(len(bundles), dtype=np.float64)
            return (errors * bundles).sum(axis=1)
        
        self._error_oracle = modular_error_oracle
        self._vectorized_errors = vectorized_modular_errors
        return modular_error_oracle

    # ============================================================================
    # Feature Computation
    # ============================================================================

    def features_oracle(self, agent_id: int, bundle: NDArray[np.float64], 
                       data_override: Optional[Dict[str, Any]] = None) -> NDArray[np.float64]:
        """Compute features for single agent/bundle."""
        if self._features_oracle is None:
            raise RuntimeError("_features_oracle function is not set.")
        data = data_override if data_override is not None else self.input_data
        return self._features_oracle(agent_id, bundle, data)

    def _check_batch_support(self) -> bool:
        """Check if oracle supports batch computation (cached)."""
        if self._supports_batch is not None:
            return self._supports_batch
        
        try:
            test_bundles = np.zeros((2, self.num_items))
            result = self._features_oracle(None, test_bundles, self.local_data)
            self._supports_batch = (
                isinstance(result, np.ndarray) and 
                result.shape == (2, self.num_features)
            )
        except Exception:
            self._supports_batch = False
        
        return self._supports_batch

    def compute_rank_features(self, local_bundles: Optional[NDArray[np.float64]]) -> NDArray[np.float64]:
        """Compute features for all local agents on this rank. Uses vectorized if available."""
        if self.num_local_agents == 0 or local_bundles is None or len(local_bundles) == 0:
            return np.empty((0, self.num_features), dtype=np.float64)
        
        assert self.num_local_agents == len(local_bundles), \
            f"num_local_agents ({self.num_local_agents}) != len(local_bundles) ({len(local_bundles)})"
        
        # Use vectorized version if available (built from data)
        if self._vectorized_features is not None:
            return self._vectorized_features(local_bundles, self.local_data)
        
        # Fallback to batch support check (user-defined oracle)
        if self._check_batch_support():
            return self._features_oracle(None, local_bundles, self.local_data)
        
        # Final fallback: loop over agents
        return np.stack([self.features_oracle(i, local_bundles[i], self.local_data) 
                        for i in range(self.num_local_agents)])

    def compute_gathered_features(self, local_bundles: Optional[NDArray[np.float64]]) -> Optional[NDArray[np.float64]]:
        """Compute features for all agents, gather to rank 0."""
        features_local = self.compute_rank_features(local_bundles)
        return self.comm_manager.concatenate_array_at_root_fast(features_local, root=0)

    def compute_rank_errors(self, local_bundles: Optional[NDArray[np.float64]]) -> NDArray[np.float64]:
        """Compute errors for all local agents on this rank. Uses vectorized if available."""
        if self.num_local_agents == 0 or local_bundles is None or len(local_bundles) == 0:
            return np.empty(0, dtype=np.float64)
        
        if self._error_oracle is None:
            raise RuntimeError("_error_oracle function is not set. Call build_error_oracle_from_data() first.")
        
        # Use vectorized version if available (built from data)
        if self._vectorized_errors is not None:
            return self._vectorized_errors(local_bundles, self.local_data)
        
        # Fallback: loop over agents
        return np.array([self._error_oracle(i, local_bundles[i], self.local_data) 
                        for i in range(self.num_local_agents)])

    def compute_gathered_features_and_errors(self, local_bundles: NDArray[np.float64]) -> Tuple[Optional[NDArray[np.float64]], Optional[NDArray[np.float64]]]:
        """
        Compute features and errors for all agents in one pass, gather to rank 0.
        
        Returns:
            Tuple of (features, errors) arrays, both None on non-root ranks
        """
        features_local = self.compute_rank_features(local_bundles)
        errors_local = self.compute_rank_errors(local_bundles)
        features = self.comm_manager.concatenate_array_at_root_fast(features_local, root=0)
        errors = self.comm_manager.concatenate_array_at_root_fast(errors_local, root=0)
        return features, errors

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

    # ============================================================================
    # Auto-Generated Oracle Builders
    # ============================================================================

    def build_oracles_from_data(self) -> None:
        """
        Auto-build both feature and error oracles from data structure.
        
        Builds feature oracle from modular/quadratic keys in data.
        Builds modular error oracle from data['errors'].
        """
        self.build_features_oracle_from_data()
        self.build_error_oracle_from_data()

    # Backward compatibility alias
    def build_from_data(self, build_error_oracle: bool = True) -> Callable:
        """Alias for build_features_oracle_from_data (backward compatibility)."""
        return self.build_features_oracle_from_data(build_error_oracle=build_error_oracle)

    def build_features_oracle_from_data(self, build_error_oracle: bool = True) -> Callable[[int, NDArray[np.float64], Dict[str, Any]], NDArray[np.float64]]:
        """
        Build features_oracle from data structure (modular/quadratic keys) with vectorized support.
        Also builds error_oracle automatically if build_error_oracle=True.
        
        Args:
            build_error_oracle: If True, also builds modular error oracle
        
        Also auto-sets feature names from data sources if not already set.
        
        Returns:
            Generated features_oracle function
        """
        if self._features_oracle is not None:
            logger.info("Rebuilding feature oracle (overwriting existing)")
        
        self.data_manager.verify_feature_count()
        
        if self.is_root():
            data_info = self.data_manager.get_data_info(self.input_data)
            flags = (
                data_info["has_modular_agent"],
                data_info["has_quadratic_agent"],
                data_info["has_modular_item"],
                data_info["has_quadratic_item"]
            )
            
            # Auto-set feature names from data sources if not already set
            if self.dimensions_cfg.feature_names is None:
                auto_names = self.data_manager.get_feature_names_from_data()
                if auto_names:
                    self.dimensions_cfg.feature_names = auto_names
                    logger.info(f"Auto-set {len(auto_names)} feature names from data sources")
        else:
            flags = None
        
        flags = self.comm_manager.broadcast_from_root(flags, root=0)
        has_modular_agent, has_quadratic_agent, has_modular_item, has_quadratic_item = flags

        # Build single-agent oracle using closure
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
        
        # Build vectorized oracle using closure
        def make_vectorized_features(ma, qa, mi, qi):
            def vectorized_features(bundles: NDArray[np.float64], data: Dict[str, Any]) -> NDArray[np.float64]:
                """Vectorized feature computation for all agents at once."""
                feats = []
                if ma:
                    # modular: (num_agents, num_items, num_features), bundles: (num_agents, num_items)
                    modular = data['agent_data']['modular']
                    feats.append(np.einsum('ijk,ij->ik', modular, bundles))
                if qa:
                    # quadratic: (num_agents, num_items, num_items, num_features)
                    quadratic = data['agent_data']['quadratic']
                    feats.append(np.einsum('ijlk,ij,il->ik', quadratic, bundles, bundles))
                if mi:
                    # modular: (num_items, num_features), bundles: (num_agents, num_items)
                    modular = data['item_data']['modular']
                    feats.append(np.einsum('jk,ij->ik', modular, bundles))
                if qi:
                    # quadratic: (num_items, num_items, num_features)
                    quadratic = data['item_data']['quadratic']
                    feats.append(np.einsum('jlk,ij,il->ik', quadratic, bundles, bundles))
                return np.concatenate(feats, axis=1)
            return vectorized_features
        
        self._features_oracle = make_features_oracle(has_modular_agent, has_quadratic_agent, has_modular_item, has_quadratic_item)
        self._vectorized_features = make_vectorized_features(has_modular_agent, has_quadratic_agent, has_modular_item, has_quadratic_item)
        
        # Also build error oracle automatically
        if build_error_oracle and self._error_oracle is None:
            self.build_error_oracle_from_data()
        
        return self._features_oracle

