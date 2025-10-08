import numpy as np
from typing import Any, Callable, Optional, Dict
from numpy.typing import NDArray
from bundlechoice.utils import get_logger
from bundlechoice.config import DimensionsConfig
from bundlechoice.data_manager import DataManager
from bundlechoice.base import HasDimensions, HasData, HasComm
logger = get_logger(__name__)

class FeatureManager(HasDimensions, HasComm, HasData):
    """
    Encapsulates feature extraction logic for the bundle choice model.
    User supplies features_oracle(agent_id, bundle, data); this class provides features_oracle and related methods.
    Dynamically references num_agents and num_simuls from the provided config.
    """

    def __init__(   self, 
                    dimensions_cfg: DimensionsConfig, 
                    comm_manager, 
                    data_manager: DataManager
                    ):
        """
        Initialize the FeatureManager.

        Args:
            dimensions_cfg (DimensionsConfig): Configuration object with num_agents, num_items, num_features, num_simuls.
            comm_manager: Communication manager for MPI operations.
            data_manager (DataManager): DataManager instance.
        """
        self.dimensions_cfg = dimensions_cfg
        self.comm_manager = comm_manager
        self.data_manager = data_manager
        self._features_oracle = None
        self._supports_batch = None

        self.num_global_agents = self.num_simuls * self.num_agents

    def set_oracle(self, _features_oracle: Callable[[int, NDArray[np.float64], Dict[str, Any]], NDArray[np.float64]]) -> None:
        """
        Load a user-supplied feature extraction function.

        Args:
            _features_oracle: Function with signature (agent_id, bundle, data) -> features array.
        """
        self._features_oracle = _features_oracle
        self.validate_oracle()

    def validate_oracle(self) -> None:
        """Validate that the features oracle returns the expected shape."""
        # check that features_oracle returns array of shape (num_features,)
        if self._features_oracle is not None:
            test_bundle = np.ones(self.num_items)
            if self.input_data is not None:
                test_features = self._features_oracle(0, test_bundle, self.input_data)
            elif self.local_data is not None:
                test_features = self._features_oracle(0, test_bundle, self.local_data)
            else:
                raise RuntimeError("No data to validate features_oracle.")
            assert test_features.shape == (self.num_features,), f"features_oracle must return array of shape {self.num_features}. Got {test_features.shape} instead."
        else:
            raise RuntimeError("_features_oracle function is not set.")

    # --- Feature extraction methods ---
    def features_oracle(self, agent_id: int, bundle: NDArray[np.float64], data_override: Optional[Dict[str, Any]] = None) -> NDArray[np.float64]:
        """
        Compute features for a single agent/bundle using the user-supplied function.
        By default, uses input_data from the FeatureManager. If data_override is provided, uses that instead (for local/MPI calls).

        Args:
            agent_id: Agent index.
            bundle: Bundle selection array.
            data_override: Data dictionary to override default input_data.
        Returns:
            Feature vector for the agent/bundle.
        Raises:
            RuntimeError: If _features_oracle function is not set or data is missing required keys.
        """
        if self._features_oracle is None:
            raise RuntimeError("_features_oracle function is not set.")
        if data_override is None:
            data = self.input_data
        else:
            data = data_override
       
        return self._features_oracle(agent_id, bundle, data)

    def _check_batch_support(self) -> bool:
        """Check if oracle supports batch computation across agents."""
        if self._supports_batch is not None:
            return self._supports_batch
        
        try:
            test_bundles = np.zeros((2, self.num_items))
            result = self._features_oracle(None, test_bundles, self.local_data)
            self._supports_batch = (
                isinstance(result, np.ndarray) and 
                result.shape == (2, self.num_features)
            )
        except:
            self._supports_batch = False
        
        return self._supports_batch

    def compute_rank_features(self, local_bundles: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute features for all local agents (on this MPI rank only).
        Automatically uses batch computation if oracle supports it.

        Args:
            local_bundles: Array of bundles for local agents (shape: num_local_agents x num_items).
        Returns:
            Features for all local agents on this rank (shape: num_local_agents x num_features).
        """
        assert self.num_local_agents == len(local_bundles), f"num_local_agents and local_bundles must have the same length. Bundle shape: {local_bundles.shape} while num_local_agents: {self.num_local_agents}"
        data = self.local_data
        
        # Try batch computation first
        if self._check_batch_support():
            return self._features_oracle(None, local_bundles, data)
        
        # Fallback to per-agent computation
        return np.stack([self.features_oracle(i, local_bundles[i], data) for i in range(self.num_local_agents)])

    def compute_gathered_features(self, local_bundles: NDArray[np.float64]) -> Optional[NDArray[np.float64]]:
        """
        Compute features for all simulated agents in parallel using MPI.
        Gathers and concatenates all local results on rank 0.

        Args:
            local_bundles: Array of bundles for local agents (shape: num_local_agents x num_items).
        Returns:
            Features for all simulated agents (on rank 0), None on other ranks.
        """
        features_local = self.compute_rank_features(local_bundles)
        return self.comm_manager.concatenate_array_at_root_fast(features_local, root=0)


    def compute_gathered_utilities(self, local_bundles: NDArray[np.float64], theta: NDArray[np.float64]) -> Optional[NDArray[np.float64]]:
        """
        Compute utilities for all simulated agents in parallel using MPI.
        Gathers and concatenates all local results on rank 0.

        Args:
            local_bundles: Array of bundles for local agents.
            theta: Parameter vector.
        Returns:
            Utilities for all simulated agents (on rank 0), None on other ranks.
        """
        features_local = self.compute_rank_features(local_bundles)
        errors_local = (self.data_manager.local_data["errors"]* local_bundles).sum(1)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            utilities_local = features_local @ theta + errors_local
        return self.comm_manager.concatenate_array_at_root_fast(utilities_local, root=0)
    

    def compute_gathered_errors(self, local_bundles: NDArray[np.float64]) -> Optional[NDArray[np.float64]]:
        """
        Compute errors for all simulated agents in parallel using MPI.
        Gathers and concatenates all local results on rank 0.

        Args:
            local_bundles: Array of bundles for local agents.
        Returns:
            Errors for all simulated agents (on rank 0), None on other ranks.
        """
        errors_local = (self.data_manager.local_data["errors"]* local_bundles).sum(1)
        return self.comm_manager.concatenate_array_at_root_fast(errors_local, root=0)


    # --- Feature oracle builder ---
    def build_from_data(self) -> Callable[[int, NDArray[np.float64], Dict[str, Any]], NDArray[np.float64]]:
        """
        Dynamically build and return a features_oracle function based on the structure of input_data.
        Inspects agent_data and item_data for 'modular' and 'quadratic' keys and builds an efficient function.
        Sets self._features_oracle to the new function.

        Returns:
            The new _features_oracle function.
        """
        if self._features_oracle is not None:
            logger.info("Rebuilding feature oracle (overwriting existing)")
        
        self.data_manager.validate_quadratic_input_data()
        if self.is_root():
            input_data = self.input_data
            agent_data = input_data.get("agent_data")
            item_data = input_data.get("item_data")
            if agent_data is not None:
                has_modular_agent = "modular" in agent_data
                has_quadratic_agent = "quadratic" in agent_data
            else:
                has_modular_agent = None
                has_quadratic_agent = None
            if item_data is not None:
                has_modular_item = "modular" in item_data
                has_quadratic_item = "quadratic" in item_data
            else:
                has_modular_item = None
                has_quadratic_item = None
        else:
            has_modular_agent = None
            has_quadratic_agent = None
            has_modular_item = None
            has_quadratic_item = None
        has_modular_agent = self.comm_manager.broadcast_from_root(has_modular_agent, root=0)
        has_quadratic_agent = self.comm_manager.broadcast_from_root(has_quadratic_agent, root=0)
        has_modular_item = self.comm_manager.broadcast_from_root(has_modular_item, root=0)
        has_quadratic_item = self.comm_manager.broadcast_from_root(has_quadratic_item, root=0)

        code_lines = ["def features_oracle(agent_id, bundle, data):", "    feats = []"]
        if has_modular_agent:
            code_lines.append("    modular = data['agent_data']['modular'][agent_id]")
            code_lines.append("    feats.append(np.einsum('jk,j->k', modular, bundle))")
        if has_quadratic_agent:
            code_lines.append("    quadratic = data['agent_data']['quadratic'][agent_id]")
            code_lines.append("    feats.append(np.einsum('jlk,j,l->k', quadratic, bundle, bundle))")
        if has_modular_item:
            code_lines.append("    modular = data['item_data']['modular']")
            code_lines.append("    feats.append(np.einsum('jk,j->k', modular, bundle))")
        if has_quadratic_item:
            code_lines.append("    quadratic = data['item_data']['quadratic']")
            code_lines.append("    feats.append(np.einsum('jlk,j,l->k', quadratic, bundle, bundle))")
        code_lines.append("    return np.concatenate(feats)")
        code_str = "\n".join(code_lines)

        namespace = {}
        exec(code_str, {"np": np}, namespace)
        _features_oracle = namespace["features_oracle"]
        self._features_oracle = _features_oracle
        return _features_oracle 
            

