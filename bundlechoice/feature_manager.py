import numpy as np
from typing import Any, Callable, Optional, Dict, Tuple
from numpy.typing import NDArray
import sys
from bundlechoice.utils import get_logger
from bundlechoice.config import DimensionsConfig
from bundlechoice.data_manager import DataManager
from bundlechoice.base import HasDimensions, HasData, HasComm

# Try to import tracemalloc for memory profiling (optional)
try:
    import tracemalloc
    TRACEMALLOC_AVAILABLE = True
except ImportError:
    TRACEMALLOC_AVAILABLE = False
    tracemalloc = None

logger = get_logger(__name__)


# ============================================================================
# FeatureManager
# ============================================================================

class FeatureManager(HasDimensions, HasComm, HasData):
    """
    Manages feature extraction for bundle choice model.
    
    User supplies features_oracle(agent_id, bundle, data) function.
    Supports batch computation when oracle supports it.
    """

    def __init__(self, dimensions_cfg: DimensionsConfig, comm_manager, data_manager: DataManager) -> None:
        """Initialize FeatureManager."""
        self.dimensions_cfg = dimensions_cfg
        self.comm_manager = comm_manager
        self.data_manager = data_manager
        self._features_oracle: Optional[Callable] = None
        self._supports_batch: Optional[bool] = None
        self.num_global_agents = self.num_simulations * self.num_agents

    # ============================================================================
    # Oracle Management
    # ============================================================================

    def set_oracle(self, _features_oracle: Callable[[int, NDArray[np.float64], Dict[str, Any]], NDArray[np.float64]]) -> None:
        """Set user-supplied feature extraction function."""
        self._features_oracle = _features_oracle
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
        except:
            self._supports_batch = False
        
        return self._supports_batch

    def compute_rank_features(self, local_bundles: Optional[NDArray[np.float64]]) -> NDArray[np.float64]:
        """Compute features for all local agents on this rank. Uses batch if supported."""
        if self.num_local_agents == 0 or local_bundles is None or len(local_bundles) == 0:
            return np.empty((0, self.num_features), dtype=np.float64)
        
        assert self.num_local_agents == len(local_bundles), \
            f"num_local_agents ({self.num_local_agents}) != len(local_bundles) ({len(local_bundles)})"
        
        if self._check_batch_support():
            return self._features_oracle(None, local_bundles, self.local_data)
        
        return np.stack([self.features_oracle(i, local_bundles[i], self.local_data) 
                        for i in range(self.num_local_agents)])
    
    def compute_all_features_on_root(self, all_bundles: NDArray[np.float64]) -> Optional[NDArray[np.float64]]:
        """
        Compute features for all agents on root rank using input_data.
        
        This method is used for combined gather optimization where bundles are already
        gathered to root, avoiding the need for a separate feature gather operation.
        
        Args:
            all_bundles: Gathered bundles array of shape (num_simulations * num_agents, num_items)
            
        Returns:
            Features array of shape (num_simulations * num_agents, num_features) on root, None on other ranks
        """
        if not self.is_root():
            return None
        
        if all_bundles is None or len(all_bundles) == 0:
            return np.empty((0, self.num_features), dtype=np.float64)
        
        # Verify we have input_data on root
        if self.data_manager.input_data is None:
            raise RuntimeError("input_data not available on root rank for compute_all_features_on_root")
        
        num_total_agents = self.num_simulations * self.num_agents
        assert len(all_bundles) == num_total_agents, \
            f"Expected {num_total_agents} bundles, got {len(all_bundles)}"
        
        # Check if batch computation is supported
        if self._check_batch_support():
            # Use batch computation with input_data
            return self._features_oracle(None, all_bundles, self.data_manager.input_data)
        
        # Fallback: compute features one agent at a time
        # Map global agent index to (simulation, agent) pair
        features_list = []
        for global_idx in range(num_total_agents):
            agent_id = global_idx % self.num_agents
            features_list.append(self.features_oracle(agent_id, all_bundles[global_idx], self.data_manager.input_data))
        
        return np.stack(features_list)

    def compute_gathered_features(self, local_bundles: Optional[NDArray[np.float64]], 
                                  timing_dict: Optional[Dict[str, float]] = None) -> Optional[NDArray[np.float64]]:
        """
        Compute features for all agents, gather to rank 0.
        
        Args:
            local_bundles: Local bundles array
            timing_dict: Optional dict to record computation and communication times
            
        Returns:
            Gathered features array (rank 0 only, None on other ranks)
        """
        from datetime import datetime
        
        # Time computation separately
        t_comp_start = datetime.now()
        features_local = self.compute_rank_features(local_bundles)
        comp_time = (datetime.now() - t_comp_start).total_seconds()
        
        # Time communication separately with memory profiling
        t_comm_start = datetime.now()
        tracemalloc_started = False
        if timing_dict is not None and TRACEMALLOC_AVAILABLE:
            if not tracemalloc.is_tracing():
                tracemalloc.start()
                tracemalloc_started = True
        
        if self.is_root():
            print("DEBUG: compute_gathered_features: about to call concatenate_array_at_root_fast", flush=True)
            sys.stdout.flush()
        
        features_gathered = self.comm_manager.concatenate_array_at_root_fast(features_local, root=0)
        
        if self.is_root():
            print("DEBUG: compute_gathered_features: concatenate_array_at_root_fast returned", flush=True)
            sys.stdout.flush()
        comm_time = (datetime.now() - t_comm_start).total_seconds()
        
        if timing_dict is not None and TRACEMALLOC_AVAILABLE and tracemalloc_started:
            current, peak = tracemalloc.get_traced_memory()
            timing_dict['gather_features_memory_peak_mb'] = peak / 1024 / 1024
            tracemalloc.stop()
        
        # Record timing if dict provided
        if timing_dict is not None:
            timing_dict['compute_features'] = comp_time
            timing_dict['gather_features'] = comm_time
            if features_local is not None and len(features_local) > 0:
                data_size = features_local.nbytes
                timing_dict['gather_features_size'] = data_size
                if comm_time > 0:
                    timing_dict['gather_features_bandwidth_mbps'] = (data_size / comm_time) / 1e6
        
        return features_gathered

    def compute_gathered_features_and_errors(self, local_bundles: NDArray[np.float64]) -> Tuple[Optional[NDArray[np.float64]], Optional[NDArray[np.float64]]]:
        """
        Compute features and errors for all agents in one pass, gather to rank 0.
        
        Returns:
            Tuple of (features, errors) arrays, both None on non-root ranks
        """
        features_local = self.compute_rank_features(local_bundles)
        errors_local = (self.data_manager.local_data["errors"] * local_bundles).sum(1)
        features = self.comm_manager.concatenate_array_at_root_fast(features_local, root=0)
        errors = self.comm_manager.concatenate_array_at_root_fast(errors_local, root=0)
        return features, errors

    def compute_gathered_utilities(self, local_bundles: NDArray[np.float64], 
                                   theta: NDArray[np.float64]) -> Optional[NDArray[np.float64]]:
        """Compute utilities for all agents, gather to rank 0."""
        features_local = self.compute_rank_features(local_bundles)
        errors_local = (self.data_manager.local_data["errors"] * local_bundles).sum(1)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            utilities_local = features_local @ theta + errors_local
        return self.comm_manager.concatenate_array_at_root_fast(utilities_local, root=0)

    def compute_gathered_errors(self, local_bundles: NDArray[np.float64],
                                timing_dict: Optional[Dict[str, float]] = None) -> Optional[NDArray[np.float64]]:
        """
        Compute errors for all agents, gather to rank 0.
        
        Args:
            local_bundles: Local bundles array
            timing_dict: Optional dict to record computation and communication times
            
        Returns:
            Gathered errors array (rank 0 only, None on other ranks)
        """
        from datetime import datetime
        
        # Time computation separately
        t_comp_start = datetime.now()
        errors_local = (self.data_manager.local_data["errors"] * local_bundles).sum(1)
        comp_time = (datetime.now() - t_comp_start).total_seconds()
        
        # Time communication separately with memory profiling
        t_comm_start = datetime.now()
        tracemalloc_started = False
        if timing_dict is not None and TRACEMALLOC_AVAILABLE:
            if not tracemalloc.is_tracing():
                tracemalloc.start()
                tracemalloc_started = True
        
        if self.is_root():
            print("DEBUG: compute_gathered_errors: about to call concatenate_array_at_root_fast", flush=True)
            sys.stdout.flush()
        
        errors_gathered = self.comm_manager.concatenate_array_at_root_fast(errors_local, root=0)
        
        if self.is_root():
            print("DEBUG: compute_gathered_errors: concatenate_array_at_root_fast returned", flush=True)
            sys.stdout.flush()
        comm_time = (datetime.now() - t_comm_start).total_seconds()
        
        if timing_dict is not None and TRACEMALLOC_AVAILABLE and tracemalloc_started:
            current, peak = tracemalloc.get_traced_memory()
            timing_dict['gather_errors_memory_peak_mb'] = peak / 1024 / 1024
            tracemalloc.stop()
        
        # Record timing if dict provided
        if timing_dict is not None:
            timing_dict['compute_errors'] = comp_time
            timing_dict['gather_errors'] = comm_time
            if errors_local is not None and len(errors_local) > 0:
                data_size = errors_local.nbytes
                timing_dict['gather_errors_size'] = data_size
                if comm_time > 0:
                    timing_dict['gather_errors_bandwidth_mbps'] = (data_size / comm_time) / 1e6
        
        return errors_gathered

    # ============================================================================
    # Auto-Generated Oracle Builder
    # ============================================================================

    def build_from_data(self) -> Callable[[int, NDArray[np.float64], Dict[str, Any]], NDArray[np.float64]]:
        """
        Build features_oracle from data structure (modular/quadratic keys).
        
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
        else:
            flags = None
        
        flags = self.comm_manager.broadcast_from_root(flags, root=0)
        has_modular_agent, has_quadratic_agent, has_modular_item, has_quadratic_item = flags

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
            

