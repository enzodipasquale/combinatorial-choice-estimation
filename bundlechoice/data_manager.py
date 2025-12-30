import numpy as np
from typing import Optional, Dict, Any, List
from bundlechoice.utils import get_logger
from bundlechoice.config import DimensionsConfig
from bundlechoice.base import HasDimensions, HasComm
logger = get_logger(__name__)


# ============================================================================
# DataManager
# ============================================================================

class DataManager(HasDimensions, HasComm):
    """
    Handles input data distribution across MPI ranks.
    
    Expected input_data structure:
        {
            'item_data': dict[str, np.ndarray],
            'agent_data': dict[str, np.ndarray],
            'errors': np.ndarray,
            'obs_bundle': np.ndarray,
        }
    """

    def __init__(self, dimensions_cfg: DimensionsConfig, comm_manager) -> None:
        """Initialize DataManager with dimensions config and MPI communicator."""
        self.dimensions_cfg = dimensions_cfg
        self.comm_manager = comm_manager
        self.input_data: Optional[Dict[str, Any]] = None
        self.local_data: Optional[Dict[str, Any]] = None
        self.num_local_agents: Optional[int] = None
        # Cache for get_data_info() results
        self._cached_data_info: Optional[Dict[str, Any]] = None
        self._cached_data_info_source: Optional[str] = None  # 'input' or 'local'

    # ============================================================================
    # Data Loading
    # ============================================================================

    def load(self, input_data: Dict[str, Any]) -> None:
        """Load input data (validated, stored on rank 0 only)."""
        self._validate_input_data(input_data)
        if self.is_root():
            self.input_data = input_data
            # Invalidate cache when input_data changes
            self._cached_data_info = None
            self._cached_data_info_source = None
        else:
            self.input_data = None
            self._cached_data_info = None
            self._cached_data_info_source = None

    def load_and_scatter(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Load input data and scatter across MPI ranks."""
        self._validate_input_data(input_data)
        self.load(input_data)
        self.scatter()
        return self.local_data

    # ============================================================================
    # Data Scattering (MPI)
    # ============================================================================

    def scatter(self) -> None:
        """
        Distribute input data across MPI ranks using buffer-based MPI.
        
        Uses buffer operations (5-20x faster than pickle). Each rank receives:
        - Local agent data chunk
        - Local errors array
        - Broadcast item_data (same on all ranks)
        """
        if self.is_root():
            errors = self._prepare_errors(self.input_data.get("errors"))
            obs_bundles = self.input_data.get("obs_bundle")
            agent_data = self.input_data.get("agent_data") or {}
            item_data = self.input_data.get("item_data")
            
            if "constraint_mask" in self.input_data and self.input_data["constraint_mask"] is not None:
                # Only copy if constraint_mask isn't already in agent_data (avoid unnecessary copy)
                if "constraint_mask" not in agent_data:
                    # Shallow copy of dict structure only (arrays remain references)
                    agent_data = {k: v for k, v in (agent_data.items() if agent_data else [])}
                agent_data["constraint_mask"] = self.input_data["constraint_mask"]
            
            idx_chunks = np.array_split(np.arange(self.num_simulations * self.num_agents), self.comm_size)
            counts = [len(idx) for idx in idx_chunks]
            
            has_agent_data = len(agent_data) > 0
            has_obs_bundles = obs_bundles is not None
            has_item_data = item_data is not None
            
            print("=" * 70)
            print("DATA SCATTER")
            print("=" * 70)
            total_agents = self.num_simulations * self.num_agents
            sim_info = f" ({self.num_simulations} simuls × {self.num_agents} agents)" if self.num_simulations > 1 else ""
            print(f"  Scattering: {total_agents} agents{sim_info} → {self.comm_size} ranks")
            if len(set(counts)) == 1:
                print(f"  Distribution: {counts[0]} agents/rank (balanced)")
            else:
                print(f"  Distribution: {min(counts)}-{max(counts)} agents/rank")
        else:
            errors = None
            obs_bundles = None
            agent_data = None
            item_data = None
            idx_chunks = None
            counts = None
            has_agent_data = False
            has_obs_bundles = False
            has_item_data = False
        
        counts, has_agent_data, has_obs_bundles, has_item_data = self.comm_manager.broadcast_from_root(
            (counts, has_agent_data, has_obs_bundles, has_item_data), root=0
        )
        
        num_local_agents = counts[self.comm_manager.rank]
        
        flat_counts = [c * self.num_items for c in counts]
        if self.is_root():
            print("  Arrays:")
            print(f"    Errors: shape=({self.num_simulations * self.num_agents}, {self.num_items})")
        local_errors_flat = self.comm_manager.scatter_array(
            send_array=errors, counts=flat_counts, root=0, 
            dtype=errors.dtype if self.is_root() else np.float64
        )
        local_errors = local_errors_flat.reshape(num_local_agents, self.num_items)
        
        if has_obs_bundles:
            if self.is_root():
                print(f"    Obs_bundles: shape=({self.num_agents}, {self.num_items})")
            if self.is_root():
                # Optimized: use advanced indexing instead of list comprehension + concatenate
                # Concatenate all index chunks, compute modulo (vectorized), then index
                all_indices = np.concatenate(idx_chunks)
                agent_indices = all_indices % self.num_agents
                indexed_obs_bundles = obs_bundles[agent_indices]
            else:
                indexed_obs_bundles = None
            
            local_obs_bundles_flat = self.comm_manager.scatter_array(
                send_array=indexed_obs_bundles, counts=flat_counts, root=0,
                dtype=indexed_obs_bundles.dtype if self.is_root() else np.bool_
            )
            local_obs_bundles = local_obs_bundles_flat.reshape(num_local_agents, self.num_items)
        else:
            local_obs_bundles = None
        
        if has_agent_data:
            if self.is_root():
                print("  Dictionaries:")
                keys_str = ", ".join(agent_data.keys())
                print(f"    Agent_data: {len(agent_data)} keys [{keys_str}], {self.num_agents} agents")
                # Prepare agent_data for buffer-based scatter: repeat arrays across simulations
                agent_data_expanded = {}
                for k, array in agent_data.items():
                    # Repeat agent data across simulations if needed
                    if self.num_simulations > 1:
                        if array.ndim == 1:
                            # For 1D arrays, tile along single dimension: (num_agents,) -> (num_simulations * num_agents,)
                            agent_data_expanded[k] = np.tile(array, self.num_simulations)
                        else:
                            # For 2D+ arrays, tile along first dimension: (num_agents, ...) -> (num_simulations, num_agents, ...)
                            agent_data_expanded[k] = np.tile(array, (self.num_simulations, 1) + (1,) * (array.ndim - 2))
                    else:
                        agent_data_expanded[k] = array
            else:
                agent_data_expanded = None
            
            local_agent_data = self.comm_manager.scatter_dict(agent_data_expanded, counts=counts, root=0)
        else:
            local_agent_data = None
        
        if has_item_data:
            if self.is_root():
                if not has_agent_data:
                    print("  Dictionaries:")
                keys_str = ", ".join(item_data.keys())
                print(f"    Item_data: {len(item_data)} keys [{keys_str}], {self.num_items} items")
            item_data = self.comm_manager.broadcast_dict(item_data, root=0)
        else:
            item_data = None
        
        self.local_data = {
            "item_data": item_data,
            "agent_data": local_agent_data,
            "errors": local_errors,
            "obs_bundles": local_obs_bundles,
        }
        self.num_local_agents = num_local_agents
        # Invalidate cache when local_data changes
        self._cached_data_info = None
        self._cached_data_info_source = None
        if self.is_root():
            print(f"  Complete: {num_local_agents} local agents/rank")
            print() 

    # --- Helper Methods ---
    def _prepare_errors(self, errors: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if (self.num_simulations == 1 and errors.ndim == 2):
            return errors
        elif errors.ndim == 3:
            return errors.reshape(-1, self.num_items)
        else:
            raise ValueError(f"errors has shape {errors.shape}, while num_simulations is {self.num_simulations} and num_agents is {self.num_agents}")


    def _validate_input_data(self, input_data: Dict[str, Any]) -> None:
        """Validate input_data structure and dimensions (rank 0 only)."""
        if self.is_root():
            from bundlechoice.validation import validate_input_data_comprehensive
            validate_input_data_comprehensive(input_data, self.dimensions_cfg)

    def verify_feature_count(self) -> None:
        """Verify num_features in config matches data structure."""
        if self.is_root():
            from bundlechoice.validation import validate_feature_count
            validate_feature_count(self.input_data, self.num_features)

    def update_errors(self, errors: np.ndarray) -> None:
        """
        Update errors in local_data by scattering from rank 0.
        
        Args:
            errors: Array of shape (num_agents, num_items) on rank 0, None on other ranks.
        """
        if self.is_root():
            errors_flat = errors.reshape(-1)  # Flatten to (num_agents * num_items,)
            dtype = errors_flat.dtype
            size = self.comm_manager.comm.Get_size()
            idx_chunks = np.array_split(np.arange(self.num_agents), size)
            counts = [len(chunk) * self.num_items for chunk in idx_chunks]
        else:
            errors_flat = None
            dtype = np.float64
            counts = None
        
        local_errors_flat = self.comm_manager.scatter_array(
            errors_flat, counts=counts, root=0, dtype=dtype
        )
        
        if self.num_local_agents == 0:
            self.local_data["errors"] = np.empty((0, self.num_items), dtype=dtype)
        else:
            self.local_data["errors"] = local_errors_flat.reshape(self.num_local_agents, self.num_items)

    def get_data_info(self, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get boolean flags and feature dimensions for data components.
        
        Args:
            data: Data dictionary to check (defaults to local_data).
                 Can be input_data or local_data structure.
        
        Returns:
            Dictionary with flags and dimensions:
            - has_modular_agent: bool
            - has_modular_item: bool
            - has_quadratic_agent: bool
            - has_quadratic_item: bool
            - has_errors: bool
            - has_constraint_mask: bool
            - num_modular_agent: int (0 if not present)
            - num_modular_item: int (0 if not present)
            - num_quadratic_agent: int (0 if not present)
            - num_quadratic_item: int (0 if not present)
        """
        if data is None:
            data = self.local_data
            source = 'local'
        elif data is self.input_data:
            source = 'input'
        else:
            # Custom data dict - don't cache
            source = None
        
        # Return cached result if available and source matches
        if source is not None and self._cached_data_info is not None and self._cached_data_info_source == source:
            return self._cached_data_info
        
        if data is None:
            result = {
                "has_modular_agent": False,
                "has_modular_item": False,
                "has_quadratic_agent": False,
                "has_quadratic_item": False,
                "has_errors": False,
                "has_constraint_mask": False,
                "num_modular_agent": 0,
                "num_modular_item": 0,
                "num_quadratic_agent": 0,
                "num_quadratic_item": 0,
            }
        else:
            agent_data = data.get("agent_data") or {}
            item_data = data.get("item_data") or {}
            
            has_modular_agent = "modular" in agent_data
            has_modular_item = "modular" in item_data
            has_quadratic_agent = "quadratic" in agent_data
            has_quadratic_item = "quadratic" in item_data
            
            result = {
                "has_modular_agent": has_modular_agent,
                "has_modular_item": has_modular_item,
                "has_quadratic_agent": has_quadratic_agent,
                "has_quadratic_item": has_quadratic_item,
                "has_errors": "errors" in data,
                "has_constraint_mask": "constraint_mask" in agent_data or "constraint_mask" in data,
                "num_modular_agent": agent_data["modular"].shape[-1] if has_modular_agent else 0,
                "num_modular_item": item_data["modular"].shape[-1] if has_modular_item else 0,
                "num_quadratic_agent": agent_data["quadratic"].shape[-1] if has_quadratic_agent else 0,
                "num_quadratic_item": item_data["quadratic"].shape[-1] if has_quadratic_item else 0,
            }
        
        # Cache result if source is known
        if source is not None:
            self._cached_data_info = result
            self._cached_data_info_source = source
        
        return result