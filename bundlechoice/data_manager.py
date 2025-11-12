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

    # ============================================================================
    # Data Loading
    # ============================================================================

    def load(self, input_data: Dict[str, Any]) -> None:
        """Load input data (validated, stored on rank 0 only)."""
        logger.info("Loading input data.")
        self._validate_input_data(input_data)
        if self.is_root():
            self.input_data = input_data
        else:
            self.input_data = None

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
            
            # Merge constraint_mask into agent_data if present
            if "constraint_mask" in self.input_data and self.input_data["constraint_mask"] is not None:
                agent_data = agent_data.copy() if agent_data else {}
                agent_data["constraint_mask"] = self.input_data["constraint_mask"]
            
            # Compute agent indices for each rank
            idx_chunks = np.array_split(np.arange(self.num_simuls * self.num_agents), self.comm_size)
            counts = [len(idx) for idx in idx_chunks]
            
            # Prepare data for scattering
            has_agent_data = len(agent_data) > 0
            has_obs_bundles = obs_bundles is not None
            has_item_data = item_data is not None
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
        
        # Broadcast metadata flags and counts (small, pickle is fine)
        counts, has_agent_data, has_obs_bundles, has_item_data = self.comm_manager.broadcast_from_root(
            (counts, has_agent_data, has_obs_bundles, has_item_data), root=0
        )
        
        # Get local count for this rank
        num_local_agents = counts[self.comm_manager.rank]
        
        # Scatter errors using buffer-based method (5-20x faster than pickle)
        # Need to multiply counts by num_items for 2D arrays (shape: [agents, items])
        flat_counts = [c * self.num_items for c in counts]
        local_errors_flat = self.comm_manager.scatter_array(
            send_array=errors, counts=flat_counts, root=0, 
            dtype=errors.dtype if self.is_root() else np.float64
        )
        local_errors = local_errors_flat.reshape(num_local_agents, self.num_items)
        
        # Scatter obs_bundles if present - use buffer scatter_array for speed
        if has_obs_bundles:
            if self.is_root():
                # Create obs_chunks the old way
                obs_chunks = []
                for idx in idx_chunks:
                    obs_chunks.append(obs_bundles[idx % self.num_agents])
                # Concatenate for scatter_array
                indexed_obs_bundles = np.concatenate(obs_chunks, axis=0)
            else:
                indexed_obs_bundles = None
            
            local_obs_bundles_flat = self.comm_manager.scatter_array(
                send_array=indexed_obs_bundles, counts=flat_counts, root=0,
                dtype=indexed_obs_bundles.dtype if self.is_root() else np.bool_
            )
            local_obs_bundles = local_obs_bundles_flat.reshape(num_local_agents, self.num_items)
        else:
            local_obs_bundles = None
        
        # Scatter agent_data dict if present - use pickle scatter for flexibility
        if has_agent_data:
            # Create chunks the old way
            if self.is_root():
                agent_data_chunks = []
                for idx in idx_chunks:
                    chunk_dict = {
                        k: array[idx % self.num_agents] for k, array in agent_data.items()
                    }
                    agent_data_chunks.append(chunk_dict)
            else:
                agent_data_chunks = None
            
            local_agent_data = self.comm_manager.comm.scatter(agent_data_chunks, root=0)
        else:
            local_agent_data = None
        
        # Broadcast item_data (already optimized with buffer-based broadcast_dict)
        if has_item_data:
            item_data = self.comm_manager.broadcast_dict(item_data, root=0)
        else:
            item_data = None
        
        # Build local data structure
        self.local_data = {
            "item_data": item_data,
            "agent_data": local_agent_data,
            "errors": local_errors,
            "obs_bundles": local_obs_bundles,
        }
        self.num_local_agents = num_local_agents 

    # --- Helper Methods ---
    def _prepare_errors(self, errors: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if (self.num_simuls == 1 and errors.ndim == 2):
            return errors
        elif errors.ndim == 3:
            return errors.reshape(-1, self.num_items)
        else:
            raise ValueError(f"errors has shape {errors.shape}, while num_simuls is {self.num_simuls} and num_agents is {self.num_agents}")


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