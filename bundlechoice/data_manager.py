import numpy as np
from typing import Optional, Dict, Any, List
from bundlechoice.utils import get_logger
from bundlechoice.config import DimensionsConfig
from bundlechoice.base import HasDimensions, HasComm
logger = get_logger(__name__)

class DataManager(HasDimensions, HasComm):
    """
    Handles input data distribution and conversion for the bundle choice model.
    Supports MPI-based data scattering and PyTorch conversion.

    Expected input_data structure::
        {
            'item_data': dict[str, np.ndarray],
            'agent_data': dict[str, np.ndarray],
            'errors': np.ndarray,
            'obs_bundle': np.ndarray,
        }
    Each chunk sent to a rank contains::
        {
            'agent_indices': np.ndarray,
            'agent_data': dict[str, np.ndarray] or None,
            'errors': np.ndarray or None
        }
    """

    def __init__(self, dimensions_cfg: DimensionsConfig, comm_manager) -> None:
        """
        Initialize the DataManager.

        Args:
            dimensions_cfg (DimensionsConfig): Configuration for problem dimensions.
            comm_manager: Communication manager for MPI operations.
        """
        self.dimensions_cfg = dimensions_cfg
        self.comm_manager = comm_manager
        self.input_data = None
        self.local_data: Optional[Dict[str, Any]] = None
        self.num_local_agents: Optional[int] = None
  
    # --- Data loading ---
    def load(self, input_data: Dict[str, Any]) -> None:
        """
        Load or update the input data dictionary after initialization.
        Validates the input data structure and dimensions before loading.

        Args:
            input_data (dict): Dictionary of input data.
        """
        logger.info("Loading input data.")
        self._validate_input_data(input_data)
        if self.is_root():
            self.input_data = input_data
        else:
            self.input_data = None

    def load_and_scatter(self, input_data: Dict[str, Any]) -> None:
        """ 
        Load input data and scatter it across MPI ranks.

        Args:
            input_data (dict): Dictionary of input data.
        """
        self._validate_input_data(input_data)
        self.load(input_data)
        self.scatter()

        return self.local_data

    # --- Data scattering ---

    def scatter(self) -> None:
        """
        Distribute input data across MPI ranks using buffer-based MPI (5-20x faster).
        Sets up local and global data attributes for each rank.
        Expects input_data to have keys: 'item_data', 'agent_data', 'errors', 'obs_bundle'.
        Each chunk sent to a rank contains: 'agent_indices', 'agent_data', 'errors'.

        Raises:
            ValueError: If dimensions_cfg is not set, or if input_data is not set on rank 0.
            RuntimeError: If no data chunk is received after scatter.
        """
        # Prepare data and compute chunk indices on root
        if self.is_root():
            errors = self._prepare_errors(self.input_data.get("errors"))
            obs_bundles = self.input_data.get("obs_bundle")
            agent_data = self.input_data.get("agent_data")
            item_data = self.input_data.get("item_data")
            
            # Compute agent indices for each rank
            idx_chunks = np.array_split(np.arange(self.num_simuls * self.num_agents), self.comm_size)
            counts = [len(idx) for idx in idx_chunks]
            
            # Prepare data for scattering
            has_agent_data = agent_data is not None
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
        
        # Scatter obs_bundles if present
        if has_obs_bundles:
            if self.is_root():
                # Index obs_bundles properly (modulo for simulations)
                indexed_obs_bundles = np.vstack([obs_bundles[idx % self.num_agents] for idx in idx_chunks])
            else:
                indexed_obs_bundles = None
            
            local_obs_bundles_flat = self.comm_manager.scatter_array(
                send_array=indexed_obs_bundles, counts=flat_counts, root=0,
                dtype=indexed_obs_bundles.dtype if self.is_root() else np.bool_
            )
            local_obs_bundles = local_obs_bundles_flat.reshape(num_local_agents, self.num_items)
        else:
            local_obs_bundles = None
        
        # Scatter constraint_mask if present
        has_constraint_mask = self.is_root() and "constraint_mask" in self.input_data and self.input_data["constraint_mask"] is not None
        if has_constraint_mask:
            if self.is_root():
                constraint_masks = self.input_data["constraint_mask"]
                # constraint_mask is a list of arrays, one per agent
                # Split into chunks for each rank
                all_chunks = []
                for chunk_indices in np.array_split(np.arange(self.num_simuls * self.num_agents), self.comm_size):
                    chunk_masks = [constraint_masks[idx % self.num_agents] for idx in chunk_indices]
                    all_chunks.append(chunk_masks)
            else:
                all_chunks = None
            # Scatter as a simple list (each rank gets its subset)
            local_constraint_mask = self.comm_manager.comm.scatter(all_chunks, root=0)
        else:
            local_constraint_mask = None
        
        # Scatter agent_data dict if present using buffer-based scatter_dict
        if has_agent_data:
            if self.is_root():
                # Index agent_data properly (modulo for simulations)
                indexed_agent_data = {}
                for key, array in agent_data.items():
                    indexed_agent_data[key] = (np.concatenate if array.ndim == 1 else np.vstack)([array[idx % self.num_agents] for idx in idx_chunks])
            else:
                indexed_agent_data = None
            
            local_agent_data = self.comm_manager.scatter_dict(
                data_dict=indexed_agent_data, counts=counts, root=0
            )
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
        
        # Add constraint_mask if present
        if has_constraint_mask:
            self.local_data["constraint_mask"] = local_constraint_mask
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
        """
        Validate that input_data has the required structure and dimensions.
        Uses comprehensive validation utilities for better error messages.

        Args:
            input_data (dict): Dictionary containing the input data to validate.
        Raises:
            ValidationError: If input_data structure or dimensions don't match expectations.
        """ 
        if self.is_root():
            from bundlechoice.validation import validate_input_data_comprehensive
            validate_input_data_comprehensive(input_data, self.dimensions_cfg)

    def verify_feature_count(self):
        """Verify that num_features in config matches data structure."""
        if self.is_root():
            from bundlechoice.validation import validate_feature_count
            validate_feature_count(self.input_data, self.num_features)