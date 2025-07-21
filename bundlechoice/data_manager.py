import numpy as np
from typing import Optional, Dict, Any, List
from bundlechoice.utils import get_logger
from bundlechoice.config import DimensionsConfig
from mpi4py import MPI
logger = get_logger(__name__)

class DataManager:
    """
    Handles input data distribution and conversion for the bundle choice model.
    Supports MPI-based data scattering and PyTorch conversion.

    Expected input_data structure:
        {
            'item_data': dict[str, np.ndarray],
            'agent_data': dict[str, np.ndarray],
            'errors': np.ndarray,
            'obs_bundle': np.ndarray,
        }
    Each chunk sent to a rank contains:
        {
            'agent_indices': np.ndarray,
            'agent_data': dict[str, np.ndarray] or None,
            'errors': np.ndarray or None
        }
    """

    def __init__(self, 
                 dimensions_cfg: DimensionsConfig,
                 comm: MPI.Comm,
                 input_data: Optional[Dict[str, Any]] = None
                 ) -> None:
        """
        Initialize the DataManager.
        """
        self.dimensions_cfg = dimensions_cfg
        self.comm = comm
        self.input_data = input_data
        self.rank = comm.Get_rank() if comm is not None else None
        self.comm_size = comm.Get_size() if comm is not None else None
        self.local_data: Optional[Dict[str, Any]] = None
        self.num_local_agents: Optional[int] = None
        self.using_torch: bool = False

    # --- Properties ---
    @property
    def num_agents(self):
        return self.dimensions_cfg.num_agents if self.dimensions_cfg else None

    @property
    def num_items(self):
        return self.dimensions_cfg.num_items if self.dimensions_cfg else None

    @property
    def num_features(self):
        return self.dimensions_cfg.num_features if self.dimensions_cfg else None

    @property
    def num_simuls(self):
        return self.dimensions_cfg.num_simuls if self.dimensions_cfg else None

    # --- Helper Methods ---
    def _scatter_common_setup(self) -> None:
        """Common validation and setup for scatter methods."""
        if self.dimensions_cfg is None:
            logger.error("dimensions_cfg must be set before calling scatter.")
            raise ValueError("dimensions_cfg must be set before calling scatter.")
        if self.input_data is None and self.rank == 0:
            logger.error("input_data must be set on rank 0 before calling scatter.")
            raise ValueError("input_data must be set on rank 0 before calling scatter.")
        self._validate_num_ranks_vs_problems()

    def _prepare_errors(self, errors: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Prepare errors for scattering, handling reshaping automatically."""
        if errors is None:
            return None
        if (self.num_simuls == 1 and 
            self.num_agents is not None and 
            self.num_items is not None and 
            errors.ndim == 2):
            if errors.shape == (self.num_agents, self.num_items):
                return errors.reshape(1, self.num_agents, self.num_items)
        return errors

    def _create_agent_chunks(self, agent_data: Optional[Dict], errors: Optional[np.ndarray], 
                            obs_bundle: Optional[np.ndarray] = None, 
                            use_modulo: bool = False) -> Optional[List[Dict]]:
        """Create chunks for agent data distribution."""
        if self.num_agents is None:
            return None
        
        all_indices = np.arange(self.num_agents)
        chunks = np.array_split(all_indices, self.comm_size)
        
        return [
            {
                "num_local_agents": len(indices),
                "agent_data": {k: v[indices] for k, v in agent_data.items()} if agent_data else None,
                "errors": errors[indices, :] if errors is not None else None,
                "obs_bundle": obs_bundle[indices, :] if obs_bundle is not None else None,
            }
            for indices in chunks
        ]

    def _create_simulated_agent_chunks(self, agent_data: Optional[Dict], 
                                     errors: Optional[np.ndarray]) -> Optional[List[Dict]]:
        """Create chunks for simulated agent data distribution."""
        if self.num_simuls is None or self.num_agents is None:
            return None
        
        total_agents = self.num_simuls * self.num_agents
        all_indices = np.arange(total_agents)
        chunks = np.array_split(all_indices, self.comm_size)
        
        return [
            {
                "num_local_agents": len(indices),
                "agent_data": {k: v[indices % self.num_agents] for k, v in agent_data.items()} 
                              if agent_data else None,
                "errors": errors[indices, :] if errors is not None else None,
            }
            for indices in chunks
        ]

    def _build_local_data(self, local_chunk: Optional[Dict], item_data: Optional[Dict]) -> Dict[str, Any]:
        """Build local_data dictionary from chunk and item_data."""
        return {
            "item_data": item_data,
            "agent_data": local_chunk.get("agent_data") if local_chunk else None,
            "errors": local_chunk.get("errors") if local_chunk else None,
            "obs_bundle": local_chunk.get("obs_bundle") if local_chunk else None,
        }

    def validate_input_data(self, input_data: Dict[str, Any]) -> None:
        """
        Validate that input_data has the required structure and dimensions.
        Only validates keys that are present in input_data.
        
        Args:
            input_data: Dictionary containing the input data to validate.
            
        Raises:
            ValueError: If input_data structure or dimensions don't match expectations.
        """
        if self.dimensions_cfg is None:
            raise ValueError("dimensions_cfg must be set before validating input_data.")
        
        # Validate agent_data if present
        agent_data = input_data.get("agent_data")
        if agent_data is not None:
            for key, value in agent_data.items():
                if not isinstance(value, np.ndarray):
                    raise ValueError(f"agent_data[{key}] must be a numpy array, got {type(value)}")
                if value.shape[0] != self.num_agents:
                    raise ValueError(f"agent_data[{key}] has leading dimension {value.shape[0]}, expected {self.num_agents}")
        
        # Validate item_data if present
        item_data = input_data.get("item_data")
        if item_data is not None:
            for key, value in item_data.items():
                if not isinstance(value, np.ndarray):
                    raise ValueError(f"item_data[{key}] must be a numpy array, got {type(value)}")
                if value.shape[0] != self.num_items:
                    raise ValueError(f"item_data[{key}] has leading dimension {value.shape[0]}, expected {self.num_items}")
        
        # Validate and reshape errors if present
        errors = input_data.get("errors")
        if errors is not None:
            if not isinstance(errors, np.ndarray):
                raise ValueError(f"errors must be a numpy array, got {type(errors)}")
            
            # Check if errors need reshaping
            reshaped_errors = self._prepare_errors(errors)
            if reshaped_errors is not None:
                input_data["errors"] = reshaped_errors
        
        # Optional: validate obs_bundle if present
        obs_bundle = input_data.get("obs_bundle")
        if obs_bundle is not None:
            if not isinstance(obs_bundle, np.ndarray):
                raise ValueError(f"obs_bundle must be a numpy array, got {type(obs_bundle)}")
            expected_shape = (self.num_agents, self.num_items)
            if obs_bundle.shape != expected_shape:
                raise ValueError(f"obs_bundle has shape {obs_bundle.shape}, expected {expected_shape}")

    def _validate_num_ranks_vs_problems(self):
        """
        Raise ValueError if the number of MPI ranks exceeds num_agents * num_simuls.
        """
        if self.num_agents is not None and self.num_simuls is not None:
            total_problems = self.num_agents * self.num_simuls
            num_ranks = self.comm.Get_size() if self.comm is not None else 1
            if num_ranks > total_problems:
                raise ValueError(f"Number of MPI ranks ({num_ranks}) exceeds number of agent problems (num_agents * num_simuls = {total_problems}). Reduce the number of ranks or increase the number of agents/simulations.")

    # --- Data loading ---
    def load_input_data(self, input_data: Dict[str, Any]) -> None:
        """
        Load or update the input data dictionary after initialization.
        Validates the input data structure and dimensions before loading.

        Args:
            input_data: Dictionary of input data.
        """
        logger.info("Loading input data.")
        self.validate_input_data(input_data)
        self.input_data = input_data

    def scatter_data_single_error(self, seed: Optional[int] = 42) -> None:
        """
        Distribute input data across MPI ranks with a single random error per agent.
        Similar to scatter_data but picks one random error realization per agent.
        Sets up local and global data attributes for each rank.
        
        Args:
            seed: Optional seed for random error selection. Defaults to 42 for reproducibility.
        
        Raises:
            ValueError: If dimensions_cfg is not set, or if input_data is not set on rank 0.
            RuntimeError: If no data chunk is received after scatter.
        """
        self._scatter_common_setup()

        if self.rank == 0:
            agent_data = self.input_data.get("agent_data")
            error_s_i_j = self.input_data.get("errors")
            
            # Pick random error realization for each agent
            if error_s_i_j is not None and self.num_simuls is not None and self.num_agents is not None:
                np.random.seed(seed)  # For reproducibility
                random_simul_indices = np.random.randint(0, self.num_simuls, size=self.num_agents)
                error_i_j = error_s_i_j[random_simul_indices, np.arange(self.num_agents), :]
            else:
                error_i_j = None
                
            obs_bundle = self.input_data.get("obs_bundle")
            agent_data_chunks = self._create_agent_chunks(agent_data, error_i_j, obs_bundle)
            item_data = self.input_data.get("item_data")
        else:
            agent_data_chunks = None
            item_data = None

        local_chunk = self.comm.scatter(agent_data_chunks, root=0)
        item_data = self.comm.bcast(item_data, root=0)
     
        self.local_data = self._build_local_data(local_chunk, item_data)
        self.num_local_agents = local_chunk["num_local_agents"]

    # --- Data scattering ---
    def scatter_data(self) -> None:
        """
        Distribute input data across MPI ranks.
        Sets up local and global data attributes for each rank.
        Expects input_data to have keys: 'item_data', 'agent_data', 'errors', 'obs_bundle'.
        Each chunk sent to a rank contains: 'agent_indices', 'agent_data', 'errors'.

        Raises:
            ValueError: If dimensions_cfg is not set, or if input_data is not set on rank 0.
            RuntimeError: If no data chunk is received after scatter.
        """
        self._scatter_common_setup()

        if self.rank == 0:
            agent_data = self.input_data.get("agent_data")
            error_s_i_j = self.input_data.get("errors")
            if error_s_i_j is not None and self.num_items is not None:
                error_si_j = error_s_i_j.reshape(-1, self.num_items)
            else:
                error_si_j = None
            obs_bundle = self.input_data.get("obs_bundle")
            
            agent_data_chunks = self._create_simulated_agent_chunks(agent_data, error_si_j)
            item_data = self.input_data.get("item_data")
        else:
            agent_data_chunks = None
            item_data = None

        local_chunk = self.comm.scatter(agent_data_chunks, root=0)
        item_data = self.comm.bcast(item_data, root=0)
     
        self.local_data = self._build_local_data(local_chunk, item_data)
        self.num_local_agents = local_chunk["num_local_agents"] 