import numpy as np
from typing import Optional, Dict, Any, List
from bundlechoice.utils import get_logger
from bundlechoice.config import DimensionsConfig
from mpi4py import MPI
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

    def __init__(self, dimensions_cfg: DimensionsConfig, comm: MPI.Comm) -> None:
        """
        Initialize the DataManager.

        Args:
            dimensions_cfg (DimensionsConfig): Configuration for problem dimensions.
            comm (MPI.Comm): MPI communicator.
        """
        self.dimensions_cfg = dimensions_cfg
        self.comm = comm
        self.input_data = None
        self.local_data: Optional[Dict[str, Any]] = None
        self.num_local_agents: Optional[int] = None
        # self.rank and self.comm_size now provided by HasComm

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
        if self.rank == 0:
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
        Distribute input data across MPI ranks.
        Sets up local and global data attributes for each rank.
        Expects input_data to have keys: 'item_data', 'agent_data', 'errors', 'obs_bundle'.
        Each chunk sent to a rank contains: 'agent_indices', 'agent_data', 'errors'.

        Raises:
            ValueError: If dimensions_cfg is not set, or if input_data is not set on rank 0.
            RuntimeError: If no data chunk is received after scatter.
        """
        if self.rank == 0:
            agent_data = self.input_data.get("agent_data")
            errors = self._prepare_errors(self.input_data.get("errors"))
            obs_bundles = self.input_data.get("obs_bundle")
            agent_data_chunks = self._create_simulated_agent_chunks(agent_data, errors, obs_bundles)
            item_data = self.input_data.get("item_data")
            
        else:
            agent_data_chunks = None
            item_data = None

        local_chunk = self.comm.scatter(agent_data_chunks, root=0)
        item_data = self.comm.bcast(item_data, root=0)
     
        self.local_data = self._build_local_data(local_chunk, item_data)
        self.num_local_agents = local_chunk["num_local_agents"] 

    # --- Helper Methods ---
    def _prepare_errors(self, errors: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if (self.num_simuls == 1 and errors.ndim == 2):
            return errors
        elif errors.ndim == 3:
            return errors.reshape(-1, self.num_items)
        else:
            raise ValueError(f"errors has shape {errors.shape}, while num_simuls is {self.num_simuls} and num_agents is {self.num_agents}")

    def _create_simulated_agent_chunks(self, agent_data: Optional[Dict], 
                                     errors: Optional[np.ndarray],
                                     obs_bundles: Optional[np.ndarray]) -> Optional[List[Dict]]:
        """
        Create chunks for simulated agent data distribution.

        Args:
            agent_data (dict or None): Agent data dictionary.
            errors (np.ndarray or None): Error array.
        Returns:
            list of dict: List of data chunks for each MPI rank.
        """
        idx_chunks = np.array_split(np.arange(self.num_simuls * self.num_agents), self.comm_size)
        return  [
                {
                "num_local_agents": len(idx),
                "agent_data": {k: v[idx % self.num_agents] for k, v in agent_data.items()} if agent_data else None,
                "errors": errors[idx, :],
                "obs_bundles": obs_bundles[idx % self.num_agents, :] if obs_bundles is not None else None,
                }
                for idx in idx_chunks
                ]

    def _build_local_data(self, local_chunk: Optional[Dict], item_data: Optional[Dict]) -> Dict[str, Any]:
        """
        Build local_data dictionary from chunk and item_data.

        Args:
            local_chunk (dict or None): Data chunk for this rank.
            item_data (dict or None): Item data dictionary.
        Returns:
            dict: Local data dictionary for this rank.
        """
        return {
                "item_data": item_data,
                "agent_data": local_chunk.get("agent_data"),
                "errors": local_chunk.get("errors"),
                "obs_bundles": local_chunk.get("obs_bundles"),
                }

    def _validate_input_data(self, input_data: Dict[str, Any]) -> None:
        """
        Validate that input_data has the required structure and dimensions.
        Only validates keys that are present in input_data.

        Args:
            input_data (dict): Dictionary containing the input data to validate.
        Raises:
            ValueError: If input_data structure or dimensions don't match expectations.
        """ 
        if self.rank == 0:
            agent_data = input_data.get("agent_data")
            if agent_data is not None:
                for key, value in agent_data.items():
                    if not isinstance(value, np.ndarray):
                        raise ValueError(f"agent_data[{key}] must be a numpy array, got {type(value)}")
                    if value.shape[0] != self.num_agents:
                        raise ValueError(f"agent_data[{key}] has leading dimension {value.shape[0]}, expected {self.num_agents}")
            
            item_data = input_data.get("item_data")
            if item_data is not None:
                for key, value in item_data.items():
                    if not isinstance(value, np.ndarray):
                        raise ValueError(f"item_data[{key}] must be a numpy array, got {type(value)}")
                    if value.shape[0] != self.num_items:
                        raise ValueError(f"item_data[{key}] has leading dimension {value.shape[0]}, expected {self.num_items}")
            
            errors = input_data.get("errors")
            if errors is not None:
                if not isinstance(errors, np.ndarray):
                    raise ValueError(f"errors must be a numpy array, got {type(errors)}")
                if errors.ndim == 2:
                    if errors.shape != (self.num_agents, self.num_items):
                        raise ValueError(f"errors has shape {errors.shape}, expected ({self.num_agents}, {self.num_items})")
                elif errors.ndim == 3:
                    if errors.shape != (self.num_simuls, self.num_agents, self.num_items):
                        raise ValueError(f"errors has shape {errors.shape}, expected ({self.num_simuls}, {self.num_agents}, {self.num_items})")
                else:
                    raise ValueError(f"errors has {errors.ndim} dimensions, expected 2 or 3")
            else:
                raise ValueError("errors must be set")
                
            obs_bundle = input_data.get("obs_bundle")
            if obs_bundle is not None:
                if not isinstance(obs_bundle, np.ndarray):
                    raise ValueError(f"obs_bundle must be a numpy array, got {type(obs_bundle)}")
                expected_shape = (self.num_agents, self.num_items)
                if obs_bundle.shape != expected_shape:
                    raise ValueError(f"obs_bundle has shape {obs_bundle.shape}, expected {expected_shape}")

    def validate_standard_inputdata(self):
        """
        Validates that the shapes of modular and quadratic data match the expected dimensions.
        Checks:
        - modular item data shape matches (num_items, num_modular_item_features)
        - modular agent data shape matches (num_agents, num_items, num_modular_agent_features)
        - num_features equals the sum of all feature dimensions
        Raises ValueError if any check fails.
        """
        input_data = self.input_data
        if self.rank == 0:
            dimensions_cfg = self.dimensions_cfg
            agent_data = input_data.get("agent_data", {})
            item_data = input_data.get("item_data", {})
            num_agents = dimensions_cfg.num_agents
            num_items = dimensions_cfg.num_items
            num_features = dimensions_cfg.num_features
            # Modular agent features
            modular_agent = agent_data.get("modular")
            num_modular_agent_features = 0
            if modular_agent is not None:
                if modular_agent.shape[0] != num_agents or modular_agent.shape[1] != num_items:
                    raise ValueError(f"modular agent data shape {modular_agent.shape} does not match (num_agents, num_items, ...)")
                num_modular_agent_features = modular_agent.shape[2] if modular_agent.ndim == 3 else 0
            # Modular item features
            modular_item = item_data.get("modular")
            num_modular_item_features = 0
            if modular_item is not None:
                if modular_item.shape[0] != num_items:
                    raise ValueError(f"modular item data shape {modular_item.shape} does not match (num_items, ...)")
                num_modular_item_features = modular_item.shape[1] if modular_item.ndim == 2 else 0
            # Quadratic item features
            quadratic_item = item_data.get("quadratic")
            num_quadratic_item_features = 0
            if quadratic_item is not None:
                if quadratic_item.shape[0] != num_items or quadratic_item.shape[1] != num_items:
                    raise ValueError(f"quadratic item data shape {quadratic_item.shape} does not match (num_items, num_items, ...)")
                num_quadratic_item_features = quadratic_item.shape[2] if quadratic_item.ndim == 3 else 0
            # Quadratic agent features
            quadratic_agent = agent_data.get("quadratic")
            num_quadratic_agent_features = 0
            if quadratic_agent is not None:
                if quadratic_agent.shape[0] != num_agents or quadratic_agent.shape[1] != num_items or quadratic_agent.shape[2] != num_items:
                    raise ValueError(f"quadratic agent data shape {quadratic_agent.shape} does not match (num_agents, num_items, num_items, ...)")
                num_quadratic_agent_features = quadratic_agent.shape[3] if quadratic_agent.ndim == 4 else 0
            # Check num_features
            total_features = num_modular_agent_features + num_modular_item_features + num_quadratic_agent_features + num_quadratic_item_features
            if num_features != total_features:
                raise ValueError(f"num_features ({num_features}) does not match the sum of all feature dimensions ({total_features})")
            return True





 # def scatter_data_single_error(self, seed: Optional[int] = 42) -> None:
    #     """
    #     Distribute input data across MPI ranks with a single random error per agent.
    #     Similar to scatter_data but picks one random error realization per agent.
    #     Sets up local and global data attributes for each rank.

    #     Args:
    #         seed (int, optional): Seed for random error selection. Defaults to 42.
    #     Raises:
    #         ValueError: If dimensions_cfg is not set, or if input_data is not set on rank 0.
    #         RuntimeError: If no data chunk is received after scatter.
    #     """
    #     self._validate_input_data(self.input_data)
    

    #     if self.rank == 0:
    #         agent_data = self.input_data.get("agent_data")
    #         error_s_i_j = self.input_data.get("errors")
            
    #         # Pick random error realization for each agent
    #         if error_s_i_j is not None and self.num_simuls is not None and self.num_agents is not None:
    #             np.random.seed(seed)  # For reproducibility
    #             random_simul_indices = np.random.randint(0, self.num_simuls, size=self.num_agents)
    #             error_i_j = error_s_i_j[random_simul_indices, np.arange(self.num_agents), :]
    #         else:
    #             error_i_j = None
                
    #         obs_bundle = self.input_data.get("obs_bundle")
    #         agent_data_chunks = self._create_agent_chunks(agent_data, error_i_j, obs_bundle)
    #         item_data = self.input_data.get("item_data")
    #     else:
    #         agent_data_chunks = None
    #         item_data = None

    #     local_chunk = self.comm.scatter(agent_data_chunks, root=0)
    #     item_data = self.comm.bcast(item_data, root=0)
     
    #     self.local_data = self._build_local_data(local_chunk, item_data)
    #     self.num_local_agents = local_chunk["num_local_agents"]

    # def _create_agent_chunks(self, 
    #                             agent_data: Optional[Dict], 
    #                             errors: Optional[np.ndarray], 
    #                             obs_bundle: Optional[np.ndarray] = None) -> Optional[List[Dict]]:
    #     """
    #     Create chunks for agent data distribution.

    #     Args:
    #         agent_data (dict or None): Agent data dictionary.
    #         errors (np.ndarray or None): Error array.
    #         obs_bundle (np.ndarray or None): Observed bundles.
    #     Returns:
    #         list of dict: List of data chunks for each MPI rank.
    #     """
    #     all_indices = np.arange(self.num_agents)
    #     chunks = np.array_split(all_indices, self.comm_size)
        
    #     return [
    #         {
    #             "num_local_agents": len(indices),
    #             "agent_data": {k: v[indices] for k, v in agent_data.items()} if agent_data else None,
    #             "errors": errors[indices, :],
    #             "obs_bundle": obs_bundle[indices, :] if obs_bundle is not None else None,
    #         }
    #         for indices in chunks
    #     ]
