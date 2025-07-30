import numpy as np
from typing import Any, Callable, Optional
from bundlechoice.utils import get_logger
from bundlechoice.config import DimensionsConfig
from bundlechoice.data_manager import DataManager
from mpi4py import MPI
from bundlechoice.base import HasDimensions, HasData, HasComm
logger = get_logger(__name__)

class FeatureManager(HasDimensions, HasComm, HasData):
    """
    Encapsulates feature extraction logic for the bundle choice model.
    User supplies get_features(i_id, B_j, data); this class provides get_x_i_k and get_features.
    Dynamically references num_agents and num_simuls from the provided config.
    """

    def __init__(   self, 
                    dimensions_cfg: DimensionsConfig, 
                    comm: MPI.Comm, 
                    data_manager: DataManager
                    ):
        """
        Initialize the FeatureManager.

        Args:
            dimensions_cfg (DimensionsConfig): Configuration object with num_agents, num_items, num_features, num_simuls.
            comm (MPI.Comm): MPI communicator.
            data_manager (DataManager): DataManager instance.
        """
        self.dimensions_cfg = dimensions_cfg
        self.comm = comm
        self.data_manager = data_manager
        self.features_oracle = None
        # self.rank and self.comm_size now provided by HasComm

    def load(self, features_oracle):
        """
        Load a user-supplied feature extraction function.

        Args:
            features_oracle (Callable): Function (i_id, B_j, data) -> np.ndarray.
        """
        self.features_oracle = features_oracle

    # --- Feature extraction methods ---
    def get_features(self, i_id, B_j, data_override=None):
        """
        Compute features for a single agent/bundle using the user-supplied function.
        By default, uses input_data from the FeatureManager. If data_override is provided, uses that instead (for local/MPI calls).

        Args:
            i_id (int): Agent index.
            B_j (array-like): Bundle.
            data_override (dict, optional): Data dictionary to override default input_data.
        Returns:
            np.ndarray: Feature vector for the agent/bundle.
        Raises:
            RuntimeError: If features_oracle function is not set or data is missing required keys.
        """
        if self.features_oracle is None:
            raise RuntimeError("features_oracle function is not set.")
        if data_override is None:
            data = self.input_data
        else:
            data = data_override
       
        return self.features_oracle(i_id, B_j, data)

    def get_all_agent_features(self, B_i_j: Any) -> Optional[np.ndarray]:
        """
        Compute features for all agents. Only works on rank 0; returns None on other ranks.

        Args:
            B_i_j (array-like): List/array of bundles for each agent.
        Returns:
            np.ndarray or None: Features for all agents (on rank 0), None on other ranks.
        Raises:
            ValueError: If num_agents is not set.
        """
        if self.rank != 0:
            return None
        if self.num_agents is None:
            raise ValueError("num_agents must be set in dimensions_cfg before calling get_all_agent_features.")
        return np.stack([self.get_features(i, B_i_j[i]) for i in range(self.num_agents)])

    def get_all_simulated_agent_features(self, B_si_j: Any) -> Optional[np.ndarray]:
        """
        Compute features for all simulated agents. Only works on rank 0; returns None on other ranks.

        Args:
            B_si_j (array-like): List/array of bundles for all simulated agents.
        Returns:
            np.ndarray or None: Features for all simulated agents (on rank 0), None on other ranks.
        Raises:
            ValueError: If num_agents or num_simuls is not set.
        """
        if self.rank != 0:
            return None
        if self.num_agents is None or self.num_simuls is None:
            raise ValueError("num_agents and num_simuls must be set in dimensions_cfg before calling get_all_simulated_agent_features.")

        return np.stack([
            self.get_features(si % self.num_agents, B_si_j[si])
            for si in range(self.num_simuls * self.num_agents)
        ])

    def get_local_agents_features(self, B_local):
        """
        Compute features for all local agents (on this MPI rank only).

        Args:
            B_local (array-like): List or array of bundles for local agents (length = num_local_agents).
        Returns:
            np.ndarray: Features for all local agents on this rank (shape: num_local_agents x num_features).
        """
        assert self.num_local_agents == len(B_local), "num_local_agents and B_local must have the same length."
        features_local = [
            self.get_features(local_id, B_local[local_id], data_override=self.local_data)
            for local_id in range(self.num_local_agents)
        ]
        features_local = np.array(features_local)
        return features_local
    
    def get_all_simulated_agent_features_MPI(self, B_local):
        """
        Compute features for all simulated agents in parallel using MPI.
        Gathers and concatenates all local results on rank 0.

        Args:
            B_local (array-like): List or array of bundles for local agents (length = num_local_agents).
        Returns:
            np.ndarray or None: Features for all simulated agents (on rank 0), None on other ranks.
        Raises:
            ValueError: If comm is not set.
        """
        if self.comm is None:
            raise ValueError("MPI communicator (comm) must be set before calling get_all_simulated_agent_features_MPI.")
        features_local = self.get_local_agents_features(B_local)
        features = self.comm.gather(features_local, root=0)
        if self.rank == 0:
            return np.concatenate(features)
        else:
            return None

    # --- Feature oracle builder ---
    def build_from_data(self):
        """
        Dynamically build and return a get_features function based on the structure of input_data.
        Inspects agent_data and item_data for 'modular' and 'quadratic' keys and builds an efficient function.
        Sets self.features_oracle to the new function.

        Returns:
            Callable: The new features_oracle function.
        """
        self.data_manager.validate_standard_inputdata()
        if self.rank == 0:
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
        has_modular_agent = self.comm.bcast(has_modular_agent, root=0)
        has_quadratic_agent = self.comm.bcast(has_quadratic_agent, root=0)
        has_modular_item = self.comm.bcast(has_modular_item, root=0)
        has_quadratic_item = self.comm.bcast(has_quadratic_item, root=0)

        code_lines = ["def get_features(i_id, B_j, data):", "    feats = []"]
        if has_modular_agent:
            code_lines.append("    modular = data['agent_data']['modular'][i_id]")
            code_lines.append("    feats.append(np.einsum('jk,j->k', modular, B_j))")
        if has_quadratic_agent:
            code_lines.append("    quadratic = data['agent_data']['quadratic'][i_id]")
            code_lines.append("    feats.append(np.einsum('jlk,j,l->k', quadratic, B_j, B_j))")
        if has_modular_item:
            code_lines.append("    modular = data['item_data']['modular']")
            code_lines.append("    feats.append(np.einsum('jk,j->k', modular, B_j))")
        if has_quadratic_item:
            code_lines.append("    quadratic = data['item_data']['quadratic']")
            code_lines.append("    feats.append(np.einsum('jlk,j,l->k', quadratic, B_j, B_j))")
        code_lines.append("    return np.concatenate(feats)")
        code_str = "\n".join(code_lines)

        namespace = {}
        exec(code_str, {"np": np}, namespace)
        features_oracle = namespace["get_features"]
        self.features_oracle = features_oracle
        return features_oracle 
            

