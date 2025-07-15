import numpy as np
from typing import Any, Callable, Optional
from bundlechoice.v2.utils import get_logger
logger = get_logger(__name__)

class FeatureManager:
    """
    Encapsulates feature extraction logic for the bundle choice model.
    User supplies get_features(i_id, B_j, data); this class provides get_x_i_k and get_features.
    Dynamically references num_agents and num_simuls from the provided config.
    """

    def __init__(self, data_manager, dimensions_cfg, get_features=None, comm=None):
        """
        Initialize the FeatureManager.

        Args:
            input_data: The input data dictionary.
            dimensions_cfg: The configuration object with num_agents, num_items, num_features, num_simuls.
            get_features: Optional user-supplied function (i_id, B_j, data) -> np.ndarray.
            comm: Optional MPI communicator.
        """
        self.data_manager = data_manager
        self.dimensions_cfg = dimensions_cfg
        self.user_get_features = get_features
        self.comm = comm
        self.rank = comm.Get_rank() if comm is not None else 0

    # --- Properties ---
    @property
    def num_agents(self):
        """Number of agents in the dataset."""
        return self.dimensions_cfg.num_agents if self.dimensions_cfg else None

    @property
    def num_items(self):
        """Number of items in the dataset."""
        return self.dimensions_cfg.num_items if self.dimensions_cfg else None

    @property
    def num_features(self):
        """Number of features in the dataset."""
        return self.dimensions_cfg.num_features if self.dimensions_cfg else None

    @property
    def num_simuls(self):
        """Number of simulations in the dataset."""
        return self.dimensions_cfg.num_simuls if self.dimensions_cfg else None

    @property
    def input_data(self):
        """Input data dictionary."""
        return self.data_manager.input_data
    
    @property
    def local_data(self):
        """Local data dictionary."""
        return self.data_manager.local_data

    # --- Feature extraction methods ---
    def get_features(self, i_id, B_j, data_override=None):
        """
        Compute features for a single agent/bundle using the user-supplied function.
        By default, uses input_data from the FeatureManager. If data_override is provided, uses that instead (for local/MPI calls).

        Args:
            i_id: Agent index
            B_j: Bundle (array-like)
            data_override: Optional data dictionary to override default input_data
        Returns:
            np.ndarray: Feature vector for the agent/bundle
        Raises:
            RuntimeError: If get_features function is not set or data is missing required keys.
        """
        if self.user_get_features is None:
            raise RuntimeError("get_features function is not set.")
        if data_override is not None:
            data = data_override
        else:
            data = self.input_data
        if data is None or (data.get('agent_data') is None and data.get('item_data') is None):
            raise RuntimeError("DataManager/input_data has neither agent_data nor item_data. If running under MPI, call scatter_data() before using features. If running locally, check input_data.")
        return self.user_get_features(i_id, B_j, data)

    def get_all_agent_features(self, B_i_j: Any) -> Optional[np.ndarray]:
        """
        Compute features for all agents. Only works on rank 0; returns None on other ranks.

        Args:
            B_i_j: List/array of bundles for each agent
        Returns:
            np.ndarray: Features for all agents (on rank 0), None on other ranks
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
            B_si_j: List/array of bundles for all simulated agents
        Returns:
            np.ndarray: Features for all simulated agents (on rank 0), None on other ranks
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
            B_local: List or array of bundles for local agents (length = num_local_agents)
        Returns:
            np.ndarray: Features for all local agents on this rank (shape: num_local_agents x num_features)
        """
        num_local_agents = self.data_manager.num_local_agents
        assert num_local_agents == len(B_local), "num_local_agents and B_local must have the same length."
        features_local = [
            self.get_features(local_id, B_local[local_id], data_override=self.local_data)
            for local_id in range(num_local_agents)
        ]
        features_local = np.array(features_local)
        return features_local
    
    def get_all_simulated_agent_features_MPI(self, B_local):
        """
        Compute features for all simulated agents in parallel using MPI.
        Gathers and concatenates all local results on rank 0.

        Args:
            B_local: List or array of bundles for local agents (length = num_local_agents)
        Returns:
            np.ndarray: Features for all simulated agents (on rank 0), None on other ranks
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
    def build_feature_oracle_from_data(self):
        """
        Dynamically build and return a get_features function based on the structure of input_data.
        Inspects agent_data and item_data for 'modular' and 'quadratic' keys and builds an efficient function.
        Sets self.user_get_features to the new function.

        Returns:
            The new get_features function
        """
        input_data = self.input_data
        agent_data = input_data["agent_data"]
        item_data = input_data["item_data"]

        code_lines = ["def get_features(i_id, B_j, data):", "    feats = []"]
        if "modular" in agent_data:
            code_lines.append("    modular = data['agent_data']['modular'][i_id]")
            code_lines.append("    feats.append(np.einsum('jk,j->k', modular, B_j))")
        if "quadratic" in agent_data:
            code_lines.append("    quadratic = data['agent_data']['quadratic'][i_id]")
            code_lines.append("    feats.append(np.einsum('jlk,j,l->k', quadratic, B_j, B_j))")
        if "modular" in item_data:
            code_lines.append("    modular = data['item_data']['modular']")
            code_lines.append("    feats.append(np.einsum('jk,j->k', modular, B_j))")
        if "quadratic" in item_data:
            code_lines.append("    quadratic = data['item_data']['quadratic']")
            code_lines.append("    feats.append(np.einsum('jlk,j,l->k', quadratic, B_j, B_j))")
        code_lines.append("    return np.concatenate(feats)")
        code_str = "\n".join(code_lines)

        namespace = {}
        exec(code_str, {"np": np}, namespace)
        get_features = namespace["get_features"]
        self.user_get_features = get_features
        return get_features 

    def compute_features_obs_bundle(self):
        if self.rank == 0:
            return self.get_all_agent_features(self.input_data["obs_bundle"])
        else:
            return None