import numpy as np
from typing import Any, Callable, Optional
from bundlechoice.utils import get_logger
from bundlechoice.config import DimensionsConfig
from bundlechoice.data_manager import DataManager
from bundlechoice.base import HasDimensions, HasData, HasComm
logger = get_logger(__name__)

class FeatureManager(HasDimensions, HasComm, HasData):
    """
    Encapsulates feature extraction logic for the bundle choice model.
    User supplies compute_features(agent_id, bundle, data); this class provides compute_features and related methods.
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
        self.features_oracle = None

        self.num_global_agents = self.num_simuls * self.num_agents

    def set_oracle(self, features_oracle):
        """
        Load a user-supplied feature extraction function.

        Args:
            features_oracle (Callable): Function (agent_id, bundle, data) -> np.ndarray.
        """
        self.features_oracle = features_oracle

    # --- Feature extraction methods ---
    def compute_features(self, agent_id, bundle, data_override=None):
        """
        Compute features for a single agent/bundle using the user-supplied function.
        By default, uses input_data from the FeatureManager. If data_override is provided, uses that instead (for local/MPI calls).

        Args:
            agent_id (int): Agent index.
            bundle (array-like): Bundle.
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
       
        return self.features_oracle(agent_id, bundle, data)

    def compute_rank_features(self, local_bundles):
        """
        Compute features for all local agents (on this MPI rank only).

        Args:
            local_bundles (array-like): List or array of bundles for local agents (length = num_local_agents).
        Returns:
            np.ndarray: Features for all local agents on this rank (shape: num_local_agents x num_features).
        """
        assert self.num_local_agents == len(local_bundles), f"num_local_agents and local_bundles must have the same length. Bundle shape: {local_bundles.shape} while num_local_agents: {self.num_local_agents}"
        data = self.local_data
        return np.stack([self.compute_features(i, local_bundles[i], data) for i in range(self.num_local_agents)])


    # def get_all_0(self, bundles: Any) -> Optional[np.ndarray]:
    #     """
    #     Compute features for all simulated agents. Only works on rank 0; returns None on other ranks.

    #     Args:
    #         bundles (array-like): List/array of bundles for all simulated agents.
    #     Returns:
    #         np.ndarray or None: Features for all simulated agents (on rank 0), None on other ranks.
    #     Raises:
    #         ValueError: If num_agents or num_simuls is not set.
    #     """
    #     if not self.is_root():
    #         return None
    #     assert self.num_global_agents == len(bundles), "num_global_agents and bundles must have the same length."
    #     return np.stack([self.compute_features(id % self.num_agents, bundles[id]) for id in range(self.num_global_agents)])


    def compute_gathered_features(self, local_bundles):
        """
        Compute features for all simulated agents in parallel using MPI.
        Gathers and concatenates all local results on rank 0.

        Args:
            local_bundles (array-like): List or array of bundles for local agents (length = num_local_agents).
        Returns:
            np.ndarray or None: Features for all simulated agents (on rank 0), None on other ranks.
        """
        features_local = self.compute_rank_features(local_bundles)
        return self.comm_manager.concatenate_at_root(features_local, root=0)


    def compute_gathered_utilities(self, local_bundles, theta):
        """
        Compute utilities for all simulated agents in parallel using MPI.
        Gathers and concatenates all local results on rank 0.
        """
        features_local = self.compute_rank_features(local_bundles)
        errors_local = (self.data_manager.local_data["errors"]* local_bundles).sum(1)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            utilities_local = features_local @ theta + errors_local
        return self.comm_manager.concatenate_at_root(utilities_local, root=0)
    

    def compute_gathered_errors(self, local_bundles):
        """
        Compute errors for all simulated agents in parallel using MPI.
        Gathers and concatenates all local results on rank 0.
        """
        errors_local = (self.data_manager.local_data["errors"]* local_bundles).sum(1)
        return self.comm_manager.concatenate_at_root(errors_local, root=0)


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

        code_lines = ["def get_features(agent_id, bundle, data):", "    feats = []"]
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
        features_oracle = namespace["get_features"]
        self.features_oracle = features_oracle
        return features_oracle 
            

