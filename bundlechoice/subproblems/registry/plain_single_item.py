from bundlechoice.subproblems.base import BatchSubproblemBase
import numpy as np
from typing import Optional, Any

class PlainSingleItemSubproblem(BatchSubproblemBase):
    """
    Subproblem that, for each agent/simulation, selects the single item with the maximum value.
    """
    def initialize(self) -> None:
        agent_data = self.local_data.get("agent_data", {})
        item_data = self.local_data.get("item_data", {})

        # Modular and quadratic features (None if missing)
        self.has_modular_agent = "modular" in agent_data
        self.has_modular_item = "modular" in item_data

        self.has_errors = "errors" in self.local_data
        self.has_constraint_mask = "constraint_mask" in self.local_data


    def solve(self, theta: np.ndarray, pb: Optional[Any] = None) -> np.ndarray:
        """
        For each agent, select the single item with the maximum utility, respecting constraint_mask.
        Args:
            theta (np.ndarray): Parameter vector.
            pb (Any, optional): Problem object (unused).
        Returns:
            np.ndarray: Boolean array of shape (num_local_agents, num_items) with True at the max item (if utility > 0).
        """
        U_i_j = self.build_utilities(theta)
        if self.has_constraint_mask:
            U_i_j = np.where(self.local_data["constraint_mask"], U_i_j, -np.inf)
        j_star = np.argmax(U_i_j, axis=1)
        max_vals = U_i_j[np.arange(self.num_local_agents), j_star]
        # Vectorized one-hot: only set True if max utility > 0
        optimal_bundles = (max_vals > 0)[:, None] & (np.arange(self.num_items) == j_star[:, None])
        return optimal_bundles

    def build_utilities(self, theta: np.ndarray) -> np.ndarray:
        """
        Build the utility matrix for all agents and items, handling missing features gracefully.
        Args:
            theta (np.ndarray): Parameter vector.
        Returns:
            np.ndarray: Utility matrix of shape (num_local_agents, num_items).
        """
        U_i_j = np.zeros((self.num_local_agents, self.num_items))
        offset = 0
        if self.has_modular_agent:
            modular_agent = self.local_data["agent_data"]["modular"]
            num_mod_agent = modular_agent.shape[-1]
            U_i_j += modular_agent @ theta[offset:offset + num_mod_agent]
            offset += num_mod_agent
        if self.has_modular_item:
            modular_item = self.local_data["item_data"]["modular"]
            num_mod_item = modular_item.shape[-1]
            U_i_j += modular_item @ theta[offset:offset + num_mod_item]
            offset += num_mod_item
        if self.has_errors:
            U_i_j += self.local_data["errors"]
        return U_i_j
 