from bundlechoice.v2.subproblems.base import BatchSubproblemBase
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
        self.modular_agent = agent_data.get("modular")
        self.modular_item = item_data.get("modular")

        self.errors = self.local_data.get("errors")
        self.constraint_mask = self.local_data.get("constraint_mask") or np.ones((self.num_local_agents, self.num_items), dtype=bool)

        # Feature counts
        self.num_mod_agent = self.modular_agent.shape[-1] if self.modular_agent is not None else 0
        self.num_mod_item = self.modular_item.shape[-1] if self.modular_item is not None else 0

        # Lambda slices
        offset = 0
        self.lambda_mod_agent_slice = slice(offset, offset + self.num_mod_agent); offset += self.num_mod_agent
        self.lambda_mod_item_slice = slice(offset, offset + self.num_mod_item); offset += self.num_mod_item

    def solve(self, lambda_k: np.ndarray, pb: Optional[Any] = None) -> np.ndarray:
        """
        For each agent, select the single item with the maximum utility, respecting constraint_mask.
        Args:
            lambda_k (np.ndarray): Parameter vector.
            pb (Any, optional): Problem object (unused).
        Returns:
            np.ndarray: Boolean array of shape (num_local_agents, num_items) with True at the max item (if utility > 0).
        """
        U_i_j = self.build_utility_matrix(lambda_k)
        U_i_j_masked = np.where(self.constraint_mask, U_i_j, -np.inf)
        j_star = np.argmax(U_i_j_masked, axis=1)
        max_vals = U_i_j_masked[np.arange(self.num_local_agents), j_star]
        # Vectorized one-hot: only set True if max utility > 0
        optimal_bundles = (max_vals > 0)[:, None] & (np.arange(self.num_items) == j_star[:, None])
        return optimal_bundles

    def build_utility_matrix(self, lambda_k: np.ndarray) -> np.ndarray:
        """
        Build the utility matrix for all agents and items, handling missing features gracefully.
        Args:
            lambda_k (np.ndarray): Parameter vector.
        Returns:
            np.ndarray: Utility matrix of shape (num_local_agents, num_items).
        """
        U_i_j = np.zeros((self.num_local_agents, self.num_items))
        if self.modular_agent is not None:
            U_i_j += self.modular_agent @ lambda_k[self.lambda_mod_agent_slice]
        if self.modular_item is not None:
            U_i_j += self.modular_item @ lambda_k[self.lambda_mod_item_slice]
        if self.errors is not None:
            U_i_j += self.errors
        return U_i_j
 