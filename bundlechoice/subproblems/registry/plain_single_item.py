"""
Plain single-item subproblem solver.

Selects the single item with maximum utility per agent.
"""

from bundlechoice.subproblems.base import BatchSubproblemBase
import numpy as np
from typing import Optional, Any
from numpy.typing import NDArray


# ============================================================================
# Plain Single Item Subproblem Solver
# ============================================================================

class PlainSingleItemSubproblem(BatchSubproblemBase):
    """Selects single item with maximum utility per agent."""
    
    def initialize(self) -> None:
        """Check available features and data."""
        agent_data = {} if self.local_data["agent_data"] is None else self.local_data["agent_data"]
        item_data = {} if self.local_data["item_data"] is None else self.local_data["item_data"]

        self.has_modular_agent = "modular" in agent_data
        self.has_modular_item = "modular" in item_data
        self.has_errors = "errors" in self.local_data
        self.has_constraint_mask = "constraint_mask" in self.local_data

    def solve(self, theta: NDArray[np.float64], pb: Optional[Any] = None) -> NDArray[np.bool_]:
        """Select single item with max utility per agent (respects constraint_mask)."""
        U_i_j = self.build_utilities(theta)
        if self.has_constraint_mask:
            U_i_j = np.where(self.local_data["constraint_mask"], U_i_j, -np.inf)
        j_star = np.argmax(U_i_j, axis=1)
        max_vals = U_i_j[np.arange(self.num_local_agents), j_star]
        optimal_bundles = (max_vals > 0)[:, None] & (np.arange(self.num_items) == j_star[:, None])
        return optimal_bundles

    def build_utilities(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Build utility matrix U_i_j for all agents and items."""
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
 