"""
Quadratic Supermodular Base Class
--------------------------------
Base class for quadratic supermodular minimization subproblems.
Handles common initialization and matrix building logic shared by different solvers.
"""
import numpy as np
from typing import Any, Optional
from ...base import BatchSubproblemBase

def assert_zero_diagonal(arr: np.ndarray, axis1: int, axis2: int, context: str = ""):
    """Utility to assert that the diagonal along given axes is zero for all slices."""
    diags = np.diagonal(arr, axis1=axis1, axis2=axis2)
    if not np.all(diags == 0):
        raise ValueError(f"Nonzero diagonal detected in {context} (shape {arr.shape})")

class QuadraticSupermodular(BatchSubproblemBase):
    """
    Base class for quadratic supermodular minimization subproblems.
    Handles common initialization and matrix building logic.
    """
    def initialize(self) -> None:
        """
        Prepare all agent/item data and precompute slices for efficient solving.
        Handles missing modular/quadratic agent/item data gracefully.
        """
        agent_data = self.local_data.get("agent_data", {})
        item_data = self.local_data.get("item_data", {})

        # Modular and quadratic features (None if missing)
        if agent_data is not None:
            self.modular_agent = agent_data.get("modular")
            self.quadratic_agent = agent_data.get("quadratic")
        else:
            self.modular_agent = None
            self.quadratic_agent = None
        if item_data is not None:
            self.modular_item = item_data.get("modular")
            self.quadratic_item = item_data.get("quadratic")
        else:
            self.modular_item = None
            self.quadratic_item = None

        # Assert that the diagonal of quadratic terms are all zero (if present)
        if self.quadratic_agent is not None:
            for k in range(self.quadratic_agent.shape[-1]):
                assert_zero_diagonal(self.quadratic_agent[:, :, :, k], axis1=1, axis2=2, context="agent quadratic")
                assert np.all(self.quadratic_agent[:, :, :, k] >= 0), f"Matrix {k} has negative values"
        if self.quadratic_item is not None:
            for k in range(self.quadratic_item.shape[-1]):
                assert_zero_diagonal(self.quadratic_item[:, :, k], axis1=0, axis2=1, context="item quadratic")
                assert np.all(self.quadratic_item[:, :, k] >= 0)

        self.errors = self.local_data.get("errors")
        self.constraint_mask = self.local_data.get("constraint_mask") or np.ones((self.num_local_agents, self.num_items), dtype=bool)

        # Feature counts
        self.num_mod_agent = self.modular_agent.shape[-1] if self.modular_agent is not None else 0
        self.num_quad_agent = self.quadratic_agent.shape[-1] if self.quadratic_agent is not None else 0
        self.num_mod_item = self.modular_item.shape[-1] if self.modular_item is not None else 0
        self.num_quad_item = self.quadratic_item.shape[-1] if self.quadratic_item is not None else 0

        # Lambda slices
        offset = 0
        self.lambda_mod_agent_slice = slice(offset, offset + self.num_mod_agent); offset += self.num_mod_agent
        self.lambda_quad_agent_slice = slice(offset, offset + self.num_quad_agent); offset += self.num_quad_agent
        self.lambda_mod_item_slice = slice(offset, offset + self.num_mod_item); offset += self.num_mod_item
        self.lambda_quad_item_slice = slice(offset, offset + self.num_quad_item); offset += self.num_quad_item

        # Precompute zeros matrix template and diagonal indices using self.num_items
        self.diag_indices = np.diag_indices(self.num_items)  # For advanced diagonal indexing

    def build_quadratic_matrix(self, lambda_k: np.ndarray) -> np.ndarray:
        """
        Build the quadratic matrix for all agents.
        Args:
            lambda_k: Parameter vector
        Returns:
            P_i_j_j: Upper Triangular Quadratic matrices for all agents (num_local_agents, num_items, num_items)
        """
        P_i_j_j = np.zeros((self.num_local_agents, self.num_items, self.num_items))

        # Add quadratic agent/item terms if present
        if self.quadratic_agent is not None:
            P_i_j_j += (self.quadratic_agent @ lambda_k[self.lambda_quad_agent_slice])
        if self.quadratic_item is not None:
            P_i_j_j += (self.quadratic_item @ lambda_k[self.lambda_quad_item_slice])
        
        # Symmetrize each agent's matrix (i.e., ensure P_i_j_j is symmetric in last two dims)
        P_i_j_j = P_i_j_j + np.transpose(P_i_j_j, (0, 2, 1))
        
        # Add modular agent/item terms to diagonal
        if self.modular_agent is not None:
            P_i_j_j[:, self.diag_indices[0], self.diag_indices[1]] += (self.modular_agent @ lambda_k[self.lambda_mod_agent_slice])
        if self.modular_item is not None:
            P_i_j_j[:, self.diag_indices[0], self.diag_indices[1]] += (self.modular_item @ lambda_k[self.lambda_mod_item_slice])

        # Add errors to diagonal if present
        if self.errors is not None:
            P_i_j_j[:, self.diag_indices[0], self.diag_indices[1]] += self.errors
        
        P_i_j_j *= np.triu(np.ones((self.num_items, self.num_items)))[None, :, :]

        return P_i_j_j 