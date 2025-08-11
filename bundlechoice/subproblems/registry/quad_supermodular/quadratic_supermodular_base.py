"""
Quadratic Supermodular Base Class
--------------------------------
Base class for quadratic supermodular minimization subproblems.
Handles common initialization and matrix building logic shared by different solvers.
"""
import numpy as np
from typing import Any, Optional
from ...base import BatchSubproblemBase

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
        agent_data = self.local_data.get("agent_data") or {}
        item_data = self.local_data.get("item_data") or {}

        self.has_modular_agent = "modular" in agent_data
        self.has_modular_item = "modular" in item_data
        self.has_quadratic_agent = "quadratic" in agent_data
        self.has_quadratic_item = "quadratic" in item_data
        self.has_errors = "errors" in self.local_data
        self.has_constraint_mask = "constraint_mask" in self.local_data

        if self.has_quadratic_agent:
            quadratic_agent = self.local_data["agent_data"]["quadratic"]    
            assert np.all(np.diagonal(quadratic_agent, axis1=1, axis2=2) == 0), f"Matrix has non-zero diagonal"
            assert np.all(quadratic_agent >= 0), f"Matrix has off-diagonal negative values"
        if self.has_quadratic_item:
            quadratic_item = self.local_data["item_data"]["quadratic"]
            assert np.all(np.diagonal(quadratic_item, axis1=0, axis2=1) == 0), f"Matrix has non-zero diagonal"
            assert np.all(quadratic_item >= 0), f"Matrix has off-diagonal negative values"

    def build_quadratic_matrix(self, theta: np.ndarray) -> np.ndarray:
        """
        Build the quadratic matrix for all agents.
        Args:
            theta: Parameter vector
        Returns:
            agents_matrices: Upper Triangular Quadratic matrices for all agents (num_local_agents, num_items, num_items)
        """
        agents_matrices = np.zeros((self.num_local_agents, self.num_items, self.num_items))
        offset = 0
        
        # Build diagonal contributions
        diagonal_contributions = np.zeros((self.num_local_agents, self.num_items))
        
        if self.has_modular_agent:
            modular_agent = self.local_data["agent_data"]["modular"]
            num_mod_agent = modular_agent.shape[-1]
            diagonal_contributions += (modular_agent @ theta[offset:offset + num_mod_agent])
            offset += num_mod_agent

        if self.has_quadratic_agent:
            quadratic_agent = self.local_data["agent_data"]["quadratic"]
            num_quad_agent = quadratic_agent.shape[-1]
            agents_matrices += (quadratic_agent @ theta[offset:offset + num_quad_agent])
            offset += num_quad_agent

        if self.has_modular_item:
            modular_item = self.local_data["item_data"]["modular"]
            num_mod_item = modular_item.shape[-1]
            diagonal_contributions += (modular_item @ theta[offset:offset + num_mod_item])
            offset += num_mod_item

        if self.has_quadratic_item:
            quadratic_item = self.local_data["item_data"]["quadratic"]
            num_quad_item = quadratic_item.shape[-1]
            agents_matrices += (quadratic_item @ theta[offset:offset + num_quad_item])
            offset += num_quad_item

        if self.has_errors:
            diagonal_contributions += self.local_data["errors"]
        
        # Apply diagonal contributions
        agents_matrices[:, np.arange(self.num_items), np.arange(self.num_items)] += diagonal_contributions
        agents_matrices = np.triu(agents_matrices, k = 0) + np.tril(agents_matrices, k = -1).transpose(0, 2, 1)
        
        # assert upper triangular
        assert np.all(np.triu(agents_matrices) == agents_matrices), f"agents_matrices is not upper triangular"
        # assert lower triangular
        # if any of the entry of P is nan give error
        if np.any(np.isnan(agents_matrices)):
            raise ValueError(f"agents_matrices contains nan, theta: {theta}")
        return agents_matrices 