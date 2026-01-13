"""
Quadratic Supermodular Base Class
--------------------------------
Base class for quadratic supermodular minimization subproblems.
Handles common initialization and matrix building logic shared by different solvers.
"""
import numpy as np
from typing import Any, Optional, Tuple
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
        info = self.data_manager.get_data_info()
        self.info = info
        self.has_modular_agent = info["has_modular_agent"]
        self.has_modular_item = info["has_modular_item"]
        self.has_quadratic_agent = info["has_quadratic_agent"]
        self.has_quadratic_item = info["has_quadratic_item"]
        self.has_errors = info["has_errors"]
        self.has_constraint_mask = info["has_constraint_mask"]

        offset = 0
        if self.has_modular_agent:
            self.modular_agent = self.local_data["agent_data"]["modular"]
            self.modular_agent_slice = slice(offset, offset + info["num_modular_agent"])
            offset += info["num_modular_agent"]
        if self.has_quadratic_agent:
            self.quadratic_agent = self.local_data["agent_data"]["quadratic"]
            self.quadratic_agent_slice = slice(offset, offset + info["num_quadratic_agent"])
            offset += info["num_quadratic_agent"]
        if self.has_modular_item:
            self.modular_item = self.local_data["item_data"]["modular"]
            self.modular_item_slice = slice(offset, offset + info["num_modular_item"])
            offset += info["num_modular_item"]
        if self.has_quadratic_item:
            self.quadratic_item = self.local_data["item_data"]["quadratic"]
            self.quadratic_item_slice = slice(offset, offset + info["num_quadratic_item"])
            offset += info["num_quadratic_item"]
        if self.has_errors:
            self.errors = self.local_data["errors"]

        if self.has_quadratic_agent: 
            assert np.all(np.diagonal(self.quadratic_agent, axis1=1, axis2=2) == 0), f"Matrix has non-zero diagonal"
            assert np.all(self.quadratic_agent >= 0), f"Matrix has off-diagonal negative values"
        if self.has_quadratic_item:
            assert np.all(np.diagonal(self.quadratic_item, axis1=0, axis2=1) == 0), f"Matrix has non-zero diagonal"
            assert np.all(self.quadratic_item >= 0), f"Matrix has off-diagonal negative values"


    def build_quadratic_matrix(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build linear and quadratic matrices for all agents.
        Args:
            theta: Parameter vector
        Returns:
            Tuple of (linear, quadratic) where:
            - linear: (num_local_agents, num_items) linear/diagonal terms
            - quadratic: (num_local_agents, num_items, num_items) off-diagonal quadratic terms
        """
        linear = np.zeros((self.num_local_agents, self.num_items))
        quadratic = np.zeros((self.num_local_agents, self.num_items, self.num_items))
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            # Build diagonal contributions
            if self.has_modular_agent:
                linear += (self.modular_agent @ theta[self.modular_agent_slice])
            if self.has_quadratic_agent:
                quadratic += (self.quadratic_agent @ theta[self.quadratic_agent_slice])
            if self.has_modular_item:
                linear += (self.modular_item @ theta[self.modular_item_slice])
            if self.has_quadratic_item:
                quadratic += (self.quadratic_item @ theta[self.quadratic_item_slice])
            if self.has_errors:
                linear += self.errors
        
        return linear, quadratic