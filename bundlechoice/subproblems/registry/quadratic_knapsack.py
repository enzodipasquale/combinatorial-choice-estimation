"""
Quadratic knapsack subproblem solver using Gurobi.

Solves max utility (with quadratic terms) subject to weight constraint.
"""

import numpy as np
import gurobipy as gp
import logging
from typing import Any
from contextlib import nullcontext
from numpy.typing import NDArray
from ..base import SerialSubproblemBase
from bundlechoice.utils import suppress_output, get_logger

logger = get_logger(__name__)
_null_context = nullcontext


# ============================================================================
# Quadratic Knapsack Subproblem Solver
# ============================================================================

class QuadraticKnapsackSubproblem(SerialSubproblemBase):
    """Quadratic knapsack solver: max utility with quadratic terms, weight constraint."""
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def initialize(self, local_id: int) -> Any:
        """Initialize Gurobi model with weight constraint."""
        output_flag = self.config.settings.get("OutputFlag", 0)
        # Only suppress output if OutputFlag is 0
        context = suppress_output() if output_flag == 0 else _null_context()
        # Suppress gurobipy logger when showing Gurobi output to avoid duplicate messages
        gurobi_logger = logging.getLogger("gurobipy")
        old_gurobi_level = gurobi_logger.level if output_flag == 1 else None
        try:
            if output_flag == 1:
                gurobi_logger.setLevel(logging.WARNING)
            with context:
                subproblem = gp.Model()
                subproblem.setParam('OutputFlag', output_flag)
                subproblem.setParam('Threads', 1)
                time_limit = self.config.settings.get("TimeLimit")
                if time_limit is not None:
                    subproblem.setParam("TimeLimit", time_limit)
                subproblem.setAttr('ModelSense', gp.GRB.MAXIMIZE)
                B_j = subproblem.addVars(self.num_items, vtype=gp.GRB.BINARY)
                weights = self.local_data["item_data"]["weights"]
                capacity = self.local_data["agent_data"]["capacity"][local_id]
                subproblem.addConstr(gp.quicksum(weights[j] * B_j[j] for j in range(self.num_items)) <= capacity)
                subproblem.update()        
        finally:
            if output_flag == 1 and old_gurobi_level is not None:
                gurobi_logger.setLevel(old_gurobi_level)
        return subproblem

    def solve(self, local_id: int, theta: NDArray[np.float64], pb: Any) -> NDArray[np.bool_]:
        """Solve quadratic knapsack: set linear + quadratic objective, optimize."""
        output_flag = self.config.settings.get("OutputFlag", 0)
        # Suppress gurobipy logger when showing Gurobi output to avoid duplicate messages
        gurobi_logger = logging.getLogger("gurobipy")
        old_gurobi_level = None
        if output_flag == 1:
            old_gurobi_level = gurobi_logger.level
            gurobi_logger.setLevel(logging.WARNING)
        
        try:
            L_j = self._build_L_j(local_id, theta)
            Q_j_j = self._build_Q_j_j(local_id, theta)
            
            B_j = pb.getVars()
            for j in range(len(B_j)):
                B_j[j].Obj = L_j[j]
            
            quad_expr = gp.QuadExpr()
            for i in range(self.num_items):
                for j in range(self.num_items):
                    if Q_j_j[i, j] != 0:
                        quad_expr.add(B_j[i] * B_j[j], Q_j_j[i, j])
            
            pb.setObjective(gp.quicksum(L_j[j] * B_j[j] for j in range(self.num_items)) + quad_expr)
            pb.optimize()
            
            optimal_bundle = np.array(pb.x, dtype=bool)
            self._check_mip_gap(pb, local_id)
        finally:
            if output_flag == 1 and old_gurobi_level is not None:
                gurobi_logger.setLevel(old_gurobi_level)
        
        return optimal_bundle

    # ============================================================================
    # Helper Methods
    # ============================================================================

    def _build_Q_j_j(self, local_id: int, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Build quadratic matrix Q_j_j from agent/item quadratic features."""
        agent_data = self.local_data.get("agent_data", {})
        item_data = self.local_data.get("item_data", {})
        
        Q_j_j = np.zeros((self.num_items, self.num_items))
        offset = 0
        
        if "modular" in agent_data:
            num_mod_agent = agent_data["modular"].shape[-1]
            offset += num_mod_agent
        
        if "quadratic" in agent_data:
            quadratic_agent = agent_data["quadratic"]
            num_quad_agent = quadratic_agent.shape[-1]
            Q_j_j += (quadratic_agent[local_id] @ theta[offset:offset + num_quad_agent])
            offset += num_quad_agent
        
        if "modular" in item_data:
            num_mod_item = item_data["modular"].shape[-1]
            offset += num_mod_item
        
        if "quadratic" in item_data:
            quadratic_item = item_data["quadratic"]
            num_quad_item = quadratic_item.shape[-1]
            Q_j_j += (quadratic_item @ theta[offset:offset + num_quad_item])
            offset += num_quad_item
        
        return Q_j_j

    def _build_L_j(self, local_id: int, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Build linear coefficients L_j from agent/item modular features."""
        error_j = self.local_data["errors"][local_id]
        agent_data = self.local_data.get("agent_data", {})
        item_data = self.local_data.get("item_data", {})
        
        L_j = error_j.copy()
        offset = 0
        
        if "modular" in agent_data:
            modular_agent = agent_data["modular"]
            num_mod_agent = modular_agent.shape[-1]
            L_j += (modular_agent[local_id] @ theta[offset:offset + num_mod_agent])
            offset += num_mod_agent
        
        if "modular" in item_data:
            modular_item = item_data["modular"]
            num_mod_item = modular_item.shape[-1]
            L_j += (modular_item @ theta[offset:offset + num_mod_item])
            offset += num_mod_item
        
        return L_j

    def _check_mip_gap(self, subproblem: Any, local_id: int) -> None:
        """Check MIP gap and warn if exceeds tolerance."""
        MIPGap_tol = self.config.settings.get("MIPGap_tol")
        if MIPGap_tol is not None:
            if subproblem.MIPGap > float(MIPGap_tol):
                logger.warning(
                    f"Subproblem {local_id} (rank {getattr(self.data_manager, 'rank', '?')}): "
                    f"MIPGap {subproblem.MIPGap:.4g} > tol {MIPGap_tol}, value: {subproblem.objVal}"
                )
