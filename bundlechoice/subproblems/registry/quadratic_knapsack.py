import logging
from typing import Any

import numpy as np
import gurobipy as gp

from bundlechoice.utils import suppress_output
from ..base import SerialSubproblemBase

logger = logging.getLogger(__name__)

class QuadraticKnapsackSubproblem(SerialSubproblemBase):
    """
    Quadratic knapsack subproblem that handles both modular and quadratic features.
    Maintains the knapsack constraint while supporting quadratic objective terms.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize(self, agent_id):
        with suppress_output():
            subproblem = gp.Model()
            subproblem.setParam('OutputFlag', 0)
            subproblem.setParam('Threads', 1)
            time_limit = self.config.settings.get("TimeLimit")
            if time_limit is not None:
                subproblem.setParam("TimeLimit", time_limit)
            subproblem.setAttr('ModelSense', gp.GRB.MAXIMIZE)
            B_j = subproblem.addVars(self.num_items, vtype=gp.GRB.BINARY)
            weights = self.local_data["item_data"]["weights"]
            capacity = self.local_data["agent_data"]["capacity"][agent_id]
            subproblem.addConstr(gp.quicksum(weights[j] * B_j[j] for j in range(self.num_items)) <= capacity)
            subproblem.update()        
        return subproblem

    def solve(self, agent_id, theta, pb: Any):
        L_j = self._build_L_j(agent_id, theta)
        Q_j_j = self._build_Q_j_j(agent_id, theta)
        
        B_j = pb.getVars()
        
        # Set linear objective
        for j in range(len(B_j)):
            B_j[j].Obj = L_j[j]
        
        # Add quadratic terms
        quad_expr = gp.QuadExpr()
        for i in range(self.num_items):
            for j in range(self.num_items):
                if Q_j_j[i, j] != 0:
                    quad_expr.add(B_j[i] * B_j[j], Q_j_j[i, j])
        
        # Set quadratic objective
        pb.setObjective(gp.quicksum(L_j[j] * B_j[j] for j in range(self.num_items)) + quad_expr)
        pb.optimize()
        
        optimal_bundle = np.array(pb.x, dtype=bool)
        self._check_mip_gap(pb, agent_id)
        return optimal_bundle

    def _build_Q_j_j(self, agent_id, theta):
        """
        Build the quadratic matrix Q_j_j for the given agent.
        """
        agent_data = self.local_data.get("agent_data", {})
        item_data = self.local_data.get("item_data", {})
        
        Q_j_j = np.zeros((self.num_items, self.num_items))
        offset = 0
        
        # Handle agent modular features
        if "modular" in agent_data:
            num_mod_agent = agent_data["modular"].shape[-1]
            offset += num_mod_agent
        
        # Handle agent quadratic features
        if "quadratic" in agent_data:
            quadratic_agent = agent_data["quadratic"]
            num_quad_agent = quadratic_agent.shape[-1]
            Q_j_j += (quadratic_agent[agent_id] @ theta[offset:offset + num_quad_agent])
            offset += num_quad_agent
        
        # Handle item modular features
        if "modular" in item_data:
            num_mod_item = item_data["modular"].shape[-1]
            offset += num_mod_item
        
        # Handle item quadratic features
        if "quadratic" in item_data:
            quadratic_item = item_data["quadratic"]
            num_quad_item = quadratic_item.shape[-1]
            Q_j_j += (quadratic_item @ theta[offset:offset + num_quad_item])
            offset += num_quad_item
        
        return Q_j_j

    def _build_L_j(self, agent_id, theta):
        """
        Build the linear coefficients L_j for the given agent.
        """
        error_j = self.local_data["errors"][agent_id]
        agent_data = self.local_data.get("agent_data", {})
        item_data = self.local_data.get("item_data", {})
        
        L_j = error_j.copy()
        offset = 0
        
        # Handle agent modular features
        if "modular" in agent_data:
            modular_agent = agent_data["modular"]
            num_mod_agent = modular_agent.shape[-1]
            L_j += (modular_agent[agent_id] @ theta[offset:offset + num_mod_agent])
            offset += num_mod_agent
        
        # Handle item modular features
        if "modular" in item_data:
            modular_item = item_data["modular"]
            num_mod_item = modular_item.shape[-1]
            L_j += (modular_item @ theta[offset:offset + num_mod_item])
            offset += num_mod_item
        
        return L_j

    def _check_mip_gap(self, subproblem, agent_id):
        """
        Check MIP gap and log warnings if it exceeds tolerance.
        """
        MIPGap_tol = self.config.settings.get("MIPGap_tol")
        if MIPGap_tol is not None:
            if subproblem.MIPGap > float(MIPGap_tol):
                logger.warning(
                    f"Subproblem {agent_id} (rank {getattr(self.data_manager, 'rank', '?')}): "
                    f"MIPGap {subproblem.MIPGap:.4g} > tol {MIPGap_tol}, value: {subproblem.objVal}"
                )
