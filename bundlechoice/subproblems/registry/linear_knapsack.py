import numpy as np
import gurobipy as gp
from ..base import BaseSerialSubproblem
import logging
from typing import Any
from bundlechoice.utils import suppress_output

logger = logging.getLogger(__name__)

class LinearKnapsackSubproblem(BaseSerialSubproblem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def initialize(self, local_id):
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
            capacity = self.local_data["agent_data"]["capacity"][local_id]
            subproblem.addConstr(gp.quicksum(weights[j] * B_j[j] for j in range(self.num_items)) <= capacity)
            subproblem.update()
        return subproblem

    def solve(self, local_id, theta, pb: Any):
        L_j = self._build_L_j(local_id, theta)
        B_j = pb.getVars()
        for j in range(len(B_j)):
            B_j[j].Obj = L_j[j]
        pb.optimize()
        optimal_bundle = np.array(pb.x, dtype=bool)
        self._check_mip_gap(pb, local_id)
        return optimal_bundle

    def _build_L_j(self, local_id, theta):
        error_j = self.local_data["errors"][local_id]
        # Agent modular
        agent_modular = self.local_data["agent_data"].get("modular", None)
        agent_modular_dim = agent_modular.shape[-1] if agent_modular is not None else 0
        # Item modular
        item_modular_k = self.local_data["item_data"].get("modular", None)
        item_modular_dim = item_modular_k.shape[-1] if item_modular_k is not None else 0
        # Slicing theta
        lambda_agent = theta[:agent_modular_dim] if agent_modular_dim > 0 else None
        lambda_item = theta[agent_modular_dim:agent_modular_dim+item_modular_dim] if item_modular_dim > 0 else None
        # Build L_j
        L_j = error_j.copy()
        if agent_modular_dim > 0:
            L_j += agent_modular[local_id] @ lambda_agent
        if item_modular_dim > 0:
            L_j += item_modular_k @ lambda_item
        return L_j

    def _check_mip_gap(self, subproblem, local_id):
        MIPGap_tol = self.config.settings.get("MIPGap_tol")
        if MIPGap_tol is not None:
            if subproblem.MIPGap > float(MIPGap_tol):
                logger.warning(
                    f"Subproblem {local_id} (rank {getattr(self.data_manager, 'rank', '?')}): "
                    f"MIPGap {subproblem.MIPGap:.4g} > tol {MIPGap_tol}, value: {subproblem.objVal}"
                ) 