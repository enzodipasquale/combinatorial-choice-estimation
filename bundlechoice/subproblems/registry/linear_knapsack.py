import numpy as np
import gurobipy as gp
from ..subproblem_base import SerialSubproblemBase
from bundlechoice.utils import suppress_output, get_logger

logger = get_logger(__name__)

class LinearKnapsackSubproblem(SerialSubproblemBase):

    def initialize_single_pb(self, local_id):
        with suppress_output():
            model = gp.Model()
            model.setParam('OutputFlag', 0)
            model.setParam('Threads', 1)
            time_limit = self.subproblem_cfg.settings.get('TimeLimit')
            if time_limit:
                model.setParam('TimeLimit', time_limit)
            model.setAttr('ModelSense', gp.GRB.MAXIMIZE)
            B = model.addVars(self.dimensions_cfg.num_items, vtype=gp.GRB.BINARY)
            weights = self.data_manager.local_data['item_data']['weights']
            capacity = self.data_manager.local_data['agent_data']['capacity'][local_id]
            model.addConstr(gp.quicksum(weights[j] * B[j] for j in range(self.dimensions_cfg.num_items)) <= capacity)
            model.update()
        return model

    def solve_single_pb(self, local_id, theta, pb):
        L_j = self._build_L_j(local_id, theta)
        B = pb.getVars()
        for j, v in enumerate(B):
            v.Obj = L_j[j]
        pb.optimize()
        return np.array(pb.x, dtype=bool)

    def _build_L_j(self, local_id, theta):
        n = self.dimensions_cfg.num_items
        error_j = self._get_item_errors(local_id)
        L_j = error_j.copy()
        ad = self.data_manager.local_data['agent_data']
        id = self.data_manager.local_data['item_data']
        offset = 0
        if 'modular' in ad:
            dim = ad['modular'].shape[-1]
            L_j += ad['modular'][local_id] @ theta[offset:offset + dim]
            offset += dim
        if 'modular' in id:
            dim = id['modular'].shape[-1]
            L_j += id['modular'] @ theta[offset:offset + dim]
        return L_j

    def _get_item_errors(self, local_id):
        n = self.dimensions_cfg.num_items
        I = np.eye(n, dtype=bool)
        return np.array([self.oracles_manager.error_oracle(I[j:j+1], np.array([local_id]))[0] for j in range(n)])
