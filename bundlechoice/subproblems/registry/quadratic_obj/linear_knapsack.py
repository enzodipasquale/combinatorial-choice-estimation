import numpy as np
import gurobipy as gp
from ...subproblem_base import SerialSubproblemBase
from .quadratic_obj_base import QuadraticObjectiveMixin
from bundlechoice.utils import suppress_output

class LinearKnapsackGRBSubproblem(QuadraticObjectiveMixin, SerialSubproblemBase):

    def initialize_single_pb(self, local_id):
        weights = self.data_manager.local_data["item_data"]['weights']
        capacity = self.data_manager.local_data["id_data"]['capacity'][local_id]
        with suppress_output():
            model = gp.Model()
            model.setParam('OutputFlag', 0)
            model.setParam('Threads', 1)
            time_limit = self.subproblem_cfg.settings.get('TimeLimit')
            if time_limit:
                model.setParam('TimeLimit', time_limit)
            model.setAttr('ModelSense', gp.GRB.MAXIMIZE)
            B = model.addMVar(self.dimensions_cfg.n_items, vtype=gp.GRB.BINARY)
            model.addConstr(weights @ B <= capacity)
            model.update()
        return model

    def solve_single_pb(self, local_id, theta, pb):
        B = pb.getVars()
        L = self._build_linear_coeff_single(local_id, theta)
        for j in range(self.dimensions_cfg.n_items):
            B[j].Obj = L[j]
        pb.optimize()
        return np.array([v.x for v in B], dtype=bool)
