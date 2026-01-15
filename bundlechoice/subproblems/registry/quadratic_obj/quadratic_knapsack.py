import numpy as np
import gurobipy as gp
from ...subproblem_base import SerialSubproblemBase
from .quadratic_obj_base import QuadraticObjectiveMixin
from bundlechoice.utils import suppress_output

class QuadraticKnapsackSubproblem(QuadraticObjectiveMixin, SerialSubproblemBase):

    weights = None
    capacity = None 

    def initialize_single_pb(self, local_id):
        weights = self.data_manager.local_data['item_data']['weights']
        capacity = self.data_manager.local_data['agent_data']['capacity'][local_id]
        with suppress_output():
            model = gp.Model()
            model.setParam('OutputFlag', 0)
            model.setParam('Threads', 1)
            time_limit = self.subproblem_cfg.settings.get('TimeLimit')
            if time_limit:
                model.setParam('TimeLimit', time_limit)
            model.setAttr('ModelSense', gp.GRB.MAXIMIZE)
            B = model.addMVar(self.dimensions_cfg.num_items, vtype=gp.GRB.BINARY)
            model.addConstr(weights @ B <= capacity)
            model.update()
        return model

    def solve_single_pb(self, local_id, theta, pb):
        L = self._build_linear_coeff_single(local_id, theta)
        Q = self._build_quadratic_coeff_single(local_id, theta)
        pb.setMObjective(Q, L, 0.0, sense=gp.GRB.MAXIMIZE)
        pb.optimize()
        try:
            return np.array(pb.x, dtype=bool)
        except:
            raise ValueError('Failed to solve quadratic knapsack subproblem')
