import numpy as np
import gurobipy as gp
from ...subproblem_base import SubproblemSolver
from .quadratic_obj_base import QuadraticObjectiveMixin
from bundlechoice.utils import suppress_output

class LinearKnapsackGRBSolver(QuadraticObjectiveMixin, SubproblemSolver):

    def initialize(self):
        weights = self.data_manager.local_data["item_data"]['weight']
        capacities = self.data_manager.local_data["id_data"]['capacity']
        self.local_problems = []
        for local_id in range(self.data_manager.num_local_agent):
            with suppress_output():
                model = gp.Model()
                model.setParam('OutputFlag', 0)
                model.setParam('Threads', 1)
                time_limit = self.subproblem_cfg.GRB_Params.get('TimeLimit')
                if time_limit:
                    model.setParam('TimeLimit', time_limit)
                model.setAttr('ModelSense', gp.GRB.MAXIMIZE)
                B = model.addMVar(self.dimensions_cfg.n_items, vtype=gp.GRB.BINARY)
                model.addConstr(weights @ B <= capacities[local_id])
                model.update()
            self.local_problems.append(model)

    def solve(self, theta):
        L_all = self._build_linear_coeff_batch(theta)
        n_items = self.dimensions_cfg.n_items
        results = np.zeros((len(self.local_problems), n_items), dtype=bool)
        for i, model in enumerate(self.local_problems):
            B = model.getVars()
            for j in range(n_items):
                B[j].Obj = L_all[i, j]
            model.optimize()
            results[i] = np.array([v.x for v in B], dtype=bool)
        return results

    def update_solver_settings(self, settings_dict):
        if hasattr(self, 'local_problems'):
            for model in self.local_problems:
                for param, value in settings_dict.items():
                    model.setParam(param, value)
