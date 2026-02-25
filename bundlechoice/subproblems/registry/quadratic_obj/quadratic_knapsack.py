import numpy as np
import gurobipy as gp
from ...solver_base import SubproblemSolver
from .quadratic_obj_base import QuadraticObjectiveMixin
from bundlechoice.utils import suppress_output

class QuadraticKnapsackGRBSolver(QuadraticObjectiveMixin, SubproblemSolver):

    def initialize(self):
        weights = self.data_manager.local_data["item_data"]['weight']
        capacities = self.data_manager.local_data["id_data"]['capacity']
        self.local_problems = []
        for local_id in range(self.comm_manager.num_local_agent):
            with suppress_output():
                model = gp.Model()
                model.setParam('OutputFlag', 0)
                model.setParam('Threads', 1)
                for k, v in self.subproblem_cfg.GRB_Params.items():
                    if v is not None:
                        model.setParam(k, v)
                model.setAttr('ModelSense', gp.GRB.MAXIMIZE)
                B = model.addMVar(self.dimensions_cfg.n_items, vtype=gp.GRB.BINARY, name='bundle')
                model.addConstr(weights @ B <= capacities[local_id])
                model.update()
            self.local_problems.append(model)

    def solve(self, theta):
        L_all = self._build_linear_coeff_batch(theta)
        Q_all = self._build_quadratic_coeff_batch(theta)
        results = np.zeros((len(self.local_problems), self.dimensions_cfg.n_items), dtype=bool)
        for i, model in enumerate(self.local_problems):
            model.setMObjective(Q_all[i], L_all[i], 0.0, sense=gp.GRB.MAXIMIZE)
            model.optimize()
            try:
                results[i] = np.array(model.x, dtype=bool)
            except Exception as e:
                raise ValueError(f'Failed to solve quadratic knapsack subproblem at local_id={i}, exception={e}')
        return results

    def update_solver_settings(self, settings_dict):
        if hasattr(self, 'local_problems'):
            for model in self.local_problems:
                for param, value in settings_dict.items():
                    model.setParam(param, value)
