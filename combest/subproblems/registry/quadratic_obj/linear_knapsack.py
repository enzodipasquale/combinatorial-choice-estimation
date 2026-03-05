import numpy as np
import gurobipy as gp
from ...solver_base import SubproblemSolver, GurobiMixin
from .quadratic_obj_base import QuadraticObjectiveMixin

class LinearKnapsackGRBSolver(GurobiMixin, QuadraticObjectiveMixin, SubproblemSolver):

    def initialize(self):
        weights = self.data_manager.local_data["item_data"]['weight']
        capacities = self.data_manager.local_data["id_data"]['capacity']
        self.local_problems = []
        for local_id in range(self.comm_manager.num_local_agent):
            model = self._create_gurobi_model()
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
