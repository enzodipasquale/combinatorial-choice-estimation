import numpy as np
import gurobipy as gp
from ...subproblem_base import SerialSubproblemBase
from .quadratic_obj_base import QuadraticObjectiveMixin
from bundlechoice.utils import suppress_output

class QuadraticKnapsackGRBSubproblem(QuadraticObjectiveMixin, SerialSubproblemBase):
    weights = None
    capacity = None 

    def _pre_solve_batched_computations(self, theta):
        self.L_all = self._build_linear_coeff_batch(theta)
        self.Q_all = self._build_quadratic_coeff_batch(theta)
        
    def initialize_single_pb(self, local_id):
        weights = self.data_manager.local_data["item_data"]['weight']
        capacity = self.data_manager.local_data["id_data"]['capacity'][local_id]
        with suppress_output():
            model = gp.Model()
            model.setParam('OutputFlag', 0)
            model.setParam('Threads', 1)
            time_limit = self.subproblem_cfg.GRB_settings.get('TimeLimit')
            if time_limit:
                model.setParam('TimeLimit', time_limit)
            model.setAttr('ModelSense', gp.GRB.MAXIMIZE)
            B = model.addMVar(self.dimensions_cfg.n_items, vtype=gp.GRB.BINARY, name = 'bundle')
            self._pre_initilize_cache.append(B)
            model.addConstr(weights @ B <= capacity)
            model.update()
        return model

    def solve_single_pb(self, local_id, theta, model):
        L = self.L_all[local_id]
        Q = self.Q_all[local_id]
        model.setMObjective(Q, L, 0.0, sense=gp.GRB.MAXIMIZE)
        model.optimize()
        
        try:
            result = np.array(model.x, dtype=bool)
        except Exception as e:
            raise ValueError(f'Failed to solve quadratic knapsack subproblem at local_id={local_id}, exception={e}')

        # for j, var in enumerate(self.local_B_vars[local_id]):
        #     var.Start = float(result[j])
        self._pre_initilize_cache[local_id].Start = result
        
        return result
