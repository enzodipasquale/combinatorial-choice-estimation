import numpy as np
import gurobipy as gp
from ..subproblem_base import SerialSubproblemBase
from bundlechoice.utils import suppress_output, get_logger

logger = get_logger(__name__)

class QuadraticKnapsackSubproblem(SerialSubproblemBase):

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
        linear_coeff = self._build_linear_coeff(local_id, theta)
        quadratic_coeff = self._build_quadratic_coeff(local_id, theta)
        pb.setMObjective(quadratic_coeff, linear_coeff, 0.0, sense=gp.GRB.MAXIMIZE)
        pb.optimize()
        try:
            return np.array(pb.x, dtype=bool)
        except:
            return np.zeros(self.dimensions_cfg.num_items, dtype=bool)

    def _build_linear_coeff(self, local_id, theta):
        linear_coeff = self.oracles_manager._modular_local_errors[local_id].copy()
        ad = self.data_manager.local_data['agent_data']
        id = self.data_manager.local_data['item_data']
        offset = 0
        if 'modular' in ad:
            dim = ad['modular'].shape[-1]
            linear_coeff += ad['modular'][local_id] @ theta[offset:offset + dim]
            offset += dim
        if 'modular' in id:
            dim = id['modular'].shape[-1]
            linear_coeff += id['modular'] @ theta[offset:offset + dim]
            offset += dim
        # skip quadratic slices (handled in _build_quadratic_coeff)
        return linear_coeff

    def _build_quadratic_coeff(self, local_id, theta):
        J = self.dimensions_cfg.num_items
        Q = np.zeros((J, J))
        ad = self.data_manager.local_data['agent_data']
        id = self.data_manager.local_data['item_data']
        offset = 0
        if 'modular' in ad:
            offset += ad['modular'].shape[-1]
        if 'modular' in id:
            offset += id['modular'].shape[-1]
        if 'quadratic' in ad:
            dim = ad['quadratic'].shape[-1]
            Q += ad['quadratic'][local_id] @ theta[offset:offset + dim]
            offset += dim
        if 'quadratic' in id:
            dim = id['quadratic'].shape[-1]
            Q += id['quadratic'] @ theta[offset:offset + dim]
        return Q

    def _get_item_errors(self, local_id):
    
        FE = np.eye(self.dimensions_cfg.num_items, dtype=bool)
        return np.array([self.oracles_manager.error_oracle(FE[j:j+1], np.array([local_id]))[0] for j in range(self.dimensions_cfg.num_items)])
