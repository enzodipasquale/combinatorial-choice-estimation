import numpy as np
from ...subproblem_base import BatchSubproblemBase
from .quadratic_obj_base import QuadraticObjectiveMixin

class PlainSingleItemSubproblem(QuadraticObjectiveMixin, BatchSubproblemBase):

    def initialize(self):
        self._init_quadratic_info()
        self.has_constraint_mask = 'constraint_mask' in self.data_manager.local_data['agent_data']

    def solve(self, theta):
        U = self._build_linear_coeff_batch(theta)
        if self.has_constraint_mask:
            mask = self.data_manager.local_data['agent_data']['constraint_mask']
            U = np.where(mask, U, -np.inf)
        j_star = np.argmax(U, axis=1)
        max_vals = U[np.arange(self.data_manager.num_local_agent), j_star]
        n = self.dimensions_cfg.num_items
        return (max_vals > 0)[:, None] & (np.arange(n) == j_star[:, None])
