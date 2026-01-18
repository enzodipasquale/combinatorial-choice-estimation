import numpy as np
from ...subproblem_base import BatchSubproblemBase
from .quadratic_obj_base import QuadraticObjectiveMixin

class PlainSingleItemSubproblem(QuadraticObjectiveMixin, BatchSubproblemBase):

    def solve(self, theta):
        U = self._build_linear_coeff_batch(theta)
        if self._qinfo.constraint_mask is not None:
            U = np.where(self._qinfo.constraint_mask, U, -np.inf)
        j_star = np.argmax(U, axis=1)
        max_vals = U[np.arange(self.data_manager.num_local_agent), j_star]
        return ((max_vals > 0)[:, None] & 
                (np.arange(self.dimensions_cfg.n_items) == j_star[:, None]))
