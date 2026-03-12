import numpy as np
from ...solver_base import SubproblemSolver
from .quadratic_obj_base import QuadraticObjectiveMixin

class UnitDemandSolver(QuadraticObjectiveMixin, SubproblemSolver):

    def _build_linear_coeff_batch(self, theta):
        L = self.features_manager.local_modular_errors.copy()
        fe_idx = self.data_manager.local_data.id_data.get('fe_index')
        if fe_idx is not None:
            L += np.where(fe_idx >= 0, theta[np.maximum(fe_idx, 0)], 0.0)
        else:
            if 'modular_agent' in self._slices:
                L += (self.data_manager.local_data.id_data['modular']
                        @ theta[self._slices['modular_agent']])
            if 'modular_item' in self._slices:
                L += (self.data_manager.local_data.item_data['modular']
                        @ theta[self._slices['modular_item']])
        return L

    def solve(self, theta):
        U = self._build_linear_coeff_batch(theta)
        mask = self.data_manager.id_data["constraint_mask"]
        if mask is not None:
            U = np.where(mask, U, -np.inf)
        j_star = np.argmax(U, axis=1)
        max_vals = U[np.arange(self.comm_manager.num_local_agent), j_star]
        return ((max_vals > 0)[:, None] &
                (np.arange(self.dimensions_cfg.n_items) == j_star[:, None]))
