import numpy as np
from functools import lru_cache

class QuadraticObjectiveMixin:
    @property
    def _qinfo(self):
        return self.data_manager.get_quadratic_data_info()
    
    @property
    def _slices(self):
        return self._qinfo.slices

    def _build_linear_coeff_batch(self, theta):
        L = self.features_manager.local_modular_errors.copy()
        if 'modular_agent' in self._slices:
            L += (self.data_manager.local_data["id_data"]['modular'] 
                    @ theta[self._slices['modular_agent']])
        if 'modular_item' in self._slices:
            L += (self.data_manager.local_data["item_data"]['modular'] 
                    @ theta[self._slices['modular_item']])
        return L

    def _build_quadratic_coeff_batch(self, theta):
        n = self.dimensions_cfg.n_items
        Q = np.zeros((self.comm_manager.num_local_agent, n, n))
        if 'quadratic_agent' in self._slices:
            Q += (self.data_manager.local_data["id_data"]['quadratic'] 
                    @ theta[self._slices['quadratic_agent']])
        if 'quadratic_item' in self._slices:
            Q += (self.data_manager.local_data["item_data"]['quadratic'] 
                    @ theta[self._slices['quadratic_item']])
        return Q

    def build_linear_and_quadratic_coef(self, theta):
        return self._build_linear_coeff_batch(theta), self._build_quadratic_coeff_batch(theta)
