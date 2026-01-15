import numpy as np

class QuadraticObjectiveMixin:

    def _init_quadratic_info(self):
        self._qinfo = self.data_manager.quadratic_data_info
        self._slices = self._qinfo.slices

    def _build_linear_coeff_single(self, local_id, theta):
        L = self.oracles_manager._modular_local_errors[local_id].copy()
        if 'modular_agent' in self._slices:
            L += (self.data_manager.local_data['agent_data']['modular'][local_id] 
                    @ theta[self._slices['modular_agent']])
        if 'modular_item' in self._slices:
            L += (self.data_manager.local_data['item_data']['modular'] 
                    @ theta[self._slices['modular_item']])
        return L

    def _build_quadratic_coeff_single(self, local_id, theta):
        Q = np.zeros((self.dimensions_cfg.num_items, self.dimensions_cfg.num_items))
        if 'quadratic_agent' in self._slices:
            Q += (self.data_manager.local_data['agent_data']['quadratic'][local_id] 
                    @ theta[self._slices['quadratic_agent']])
        if 'quadratic_item' in self._slices:
            Q += (self.data_manager.local_data['item_data']['quadratic'] 
                    @ theta[self._slices['quadratic_item']])
        return Q

    def _build_linear_coeff_batch(self, theta):
        L = self.oracles_manager._modular_local_errors.copy()
        if 'modular_agent' in self._slices:
            L += (self.data_manager.local_data['agent_data']['modular'] 
                    @ theta[self._slices['modular_agent']])
        if 'modular_item' in self._slices:
            L += (self.data_manager.local_data['item_data']['modular'] 
                    @ theta[self._slices['modular_item']])
        return L

    def _build_quadratic_coeff_batch(self, theta):
        n = self.dimensions_cfg.num_items
        Q = np.zeros((self.data_manager.num_local_agent, n, n))
        if 'quadratic_agent' in self._slices:
            Q += (self.data_manager.local_data['agent_data']['quadratic'] 
                    @ theta[self._slices['quadratic_agent']])
        if 'quadratic_item' in self._slices:
            Q += (self.data_manager.local_data['item_data']['quadratic'] 
                    @ theta[self._slices['quadratic_item']])
        return Q

    def build_linear_and_quadratic_coef(self, theta):
        return self._build_linear_coeff_batch(theta), self._build_quadratic_coeff_batch(theta)
