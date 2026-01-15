import numpy as np

class QuadraticObjectiveMixin:

    def _init_quadratic_info(self):
        self._qinfo = self.data_manager.quadratic_features_info
        self._slices = self._qinfo.slices

    def _build_linear_coeff_single(self, local_id, theta):
        ad = self.data_manager.local_data['agent_data']
        id = self.data_manager.local_data['item_data']
        L = self.oracles_manager._modular_local_errors[local_id].copy()
        if 'modular_agent' in self._slices:
            L += ad['modular'][local_id] @ theta[self._slices['modular_agent']]
        if 'modular_item' in self._slices:
            L += id['modular'] @ theta[self._slices['modular_item']]
        return L

    def _build_quadratic_coeff_single(self, local_id, theta):
        ad = self.data_manager.local_data['agent_data']
        id = self.data_manager.local_data['item_data']
        Q = np.zeros((self.dimensions_cfg.num_items, self.dimensions_cfg.num_items))
        if 'quadratic_agent' in self._slices:
            Q += ad['quadratic'][local_id] @ theta[self._slices['quadratic_agent']]
        if 'quadratic_item' in self._slices:
            Q += id['quadratic'] @ theta[self._slices['quadratic_item']]
        return Q

    def _build_linear_coeff_batch(self, theta):
        ad = self.data_manager.local_data['agent_data']
        id = self.data_manager.local_data['item_data']
        L = self.oracles_manager._modular_local_errors.copy()
        if 'modular_agent' in self._slices:
            L += ad['modular'] @ theta[self._slices['modular_agent']]
        if 'modular_item' in self._slices:
            L += id['modular'] @ theta[self._slices['modular_item']]
        return L

    def _build_quadratic_coeff_batch(self, theta):
        ad = self.data_manager.local_data['agent_data']
        id = self.data_manager.local_data['item_data']
        n_agents = self.data_manager.num_local_agent
        n_items = self.dimensions_cfg.num_items
        Q = np.zeros((n_agents, n_items, n_items))
        if 'quadratic_agent' in self._slices:
            Q += ad['quadratic'] @ theta[self._slices['quadratic_agent']]
        if 'quadratic_item' in self._slices:
            Q += id['quadratic'] @ theta[self._slices['quadratic_item']]
        return Q

    def build_linear_and_quadratic_coef(self, theta):
        return self._build_linear_coeff_batch(theta), self._build_quadratic_coeff_batch(theta)
