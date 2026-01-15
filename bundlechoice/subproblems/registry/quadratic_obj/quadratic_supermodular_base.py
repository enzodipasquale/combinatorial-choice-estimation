import numpy as np
from ...subproblem_base import BatchSubproblemBase

class QuadraticSupermodular(BatchSubproblemBase):

    def initialize(self):
        ma, qa, mi, qi = self.data_manager.quadratic_features_flags()
        self.has_modular_agent = ma
        self.has_quadratic_agent = qa
        self.has_modular_item = mi
        self.has_quadratic_item = qi
        ad = self.data_manager.local_data['agent_data']
        id = self.data_manager.local_data['item_data']
        self.has_constraint_mask = 'constraint_mask' in ad
        self._compute_slices(ad, id)
        self._validate_quadratic(ad, id)

    def _compute_slices(self, ad, id):
        offset = 0
        if self.has_modular_agent:
            dim = ad['modular'].shape[-1]
            self.modular_agent_slice = slice(offset, offset + dim)
            offset += dim
        if self.has_modular_item:
            dim = id['modular'].shape[-1]
            self.modular_item_slice = slice(offset, offset + dim)
            offset += dim
        if self.has_quadratic_agent:
            dim = ad['quadratic'].shape[-1]
            self.quadratic_agent_slice = slice(offset, offset + dim)
            offset += dim
        if self.has_quadratic_item:
            dim = id['quadratic'].shape[-1]
            self.quadratic_item_slice = slice(offset, offset + dim)

    def _validate_quadratic(self, ad, id):
        if self.has_quadratic_agent:
            Q = ad['quadratic']
            assert np.all(np.diagonal(Q, axis1=1, axis2=2) == 0), 'Agent quadratic has non-zero diagonal'
            assert np.all(Q >= 0), 'Agent quadratic has negative values'
        if self.has_quadratic_item:
            Q = id['quadratic']
            assert np.all(np.diagonal(Q, axis1=0, axis2=1) == 0), 'Item quadratic has non-zero diagonal'
            assert np.all(Q >= 0), 'Item quadratic has negative values'

    def build_quadratic_matrix(self, theta):
        n_agents = self.data_manager.num_local_agent
        n_items = self.dimensions_cfg.num_items
        ad = self.data_manager.local_data['agent_data']
        id = self.data_manager.local_data['item_data']
        linear = np.zeros((n_agents, n_items))
        quadratic = np.zeros((n_agents, n_items, n_items))
        if self.has_modular_agent:
            linear += ad['modular'] @ theta[self.modular_agent_slice]
        if self.has_modular_item:
            linear += id['modular'] @ theta[self.modular_item_slice]
        if self.has_quadratic_agent:
            quadratic += ad['quadratic'] @ theta[self.quadratic_agent_slice]
        if self.has_quadratic_item:
            quadratic += id['quadratic'] @ theta[self.quadratic_item_slice]
        for i in range(n_agents):
            I = np.eye(n_items, dtype=bool)
            for j in range(n_items):
                linear[i, j] += self.oracles_manager.error_oracle(I[j:j+1], np.array([i]))[0]
        return linear, quadratic
