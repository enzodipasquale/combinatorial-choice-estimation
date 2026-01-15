import numpy as np
from ..subproblem_base import BatchSubproblemBase

class PlainSingleItemSubproblem(BatchSubproblemBase):

    def initialize(self):
        ma, qa, mi, qi = self.data_manager.quadratic_features_flags()
        self.has_modular_agent = ma
        self.has_modular_item = mi
        ad = self.data_manager.local_data['agent_data']
        self.has_constraint_mask = 'constraint_mask' in ad

    def solve(self, theta):
        U = self._build_utilities(theta)
        if self.has_constraint_mask:
            mask = self.data_manager.local_data['agent_data']['constraint_mask']
            U = np.where(mask, U, -np.inf)
        j_star = np.argmax(U, axis=1)
        max_vals = U[np.arange(self.data_manager.num_local_agent), j_star]
        n = self.dimensions_cfg.num_items
        return (max_vals > 0)[:, None] & (np.arange(n) == j_star[:, None])

    def _build_utilities(self, theta):
        n_agents = self.data_manager.num_local_agent
        n_items = self.dimensions_cfg.num_items
        U = np.zeros((n_agents, n_items))
        ad = self.data_manager.local_data['agent_data']
        id = self.data_manager.local_data['item_data']
        offset = 0
        if self.has_modular_agent:
            dim = ad['modular'].shape[-1]
            U += ad['modular'] @ theta[offset:offset + dim]
            offset += dim
        if self.has_modular_item:
            dim = id['modular'].shape[-1]
            U += id['modular'] @ theta[offset:offset + dim]
        I = np.eye(n_items, dtype=bool)
        for i in range(n_agents):
            for j in range(n_items):
                U[i, j] += self.oracles_manager.error_oracle(I[j:j+1], np.array([i]))[0]
        return U
