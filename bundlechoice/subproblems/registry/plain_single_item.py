from bundlechoice.subproblems.base import BatchSubproblemBase
import numpy as np
from typing import Optional, Any
from numpy.typing import NDArray

class PlainSingleItemSubproblem(BatchSubproblemBase):

    def initialize(self):
        info = self.data_manager.get_data_info()
        self.has_modular_agent = info['has_modular_agent']
        self.has_modular_item = info['has_modular_item']
        self.has_errors = info['has_errors']
        self.has_constraint_mask = info['has_constraint_mask']

    def solve(self, theta, pb=None):
        U_i_j = self.build_utilities(theta)
        if self.has_constraint_mask:
            U_i_j = np.where(self.data_manager.local_data['agent_data']['constraint_mask'], U_i_j, -np.inf)
        j_star = np.argmax(U_i_j, axis=1)
        max_vals = U_i_j[np.arange(self.data_manager.num_local_agent), j_star]
        optimal_bundles = (max_vals > 0)[:, None] & (np.arange(self.dimensions_cfg.num_items) == j_star[:, None])
        return optimal_bundles

    def build_utilities(self, theta):
        info = self.data_manager.get_data_info()
        U_i_j = np.zeros((self.data_manager.num_local_agent, self.dimensions_cfg.num_items))
        offset = 0
        if self.has_modular_agent:
            modular_agent = self.data_manager.local_data['agent_data']['modular']
            U_i_j += modular_agent @ theta[offset:offset + info['num_modular_agent']]
            offset += info['num_modular_agent']
        if self.has_modular_item:
            modular_item = self.data_manager.local_data['item_data']['modular']
            U_i_j += modular_item @ theta[offset:offset + info['num_modular_item']]
            offset += info['num_modular_item']
        if self.has_errors:
            U_i_j += self.data_manager.local_data['errors']
        return U_i_j