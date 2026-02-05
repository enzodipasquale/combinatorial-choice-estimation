import numpy as np
from ..subproblem_base import SerialSubproblemBase

class GreedySubproblem(SerialSubproblemBase):
    find_best_item = None

    def initialize_single_pb(self, local_id):
        return None

    def solve_single_pb(self, local_id, theta, pb=None):
        if self.find_best_item is not None:
            return self._greedy_with_find_best_item(local_id, theta)
        return self._naive_greedy_solve(local_id, theta)

    def _naive_greedy_solve(self, local_id, theta):
        bundle = np.zeros(self.dimensions_cfg.n_items, dtype=bool)
        items_left = np.ones(self.dimensions_cfg.n_items, dtype=bool)
        base_utility = self.oracles_manager.utility_oracle_individual(bundle, theta, local_id)

        while np.any(items_left):
            best_item, best_utility = None, base_utility
            for j in np.where(items_left)[0]:
                bundle[j] = True
                utility = self.oracles_manager.utility_oracle_individual(bundle, theta, local_id)
                if utility > best_utility:
                    best_item, best_utility = j, utility
                bundle[j] = False
            if best_item is None:
                break
            bundle[best_item] = True
            items_left[best_item] = False
            base_utility = best_utility
        return bundle

    def _greedy_with_find_best_item(self, local_id, theta):
        modular_error = self.oracles_manager._local_modular_errors[local_id]
        bundle = np.zeros(self.dimensions_cfg.n_items, dtype=bool)
        items_left = np.ones(self.dimensions_cfg.n_items, dtype=bool)
        best_val = 0
        while np.any(items_left):
            best_item, val = self.find_best_item(local_id, bundle, items_left, theta, best_val, self.data_manager.local_data, modular_error)
            if val <= best_val:
                break
            bundle[best_item] = True
            items_left[best_item] = False
            best_val = val 
        return bundle
