import numpy as np
from ..subproblem_base import SerialSubproblemBase

class GreedySubproblem(SerialSubproblemBase):
    find_best_item = None
    modular_errors = None

    def initialize_single_pb(self, local_id):
        self.modular_errors = self.oracles_manager._modular_local_errors[local_id]
        return None

    def solve_single_pb(self, local_id, theta, pb=None):
        if self.find_best_item is not None:
            return self._greedy_with_find_best_item(local_id, theta)
        return self._naive_greedy_solve(local_id, theta)

    def _naive_greedy_solve(self, local_id, theta):
        bundle = np.zeros(self.dimensions_cfg.num_items, dtype=bool)
        items_left = np.ones(self.dimensions_cfg.num_items, dtype=bool)
        base_utility = 0

        while np.any(items_left):
            best_item, best_utility = None, base_utility
            for j in np.where(items_left)[0]:
                bundle[j] = True
                utility = self.oracles_manager.utilities_oracle_individual(bundle, theta, local_id)
                if utility > best_utility:
                    best_item, best_utility = j, utility
                bundle[j] = False
            if best_item is None:
                break
            bundle[best_item] = True
            items_left[best_item] = False
        return bundle

    def _greedy_with_find_best_item(self, local_id, theta):
        bundle = np.zeros(self.dimensions_cfg.num_items, dtype=bool)
        items_left = np.ones(self.dimensions_cfg.num_items, dtype=bool)
        best_val = 0
        while np.any(items_left):
            best_item, val = self.find_best_item(local_id, bundle, items_left, theta, self.modular_errors)
            if val <= best_val:
                break
            bundle[best_item] = True
            items_left[best_item] = False
        return bundle
