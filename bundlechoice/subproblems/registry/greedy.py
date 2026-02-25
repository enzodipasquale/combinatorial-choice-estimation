import numpy as np
from ..solver_base import SubproblemSolver

class GreedySolver(SubproblemSolver):
    find_best_item = None

    def solve(self, theta):
        n_agents = self.comm_manager.num_local_agent
        n_items = self.dimensions_cfg.n_items
        results = np.zeros((n_agents, n_items), dtype=bool)
        for i in range(n_agents):
            if self.find_best_item is not None:
                results[i] = self._greedy_with_find_best_item(i, theta)
            else:
                results[i] = self._naive_greedy_solve(i, theta)
        return results

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
