import numpy as np
from ..subproblem_base import SerialSubproblemBase

class GreedySubproblem(SerialSubproblemBase):

    def initialize_single_pb(self, local_id):
        return None

    def solve_single_pb(self, local_id, theta, pb=None):
        n = self.dimensions_cfg.num_items
        bundle = np.zeros(n, dtype=bool)
        items_left = list(range(n))
        base_utility = self.oracles_manager.utilities_oracle(bundle[None, :], theta, local_id)[0]
        while items_left:
            best_item, best_utility = None, base_utility
            for j in items_left:
                bundle[j] = True
                utility = self.oracles_manager.utilities_oracle(bundle[None, :], theta, local_id)[0]
                if utility > best_utility:
                    best_item, best_utility = j, utility
                bundle[j] = False
            if best_item is None:
                break
            bundle[best_item] = True
            items_left.remove(best_item)
            base_utility = best_utility
        return bundle
