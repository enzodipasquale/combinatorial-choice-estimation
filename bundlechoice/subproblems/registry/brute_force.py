import numpy as np
from itertools import product
from ..subproblem_base import SerialSubproblemBase

class BruteForceSubproblem(SerialSubproblemBase):

    def initialize_single_pb(self, local_id):
        if not hasattr(self, '_all_bundles'):
            self._all_bundles = np.array(list(product([0, 1], repeat=self.dimensions_cfg.n_items)), dtype=bool)
        return None

    def solve_single_pb(self, local_id, theta, pb=None):
        best_value, best_bundle = float('-inf'), None
        for bundle in self._all_bundles:
            value = self.oracles_manager.utility_oracle_individual(bundle, theta, local_id)
            if value > best_value:
                best_value, best_bundle = value, bundle.copy()
        return best_bundle
