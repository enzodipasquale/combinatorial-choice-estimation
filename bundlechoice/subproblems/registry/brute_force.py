import numpy as np
from itertools import product
from ..subproblem_base import SubproblemSolver

class BruteForceSolver(SubproblemSolver):

    def initialize(self):
        self._all_bundles = np.array(list(product([0, 1], repeat=self.dimensions_cfg.n_items)), dtype=bool)

    def solve(self, theta):
        n_agents = self.comm_manager.num_local_agent
        n_items = self.dimensions_cfg.n_items
        results = np.zeros((n_agents, n_items), dtype=bool)
        for i in range(n_agents):
            best_value, best_bundle = float('-inf'), None
            for bundle in self._all_bundles:
                value = self.oracles_manager.utility_oracle_individual(bundle, theta, i)
                if value > best_value:
                    best_value, best_bundle = value, bundle.copy()
            results[i] = best_bundle
        return results
