import numpy as np
from typing import Any, Optional
from numpy.typing import NDArray
from itertools import product
from ..base import SerialSubproblemBase

class BruteForceSubproblem(SerialSubproblemBase):

    def initialize(self, local_id):
        if not hasattr(self, '_all_bundles'):
            self._all_bundles = np.array(list(product([0, 1], repeat=self.dimensions_cfg.num_items)), dtype=np.float64)
        return None

    def solve(self, local_id, theta, pb=None):
        max_value = float('-inf')
        best_bundle = None
        for bundle in self._all_bundles:
            features = self.features_oracle(local_id, bundle, self.data_manager.local_data)
            error = self.error_oracle(local_id, bundle, self.data_manager.local_data)
            value = features @ theta + error
            if value > max_value:
                max_value = value
                best_bundle = bundle.copy()
        return best_bundle.astype(bool)