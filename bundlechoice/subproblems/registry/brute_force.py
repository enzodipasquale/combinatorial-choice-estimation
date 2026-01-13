"""
Brute force subproblem solver.

Enumerates all 2^J bundles - only for small J.
Uses the error_oracle, supporting non-modular errors.
"""

import numpy as np
from typing import Any, Optional
from numpy.typing import NDArray
from itertools import product
from ..base import SerialSubproblemBase


class BruteForceSubproblem(SerialSubproblemBase):
    """Brute force solver: enumerate all bundles."""
    
    def initialize(self, local_id: int) -> None:
        """Pre-compute all bundles for efficiency."""
        if not hasattr(self, '_all_bundles'):
            self._all_bundles = np.array(list(product([0, 1], repeat=self.num_items)), dtype=np.float64)
        return None
    
    def solve(self, local_id: int, theta: NDArray[np.float64], 
             pb: Optional[Any] = None) -> NDArray[np.bool_]:
        """Find best bundle by exhaustive enumeration."""
        max_value = float('-inf')
        best_bundle = None
        
        for bundle in self._all_bundles:
            features = self.features_oracle(local_id, bundle, self.local_data)
            error = self.error_oracle(local_id, bundle, self.local_data)
            value = features @ theta + error
            
            if value > max_value:
                max_value = value
                best_bundle = bundle.copy()
        
        return best_bundle.astype(bool)
