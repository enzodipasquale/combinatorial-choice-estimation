"""
Greedy subproblem solver for bundle choice.

Iteratively adds items with highest marginal value until no improvement.
"""

import numpy as np
from typing import Any, Optional
from numpy.typing import NDArray
from ..base import BaseSerialSubproblem


# ============================================================================
# Greedy Subproblem Solver
# ============================================================================

class GreedySubproblem(BaseSerialSubproblem):
    """Greedy subproblem solver: iteratively adds best items."""
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._supports_vectorized_features: Optional[bool] = None
    
    def initialize(self, local_id: int) -> None:
        """Initialize greedy solver."""
        self._check_vectorized_feature_support(local_id)
        return None
    
    def solve(self, local_id: int, theta: NDArray[np.float64], 
             pb: Optional[Any] = None) -> NDArray[np.bool_]:
        """Solve greedy subproblem: iteratively add best items."""
        error_j = self.local_data["errors"][local_id]
        num_items = self.num_items
        
        bundle = np.zeros(num_items, dtype=bool)
        items_left = np.arange(num_items)
        
        while len(items_left) > 0:
            base_features = self.features_oracle(local_id, bundle, self.local_data)
            base_value = base_features @ theta

            best_item, best_val = self.find_best_item(
                local_id, bundle, items_left, theta, error_j
            )
            if best_val <= base_value:
                break
                
            bundle[best_item] = True
            
            items_left = items_left[items_left != best_item]
        
        return bundle


    def find_best_item(
        self, local_id: int, base_bundle: NDArray[np.bool_], items_left: NDArray[np.int_], 
        theta: NDArray[np.float64], error_j: NDArray[np.float64]
    ) -> tuple[int, float]:
      
        if self._supports_vectorized_features and len(items_left) > 1:
            bundles = np.repeat(base_bundle[:, None], len(items_left), axis=1)
            bundles[items_left, np.arange(len(items_left))] = True
            vectorized_features = self.features_oracle(local_id, bundles, self.local_data)
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                values = theta @ vectorized_features  + error_j[items_left]
            best_idx = np.argmax(values)
            return items_left[best_idx], values[best_idx]
        else:
            best_val = -np.inf
            best_item = -1
            
            for j in items_left:
                base_bundle[j] = True
                features_with_j = self.features_oracle(local_id, base_bundle, self.local_data)
                value = error_j[j] + features_with_j  @ theta
                base_bundle[j] = False
                
                if value > best_val:
                    best_val = value
                    best_item = j
            
            return best_item, best_val
    

    def _check_vectorized_feature_support(self, local_id: int) -> None:
        """Check if features_oracle supports vectorized computation."""
        try:
            test_bundles = np.zeros((self.num_items, 2), dtype=bool)
            test_bundles[0, 0] = True
            test_bundles[1, 1] = True
            
            data_to_use = self.local_data if self.local_data is not None else None
            vectorized_result = self.features_oracle(local_id, test_bundles, data_to_use)
            
            self._supports_vectorized_features = (
                isinstance(vectorized_result, np.ndarray) and 
                len(vectorized_result.shape) == 2 and 
                vectorized_result.shape[1] == 2
            )
        except (TypeError, ValueError, AttributeError, IndexError):
            self._supports_vectorized_features = False