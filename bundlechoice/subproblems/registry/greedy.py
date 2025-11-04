"""
Greedy subproblem solver for bundle choice.

Iteratively adds items with highest marginal value until no improvement.
"""

import numpy as np
from typing import Any, Optional
from numpy.typing import NDArray
from ..base import SerialSubproblemBase


# ============================================================================
# Greedy Subproblem Solver
# ============================================================================

class GreedySubproblem(SerialSubproblemBase):
    """Greedy subproblem solver: iteratively adds best items."""
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._supports_vectorized_features: Optional[bool] = None
    
    def initialize(self, local_id: int) -> None:
        """Initialize greedy solver and check vectorized feature support."""
        self._check_vectorized_feature_support(local_id)
        return None
    
    def solve(self, local_id: int, theta: NDArray[np.float64], 
             pb: Optional[Any] = None) -> NDArray[np.bool_]:
        """Solve greedy subproblem: iteratively add best items."""
        error_j = self.local_data["errors"][local_id]
        num_items = self.num_items
        
        bundle = np.zeros(num_items, dtype=bool)
        items_left = np.arange(num_items)
        base_features = self.features_oracle(local_id, bundle, self.local_data)
        
        while len(items_left) > 0:
            best_item, best_val = self._find_best_item_cached(
                local_id, bundle, items_left, theta, error_j, base_features
            )
            if best_val <= 0:
                break
                
            bundle[best_item] = True
            base_features = self.features_oracle(local_id, bundle, self.local_data)
            items_left = items_left[items_left != best_item]
        
        return bundle

    def _check_vectorized_feature_support(self, local_id: int) -> None:
        """Check if features_oracle supports vectorized computation."""
        try:
            test_bundles = np.zeros((self.num_items, 2), dtype=bool)
            test_bundles[0, 0] = True
            test_bundles[1, 1] = True
            
            vectorized_result = self.features_oracle(local_id, test_bundles, self.local_data)
            
            self._supports_vectorized_features = (
                isinstance(vectorized_result, np.ndarray) and 
                len(vectorized_result.shape) == 2 and 
                vectorized_result.shape[1] == 2
            )
        except (TypeError, ValueError, AttributeError, IndexError):
            self._supports_vectorized_features = False
    
    def _get_vectorized_features(self, local_id: int, base_bundle: NDArray[np.bool_], 
                                 items_to_add: NDArray[np.int_]) -> NDArray[np.float64]:
        """Get features for multiple items vectorized (creates bundles with each item added to base)."""
        bundles = np.repeat(base_bundle[:, None], len(items_to_add), axis=1)
        bundles[items_to_add, np.arange(len(items_to_add))] = True
        return self.features_oracle(local_id, bundles, self.local_data)

    def _find_best_item(
        self, local_id: int, bundle: NDArray[np.bool_], items_left: NDArray[np.int_], 
        theta: NDArray[np.float64], error_j: NDArray[np.float64]
    ) -> tuple[int, float]:
        """Find best item to add (uses vectorized if supported)."""
        if self._supports_vectorized_features and len(items_left) > 1:
            return self._find_best_item_vectorized(local_id, bundle, items_left, theta, error_j)
        return self._find_best_item_standard(local_id, bundle, items_left, theta, error_j)
    
    def _find_best_item_cached(
        self, local_id: int, bundle: NDArray[np.bool_], items_left: NDArray[np.int_], 
        theta: NDArray[np.float64], error_j: NDArray[np.float64], base_features: NDArray[np.float64]
    ) -> tuple[int, float]:
        """Find best item using cached base features (optimized)."""
        if self._supports_vectorized_features and len(items_left) > 1:
            vectorized_features = self._get_vectorized_features(local_id, bundle, items_left)
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                marginal_values = theta @ vectorized_features - theta @ base_features + error_j[items_left]
            best_idx = np.argmax(marginal_values)
            return items_left[best_idx], float(marginal_values[best_idx])
        else:
            best_val = -np.inf
            best_item = -1
            
            for j in items_left:
                bundle[j] = True
                new_x_k = self.features_oracle(local_id, bundle, self.local_data)
                marginal_j = float(error_j[j]) + float((new_x_k - base_features) @ theta)
                bundle[j] = False
                
                if marginal_j > best_val:
                    best_val = marginal_j
                    best_item = j
            
            return best_item, best_val
    
    def _find_best_item_vectorized(
        self, local_id: int, bundle: NDArray[np.bool_], items_left: NDArray[np.int_], 
        theta: NDArray[np.float64], error_j: NDArray[np.float64]
    ) -> tuple[int, float]:
        """Find best item using vectorized approach."""
        base_features = self.features_oracle(local_id, bundle, self.local_data)
        vectorized_features = self._get_vectorized_features(local_id, bundle, items_left)
        
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            marginal_values = theta @ vectorized_features - theta @ base_features + error_j[items_left]
        
        best_idx = np.argmax(marginal_values)
        return items_left[best_idx], float(marginal_values[best_idx])

    def _find_best_item_standard(
        self, local_id: int, bundle: NDArray[np.bool_], items_left: NDArray[np.int_], 
        theta: NDArray[np.float64], error_j: NDArray[np.float64]
    ) -> tuple[int, float]:
        """Find best item using standard (non-vectorized) approach."""
        base_features = self.features_oracle(local_id, bundle, self.local_data)
        best_val = -np.inf
        best_item = -1
        
        for j in items_left:
            bundle[j] = True
            new_x_k = self.features_oracle(local_id, bundle, self.local_data)
            marginal_j = float(error_j[j])
            if base_features is not None and new_x_k is not None:
                marginal_j += float((new_x_k - base_features) @ theta)
            bundle[j] = False
            
            if marginal_j > best_val:
                best_val = marginal_j
                best_item = j
                
        return best_item, best_val