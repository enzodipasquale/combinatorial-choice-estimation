import numpy as np
from typing import Any, Optional
from ..base import SerialSubproblemBase
# import time

class GreedySubproblem(SerialSubproblemBase):
    """
    Greedy subproblem solver for bundle choice estimation.
    
    This solver uses a greedy algorithm to find approximately optimal bundles
    by iteratively adding items with the highest marginal value.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._supports_vectorized_features = None
    
    def initialize(self, local_id: int) -> Optional[Any]:
        """
        Initialize the greedy subproblem and check for vectorized feature support.
        
        Args:
            local_id: Local agent ID
            
        Returns:
            Problem state (None for greedy, but could be used for caching)
        """
        self._check_vectorized_feature_support(local_id)
        return None
    
    def solve(self, local_id: int, theta: np.ndarray, pb: Optional[Any] = None) -> np.ndarray:
        """
        Solve the greedy subproblem for the given agent and parameters.
        
        Args:
            local_id: Local agent ID
            theta: Parameter vector
            pb: Problem state from initialize() (ignored for greedy algorithm)
            
        Returns:
            Binary array representing the optimal bundle
        """
        # Get local data
        error_j = self.local_data["errors"][local_id]
        num_items = self.num_items
        
        # Initialize empty bundle
        bundle = np.zeros(num_items, dtype=bool)
        items_left = np.arange(num_items)
        # Greedy algorithm: iteratively add best item
        # tic = time.time()
        while len(items_left) > 0:
            best_item, best_val = self._find_best_item(local_id, bundle, items_left, theta, error_j)
            # If no positive marginal value, stop
            if best_val <= 0:
                break
                
            # Add best item to bundle
            bundle[best_item] = True
            items_left = items_left[items_left != best_item]
        # toc = time.time() - tic
        # print(f"Done with local id {local_id} time {toc}")
        return bundle

    def _check_vectorized_feature_support(self, local_id: int) -> None:
        """
        Check if the current get_features function supports vectorized computation.
        Tests if get_features can handle multiple bundles as a numpy array.
        """
        try:
            # Test with multiple bundles: shape (num_items, m) where m=2
            test_bundles = np.zeros((self.num_items, 2), dtype=bool)
            test_bundles[0, 0] = True  # First bundle has item 0
            test_bundles[1, 1] = True  # Second bundle has item 1
            
            # Try vectorized call with multiple bundles
            vectorized_result = self.get_features(local_id, test_bundles, self.local_data)
            
            # Check if result has expected shape (k features × m bundles)
            self._supports_vectorized_features = (
                isinstance(vectorized_result, np.ndarray) and 
                len(vectorized_result.shape) == 2 and 
                vectorized_result.shape[1] == 2
            )
        except (TypeError, ValueError, AttributeError, IndexError):
            self._supports_vectorized_features = False
    
    def _get_vectorized_features(self, local_id: int, base_bundle: np.ndarray, items_to_add: np.ndarray) -> np.ndarray:
        """
        Get features for multiple items in a vectorized fashion.
        Creates multiple bundles by adding each item to the base bundle.
        
        Args:
            local_id: Local agent ID
            base_bundle: Current bundle
            items_to_add: Array of item indices to add
            
        Returns:
            Array of shape (num_features, num_items) with features for each item
        """
        bundles = np.repeat(base_bundle[:, None], len(items_to_add), axis=1)
        bundles[items_to_add, np.arange(len(items_to_add))] = True
        
        # bundles = (np.arange(self.num_items)[:, None] == items_to_add[None, :]) + base_bundle[:, None]
        return self.get_features(local_id, bundles, self.local_data)

    def _find_best_item(
        self, 
        local_id: int, 
        bundle: np.ndarray, 
        items_left: np.ndarray, 
        theta: np.ndarray, 
        error_j: np.ndarray
    ) -> tuple[int, float]:
        """
        Find the best item to add to the current bundle.
        
        Args:
            local_id: Local agent ID
            bundle: Current bundle (boolean array)
            items_left: Array of remaining item indices
            theta: Parameter vector
            error_j: Error values for current agent
            
        Returns:
            Tuple of (best_item_index, best_marginal_value)
        """
        if self._supports_vectorized_features and len(items_left) > 1:
            return self._find_best_item_vectorized(local_id, bundle, items_left, theta, error_j)
        else:
            return self._find_best_item_standard(local_id, bundle, items_left, theta, error_j)
    
    
    def _find_best_item_vectorized(
            self, 
            local_id: int, 
            bundle: np.ndarray, 
            items_left: np.ndarray, 
            theta: np.ndarray, 
            error_j: np.ndarray
        ) -> tuple[int, float]:
        """
        Find the best item using vectorized approach for better performance.
        """
        # Get base features
        base_features = self.get_features(local_id, bundle, self.local_data)
        
        # Get vectorized features for all remaining items
        vectorized_features = self._get_vectorized_features(local_id, bundle, items_left)
        
        # Calculate marginal values
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            marginal_values = theta @ vectorized_features - theta @ base_features + error_j[items_left]
        
        # Find best item
        best_idx = np.argmax(marginal_values)
        best_item = items_left[best_idx]
        best_val = float(marginal_values[best_idx])
        
        return best_item, best_val



    def _find_best_item_standard(
        self, 
        local_id: int, 
        bundle: np.ndarray, 
        items_left: np.ndarray, 
        theta: np.ndarray, 
        error_j: np.ndarray
    ) -> tuple[int, float]:
        """
        Find the best item using standard (non-vectorized) approach.
        """
        # Cache base features once - this is the key optimization
        base_features = self.get_features(local_id, bundle, self.local_data)
        
        best_val = -np.inf
        best_item = -1
        
        for j in items_left:
            # Try adding item j
            bundle[j] = True
            new_x_k = self.get_features(local_id, bundle, self.local_data)
            
            # Calculate marginal value
            marginal_j = float(error_j[j])
            if base_features is not None and new_x_k is not None:
                marginal_j += float((new_x_k - base_features) @ theta)
            
            # Remove item j for next iteration
            bundle[j] = False
            
            if marginal_j > best_val:
                best_val = marginal_j
                best_item = j
                
        return best_item, best_val