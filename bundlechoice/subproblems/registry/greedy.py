import numpy as np
from typing import Any, Optional
from ..base import SerialSubproblemBase


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
    
    def solve(self, local_id: int, lambda_k: np.ndarray, pb: Optional[Any] = None) -> np.ndarray:
        """
        Solve the greedy subproblem for the given agent and parameters.
        
        Args:
            local_id: Local agent ID
            lambda_k: Parameter vector
            pb: Problem state from initialize() (ignored for greedy algorithm)
            
        Returns:
            Binary array representing the optimal bundle
        """
        # Get local data
        error_j = self.local_data["errors"][local_id]
        num_items = self.num_items
        
        # Initialize empty bundle
        B_j = np.zeros(num_items, dtype=bool)
        items_left = np.arange(num_items)
        
        # Greedy algorithm: iteratively add best item
        while len(items_left) > 0:
            best_item, best_val = self._find_best_item(local_id, B_j, items_left, lambda_k, error_j)
            
            # If no positive marginal value, stop
            if best_val <= 0:
                break
                
            # Add best item to bundle
            B_j[best_item] = True
            items_left = items_left[items_left != best_item]
        
        return B_j

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
    
    def _get_vectorized_features(self, local_id: int, B_j: np.ndarray, items_to_add: np.ndarray) -> np.ndarray:
        """
        Get features for multiple items in a vectorized fashion.
        Creates multiple bundles by adding each item to the base bundle.
        
        Args:
            local_id: Local agent ID
            B_j: Current bundle
            items_to_add: Array of item indices to add
            
        Returns:
            Array of shape (num_features, num_items) with features for each item
        """
        # Create multiple bundles: each column is a bundle with one additional item
        num_items_to_add = len(items_to_add)
        bundles = np.tile(B_j, (num_items_to_add, 1)).T  # Shape: (num_items, num_items_to_add)
        
        # Add each item to its corresponding bundle
        for i, item in enumerate(items_to_add):
            bundles[item, i] = True
        
        # Get vectorized features
        return self.get_features(local_id, bundles, self.local_data)

    def _find_best_item(
        self, 
        local_id: int, 
        B_j: np.ndarray, 
        items_left: np.ndarray, 
        lambda_k: np.ndarray, 
        error_j: np.ndarray
    ) -> tuple[int, float]:
        """
        Find the best item to add to the current bundle.
        
        Args:
            local_id: Local agent ID
            B_j: Current bundle (boolean array)
            items_left: Array of remaining item indices
            lambda_k: Parameter vector
            error_j: Error values for current agent
            
        Returns:
            Tuple of (best_item_index, best_marginal_value)
        """
        if self._supports_vectorized_features and len(items_left) > 1:
            return self._find_best_item_vectorized(local_id, B_j, items_left, lambda_k, error_j)
        else:
            return self._find_best_item_standard(local_id, B_j, items_left, lambda_k, error_j)
    
    def _find_best_item_standard(
        self, 
        local_id: int, 
        B_j: np.ndarray, 
        items_left: np.ndarray, 
        lambda_k: np.ndarray, 
        error_j: np.ndarray
    ) -> tuple[int, float]:
        """
        Find the best item using standard (non-vectorized) approach.
        """
        # Cache base features once - this is the key optimization
        base_x_k = self.get_features(local_id, B_j, self.local_data)
        
        best_val = -np.inf
        best_item = -1
        
        for j in items_left:
            # Try adding item j
            B_j[j] = True
            new_x_k = self.get_features(local_id, B_j, self.local_data)
            
            # Calculate marginal value
            marginal_j = float(error_j[j])
            if base_x_k is not None and new_x_k is not None:
                marginal_j += float((new_x_k - base_x_k) @ lambda_k)
            
            # Remove item j for next iteration
            B_j[j] = False
            
            if marginal_j > best_val:
                best_val = marginal_j
                best_item = j
                
        return best_item, best_val
    
    def _find_best_item_vectorized(
            self, 
            local_id: int, 
            B_j: np.ndarray, 
            items_left: np.ndarray, 
            lambda_k: np.ndarray, 
            error_j: np.ndarray
        ) -> tuple[int, float]:
        """
        Find the best item using vectorized approach for better performance.
        """
        # Get base features
        base_x_k = self.get_features(local_id, B_j, self.local_data)
        
        # Get vectorized features for all remaining items
        vectorized_features = self._get_vectorized_features(local_id, B_j, items_left)
        
        # Calculate marginal values for all items at once
        marginal_values = error_j[items_left].copy()
        
        # Add feature-based marginal values
        feature_diffs = vectorized_features - base_x_k.reshape(-1, 1)
        marginal_values += (feature_diffs.T @ lambda_k).flatten()
        
        # Find best item
        best_idx = np.argmax(marginal_values)
        best_item = items_left[best_idx]
        best_val = float(marginal_values[best_idx])
        
        return best_item, best_val 