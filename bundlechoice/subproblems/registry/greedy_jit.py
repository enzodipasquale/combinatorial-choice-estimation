import numpy as np
from typing import Any, Optional
from ..base import SerialSubproblemBase

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create a dummy jit decorator if numba is not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# JIT-compiled functions for hot paths
@jit(nopython=True, cache=True)
def _compute_marginal_values_vectorized(theta, vectorized_features, base_features, error_j):
    """JIT-compiled marginal value computation for vectorized approach."""
    return theta @ vectorized_features - theta @ base_features + error_j

@jit(nopython=True, cache=True)
def _compute_marginal_value_standard(theta, new_features, base_features, error_j):
    """JIT-compiled single marginal value computation."""
    return error_j + (new_features - base_features) @ theta

@jit(nopython=True, cache=True)
def _find_best_item_vectorized_core(marginal_values, items_left):
    """JIT-compiled core logic for finding best item from marginal values."""
    best_idx = np.argmax(marginal_values)
    best_item = items_left[best_idx]
    best_val = marginal_values[best_idx]
    return best_item, best_val

class GreedyJITSubproblem(SerialSubproblemBase):
    """
    JIT-optimized Greedy subproblem solver for bundle choice estimation.
    
    This extends OptimizedGreedy with Numba JIT compilation for hot paths:
    1. Marginal value computations (vectorized and standard)
    2. Best item selection logic
    3. Core numerical operations
    
    Falls back to pure NumPy if Numba is not available.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._supports_vectorized_features = None
        self._base_features_cache = None
        self._current_bundle_hash = None
        self._use_jit = NUMBA_AVAILABLE
    
    def initialize(self, local_id: int) -> Optional[Any]:
        """
        Initialize the greedy subproblem and check for vectorized feature support.
        
        Args:
            local_id: Local agent ID
            
        Returns:
            Problem state (None for greedy, but could be used for caching)
        """
        self._check_vectorized_feature_support(local_id)
        if self._use_jit:
            print(f"  🚀 JIT compilation enabled for agent {local_id}")
        else:
            print(f"  ⚠️  JIT compilation disabled (Numba not available)")
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
        
        # Pre-compute base features for empty bundle (major optimization!)
        self._base_features_cache = self.features_oracle(local_id, bundle, self.local_data)
        self._current_bundle_hash = hash(bundle.tobytes())
        
        # Greedy algorithm: iteratively add best item
        while len(items_left) > 0:
            best_item, best_val = self._find_best_item_jit(local_id, bundle, items_left, theta, error_j)
            # If no positive marginal value, stop
            if best_val <= 0:
                break
                
            # Add best item to bundle
            bundle[best_item] = True
            items_left = items_left[items_left != best_item]
            
            # Update cache for next iteration
            self._update_base_features_cache(local_id, bundle)
            
        return bundle

    def _check_vectorized_feature_support(self, local_id: int) -> None:
        """
        Check if the current features_oracle function supports vectorized computation.
        Tests if features_oracle can handle multiple bundles as a numpy array.
        """
        try:
            # Test with multiple bundles: shape (num_items, m) where m=2
            test_bundles = np.zeros((self.num_items, 2), dtype=bool)
            test_bundles[0, 0] = True  # First bundle has item 0
            test_bundles[1, 1] = True  # Second bundle has item 1
            
            # Try vectorized call with multiple bundles
            vectorized_result = self.features_oracle(local_id, test_bundles, self.local_data)
            
            # Check if result has expected shape (k features × m bundles)
            self._supports_vectorized_features = (
                isinstance(vectorized_result, np.ndarray) and 
                len(vectorized_result.shape) == 2 and 
                vectorized_result.shape[1] == 2
            )
        except (TypeError, ValueError, AttributeError, IndexError):
            self._supports_vectorized_features = False
    
    def _update_base_features_cache(self, local_id: int, bundle: np.ndarray) -> None:
        """Update the base features cache when bundle changes."""
        bundle_hash = hash(bundle.tobytes())
        if bundle_hash != self._current_bundle_hash:
            self._base_features_cache = self.features_oracle(local_id, bundle, self.local_data)
            self._current_bundle_hash = bundle_hash

    def _get_vectorized_features_optimized(self, local_id: int, base_bundle: np.ndarray, items_to_add: np.ndarray) -> np.ndarray:
        """
        Get features for multiple items in a vectorized fashion - OPTIMIZED VERSION.
        
        Key optimization: Only create bundles for items we're actually testing,
        and use more efficient array operations.
        """
        if len(items_to_add) == 0:
            return np.empty((self._base_features_cache.shape[0], 0))
        
        # More efficient bundle creation
        bundles = np.tile(base_bundle[:, None], (1, len(items_to_add)))
        bundles[items_to_add, np.arange(len(items_to_add))] = True
        
        return self.features_oracle(local_id, bundles, self.local_data)

    def _find_best_item_jit(
        self, 
        local_id: int, 
        bundle: np.ndarray, 
        items_left: np.ndarray, 
        theta: np.ndarray, 
        error_j: np.ndarray
    ) -> tuple[int, float]:
        """
        Find the best item to add to the current bundle - JIT OPTIMIZED VERSION.
        
        Uses JIT compilation for hot paths while maintaining correctness.
        """
        if self._supports_vectorized_features and len(items_left) > 1:
            return self._find_best_item_vectorized_jit(local_id, bundle, items_left, theta, error_j)
        else:
            return self._find_best_item_standard_jit(local_id, bundle, items_left, theta, error_j)
    
    def _find_best_item_vectorized_jit(
            self, 
            local_id: int, 
            bundle: np.ndarray, 
            items_left: np.ndarray, 
            theta: np.ndarray, 
            error_j: np.ndarray
        ) -> tuple[int, float]:
        """
        Find the best item using vectorized approach with JIT optimization.
        """
        # Use cached base features (major optimization!)
        base_features = self._base_features_cache
        
        # Get vectorized features for all remaining items
        vectorized_features = self._get_vectorized_features_optimized(local_id, bundle, items_left)
        
        # Calculate marginal values using JIT-compiled function
        if self._use_jit:
            marginal_values = _compute_marginal_values_vectorized(
                theta, vectorized_features, base_features, error_j[items_left]
            )
        else:
            # Fallback to pure NumPy
            marginal_values = theta @ vectorized_features - theta @ base_features + error_j[items_left]
        
        # Find best item using JIT-compiled function
        if self._use_jit:
            best_item, best_val = _find_best_item_vectorized_core(marginal_values, items_left)
        else:
            # Fallback to pure NumPy
            best_idx = np.argmax(marginal_values)
            best_item = items_left[best_idx]
            best_val = float(marginal_values[best_idx])
        
        return best_item, best_val

    def _find_best_item_standard_jit(
        self, 
        local_id: int, 
        bundle: np.ndarray, 
        items_left: np.ndarray, 
        theta: np.ndarray, 
        error_j: np.ndarray
    ) -> tuple[int, float]:
        """
        Find the best item using standard approach with JIT optimization.
        """
        # Use cached base features (major optimization!)
        base_features = self._base_features_cache
        
        best_val = -np.inf
        best_item = -1
        
        for j in items_left:
            # Try adding item j
            bundle[j] = True
            new_x_k = self.features_oracle(local_id, bundle, self.local_data)
            
            # Calculate marginal value using JIT-compiled function
            if self._use_jit:
                marginal_j = _compute_marginal_value_standard(
                    theta, new_x_k, base_features, error_j[j]
                )
            else:
                # Fallback to pure NumPy
                marginal_j = float(error_j[j])
                if base_features is not None and new_x_k is not None:
                    marginal_j += float((new_x_k - base_features) @ theta)
            
            # Remove item j for next iteration
            bundle[j] = False
            
            if marginal_j > best_val:
                best_val = marginal_j
                best_item = j
                
        return best_item, best_val
