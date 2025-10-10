"""
Workflow and convenience methods for BundleChoice.

This module contains high-level workflow methods that combine multiple
operations for common use cases.
"""

from contextlib import contextmanager


def generate_observations(bc, theta_true):
    """
    Generate observed bundles from true parameters and reload data.
    Handles the common workflow pattern automatically.
    
    Args:
        bc: BundleChoice instance
        theta_true: True parameter vector for generating observations
    
    Returns:
        Observed bundles (rank 0 only, None on other ranks)
    
    Example:
        >>> bc.data.load_and_scatter(input_data)
        >>> bc.features.build_from_data()
        >>> bc.generate_observations(theta_true)
        >>> theta_hat = bc.row_generation.solve()
    """
    obs_bundles = bc.subproblems.init_and_solve(theta_true)
    
    # Prepare input_data on rank 0
    if bc.is_root():
        if bc.data_manager.input_data is None:
            raise RuntimeError("Cannot generate observations without input_data")
        bc.data_manager.input_data["obs_bundle"] = obs_bundles
        updated_data = bc.data_manager.input_data
    else:
        updated_data = None
    
    # All ranks participate in scatter
    bc.data.load_and_scatter(updated_data)
    
    # Rebuild features if using auto-generated oracle
    if bc.feature_manager._features_oracle is not None:
        oracle_code = bc.feature_manager._features_oracle.__code__
        if 'features_oracle' in oracle_code.co_name:
            bc.features.build_from_data()
    
    return obs_bundles


@contextmanager
def temp_config(bc, **updates):
    """
    Temporarily modify configuration.
    
    Args:
        bc: BundleChoice instance
        **updates: Configuration updates to apply temporarily
    
    Example:
        >>> with bc.temp_config(row_generation={'max_iters': 5}):
        ...     quick_theta = bc.row_generation.solve()
        >>> # Config restored to original
        >>> final_theta = bc.row_generation.solve()
    """
    import copy
    original_config = copy.deepcopy(bc.config)
    try:
        bc.configure(updates)
        yield bc
    finally:
        bc.config = original_config


def quick_setup(bc, config, input_data, features_oracle=None):
    """
    Quick setup for common workflow.
    Combines configure, load_and_scatter, features, and subproblems.load().
    
    Args:
        bc: BundleChoice instance
        config: Configuration dict or YAML path
        input_data: Input data dictionary
        features_oracle: Feature function or None to auto-generate
    
    Returns:
        bc: BundleChoice instance for method chaining
    
    Example:
        >>> bc = BundleChoice().quick_setup(cfg, data, my_features)
        >>> theta = bc.row_generation.solve()
    """
    bc.configure(config)
    bc.data.load_and_scatter(input_data)
    
    if features_oracle is not None:
        bc.features.set_oracle(features_oracle)
    else:
        bc.features.build_from_data()
    
    bc.subproblems.load()
    return bc

