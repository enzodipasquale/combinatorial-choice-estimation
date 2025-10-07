"""
Configuration profiles for common use cases.

This module provides pre-defined configuration profiles that researchers
can use as starting points for their experiments.
"""

PROFILES = {
    'fast': {
        'row_generation': {
            'max_iters': 20,
            'tolerance_optimality': 0.01,
            'gurobi_settings': {'OutputFlag': 0}
        },
        'subproblem': {'name': 'Greedy'},
        'ellipsoid': {'num_iters': 50},
    },
    'accurate': {
        'row_generation': {
            'max_iters': 200,
            'tolerance_optimality': 1e-6,
            'gurobi_settings': {'OutputFlag': 0}
        },
        'subproblem': {'name': 'QuadSupermodularNetwork'},
        'ellipsoid': {'num_iters': 500},
    },
    'debug': {
        'row_generation': {
            'max_iters': 5,
            'tolerance_optimality': 0.1,
            'gurobi_settings': {'OutputFlag': 1}
        },
        'subproblem': {'name': 'Greedy'},
        'ellipsoid': {'num_iters': 10, 'verbose': True},
    },
    'balanced': {
        'row_generation': {
            'max_iters': 100,
            'tolerance_optimality': 1e-4,
            'gurobi_settings': {'OutputFlag': 0}
        },
        'subproblem': {'name': 'OptimizedGreedy'},
        'ellipsoid': {'num_iters': 200},
    },
}

def load_profile(name, overrides=None):
    """
    Load a configuration profile with optional overrides.
    
    Args:
        name: Profile name ('fast', 'accurate', 'debug', or 'balanced')
        overrides: Optional dict to override profile settings
    
    Returns:
        dict: Configuration dictionary
    
    Raises:
        ValueError: If profile name is unknown
    
    Example:
        >>> cfg = load_profile('fast', {'dimensions': {'num_agents': 100, 'num_items': 50, 'num_features': 10}})
        >>> bc.load_config(cfg)
    """
    if name not in PROFILES:
        available = ', '.join(PROFILES.keys())
        raise ValueError(
            f"Unknown profile: '{name}'\n"
            f"Available profiles: {available}"
        )
    
    # Deep copy profile
    import copy
    cfg = copy.deepcopy(PROFILES[name])
    
    # Apply overrides
    if overrides:
        _deep_update(cfg, overrides)
    
    return cfg

def list_profiles():
    """
    List all available configuration profiles.
    
    Returns:
        list: Names of all available profiles
    """
    return list(PROFILES.keys())

def describe_profile(name):
    """
    Get a description of a configuration profile.
    
    Args:
        name: Profile name
    
    Returns:
        dict: Profile configuration
    """
    if name not in PROFILES:
        available = ', '.join(PROFILES.keys())
        raise ValueError(
            f"Unknown profile: '{name}'\n"
            f"Available profiles: {available}"
        )
    return PROFILES[name]

def _deep_update(base, updates):
    """Recursively update nested dictionaries."""
    for key, value in updates.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base
