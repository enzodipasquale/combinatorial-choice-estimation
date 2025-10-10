"""
Validation and status methods for BundleChoice.

This module contains methods for validating setup and checking component status.
"""

from bundlechoice.utils import get_logger

logger = get_logger(__name__)


def validate_setup(bc, for_method='row_generation'):
    """
    Validate that all components are initialized for the specified estimation method.
    
    Args:
        bc: BundleChoice instance
        for_method: Estimation method to validate for ('row_generation', 'ellipsoid', or 'inequalities')
    
    Raises:
        SetupError: If setup is incomplete with helpful guidance
    
    Returns:
        bool: True if setup is valid
    """
    from bundlechoice.validation import validate_workflow
    
    operation_map = {
        'row_generation': 'solve_row_generation',
        'ellipsoid': 'solve_ellipsoid',
        'inequalities': 'solve_inequalities',
    }
    
    operation = operation_map.get(for_method, for_method)
    validate_workflow(bc, operation)
    
    logger.info("✅ Setup validated for %s", for_method)
    return True


def status(bc) -> dict:
    """
    Get setup status summary.
    
    Returns a dictionary with initialization status of all components.
    Useful for debugging setup issues without raising errors.
    
    Args:
        bc: BundleChoice instance
    
    Returns:
        dict: Dictionary with status information including:
            - config_loaded: Whether config is loaded
            - data_loaded: Whether data is loaded
            - features_set: Whether features oracle is set
            - subproblems_ready: Whether subproblem solver is loaded
            - dimensions: String representation of dimensions
            - subproblem: Name of subproblem algorithm
            - mpi_rank: Current MPI rank
            - mpi_size: Total number of MPI processes
    
    Example:
        >>> bc = BundleChoice()
        >>> bc.load_config(cfg)
        >>> status = bc.status()
        >>> if not status['features_set']:
        ...     bc.features.build_from_data()
    """
    return {
        'config_loaded': bc.config is not None,
        'data_loaded': bc.data_manager is not None and bc.data_manager.local_data is not None,
        'features_set': bc.feature_manager is not None and bc.feature_manager._features_oracle is not None,
        'subproblems_ready': bc.subproblem_manager is not None and bc.subproblem_manager.demand_oracle is not None,
        'dimensions': f"agents={bc.config.dimensions.num_agents}, items={bc.config.dimensions.num_items}, features={bc.config.dimensions.num_features}" if bc.config and bc.config.dimensions else 'Not set',
        'subproblem': bc.config.subproblem.name if bc.config and bc.config.subproblem and bc.config.subproblem.name else 'Not set',
        'mpi_rank': bc.rank,
        'mpi_size': bc.comm_size,
    }


def print_status(bc):
    """
    Print formatted setup status.
    
    Displays a human-readable summary of the current setup state,
    showing which components are initialized and key configuration values.
    
    Args:
        bc: BundleChoice instance
    
    Example:
        >>> bc = BundleChoice()
        >>> bc.load_config(cfg)
        >>> bc.data.load_and_scatter(input_data)
        >>> bc.print_status()
        === BundleChoice Status ===
        Config:      ✓
        Data:        ✓
        Features:    ✗
        Subproblems: ✗
        
        Dimensions:  agents=100, items=20, features=5
        Algorithm:   Greedy
        MPI:         rank 0/10
    """
    status_dict = status(bc)
    print("\n=== BundleChoice Status ===")
    print(f"Config:      {'✓' if status_dict['config_loaded'] else '✗'}")
    print(f"Data:        {'✓' if status_dict['data_loaded'] else '✗'}")
    print(f"Features:    {'✓' if status_dict['features_set'] else '✗'}")
    print(f"Subproblems: {'✓' if status_dict['subproblems_ready'] else '✗'}")
    print(f"\nDimensions:  {status_dict['dimensions']}")
    print(f"Algorithm:   {status_dict['subproblem']}")
    print(f"MPI:         rank {status_dict['mpi_rank']}/{status_dict['mpi_size']}")

