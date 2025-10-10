"""
Initialization methods for BundleChoice components.

This module contains all the _try_init_* methods that lazily initialize
the various managers and solvers used by BundleChoice.
"""

from bundlechoice.data_manager import DataManager
from bundlechoice.feature_manager import FeatureManager
from bundlechoice.subproblems.subproblem_manager import SubproblemManager
from bundlechoice.estimation import RowGenerationSolver
from bundlechoice.estimation.ellipsoid import EllipsoidSolver
from bundlechoice.estimation.inequalities import InequalitiesSolver


def try_init_data_manager(bc):
    """
    Initialize the DataManager if dimensions_cfg is set and not already initialized.
    
    Args:
        bc: BundleChoice instance
        
    Returns:
        DataManager: The initialized DataManager instance
        
    Raises:
        SetupError: If dimensions_cfg is not set
    """
    from bundlechoice.errors import SetupError
    
    if bc.config is None or bc.config.dimensions is None:
        raise SetupError(
            "Cannot initialize data manager - dimensions configuration not loaded",
            suggestion="Call bc.load_config(config_dict) with 'dimensions' section before accessing bc.data"
        )
    
    bc.data_manager = DataManager(
        dimensions_cfg=bc.config.dimensions,
        comm_manager=bc.comm_manager
    )
    return bc.data_manager


def try_init_feature_manager(bc):
    """
    Initialize the FeatureManager if dimensions_cfg is set.
    
    Args:
        bc: BundleChoice instance
        
    Returns:
        FeatureManager: The initialized FeatureManager instance
        
    Raises:
        SetupError: If dimensions_cfg is not set
    """
    from bundlechoice.errors import SetupError
    
    if bc.config is None or bc.config.dimensions is None:
        raise SetupError(
            "Cannot initialize feature manager - dimensions configuration not loaded",
            suggestion="Call bc.load_config(config_dict) with 'dimensions' section before accessing bc.features"
        )
    
    bc.feature_manager = FeatureManager(
        dimensions_cfg=bc.config.dimensions,
        comm_manager=bc.comm_manager,
        data_manager=bc.data_manager
    )
    return bc.feature_manager


def try_init_subproblem_manager(bc):
    """
    Initialize the subproblem manager if subproblem_cfg is set.
    
    Args:
        bc: BundleChoice instance
        
    Returns:
        SubproblemManager: The initialized SubproblemManager instance
        
    Raises:
        SetupError: If required managers or configs are not set
    """
    from bundlechoice.errors import SetupError
    
    if bc.data_manager is None or bc.feature_manager is None or bc.config is None or bc.config.subproblem is None:
        missing = []
        if bc.config is None or bc.config.subproblem is None:
            missing.append("config with 'subproblem' section")
        if bc.data_manager is None:
            missing.append("data")
        if bc.feature_manager is None:
            missing.append("features")
        
        raise SetupError(
            f"Cannot initialize subproblem manager - missing: {', '.join(missing)}",
            suggestion=(
                "Complete these steps:\n"
                "  1. bc.load_config(config_dict)  # Include 'subproblem' section\n"
                "  2. bc.data.load_and_scatter(input_data)\n"
                "  3. bc.features.build_from_data() or bc.features.set_oracle(fn)"
            )
        )

    bc.subproblem_manager = SubproblemManager(
        dimensions_cfg=bc.config.dimensions,
        comm_manager=bc.comm_manager,
        data_manager=bc.data_manager,
        feature_manager=bc.feature_manager,
        subproblem_cfg=bc.config.subproblem
    )
    bc.subproblem_manager.load()
    return bc.subproblem_manager


def try_init_row_generation_manager(bc):
    """
    Initialize the RowGenerationSolver if not already present.
    
    Args:
        bc: BundleChoice instance
        
    Returns:
        RowGenerationSolver: The initialized RowGenerationSolver instance
        
    Raises:
        RuntimeError: If required managers are not set
    """
    if bc.data_manager is None or bc.feature_manager is None or bc.subproblem_manager is None or bc.config is None or bc.config.row_generation is None:
        missing = []
        if bc.data_manager is None:
            missing.append("data (call bc.data.load_and_scatter(input_data))")
        if bc.feature_manager is None:
            missing.append("features (call bc.features.set_oracle(fn) or bc.features.build_from_data())")
        if bc.subproblem_manager is None:
            missing.append("subproblem (call bc.subproblems.load())")
        if bc.config is None or bc.config.row_generation is None:
            missing.append("row_generation config (add 'row_generation' to your config)")
        raise RuntimeError(
            "Cannot initialize row generation solver - missing setup:\n  " +
            "\n  ".join(missing) +
            "\n\nRun bc.validate_setup('row_generation') to check your configuration."
        )

    bc.row_generation_manager = RowGenerationSolver(
        comm_manager=bc.comm_manager,
        dimensions_cfg=bc.config.dimensions,
        row_generation_cfg=bc.config.row_generation,
        data_manager=bc.data_manager,
        feature_manager=bc.feature_manager,
        subproblem_manager=bc.subproblem_manager
    )
    return bc.row_generation_manager


def try_init_ellipsoid_manager(bc):
    """
    Initialize the EllipsoidSolver if not already present.
    
    Args:
        bc: BundleChoice instance
        
    Returns:
        EllipsoidSolver: The initialized EllipsoidSolver instance
        
    Raises:
        RuntimeError: If required managers are not set
    """
    if bc.data_manager is None or bc.feature_manager is None or bc.subproblem_manager is None or bc.config is None or bc.config.ellipsoid is None:
        missing = []
        if bc.data_manager is None:
            missing.append("data (call bc.data.load_and_scatter(input_data))")
        if bc.feature_manager is None:
            missing.append("features (call bc.features.set_oracle(fn) or bc.features.build_from_data())")
        if bc.subproblem_manager is None:
            missing.append("subproblem (call bc.subproblems.load())")
        if bc.config is None or bc.config.ellipsoid is None:
            missing.append("ellipsoid config (add 'ellipsoid' to your config)")
        raise RuntimeError(
            "Cannot initialize ellipsoid solver - missing setup:\n  " +
            "\n  ".join(missing) +
            "\n\nRun bc.validate_setup('ellipsoid') to check your configuration."
        )

    bc.ellipsoid_manager = EllipsoidSolver(
        comm_manager=bc.comm_manager,
        dimensions_cfg=bc.config.dimensions,
        ellipsoid_cfg=bc.config.ellipsoid,
        data_manager=bc.data_manager,
        feature_manager=bc.feature_manager,
        subproblem_manager=bc.subproblem_manager
    )
    return bc.ellipsoid_manager


def try_init_inequalities_manager(bc):
    """
    Initialize the InequalitiesSolver if required managers are set.
    
    Args:
        bc: BundleChoice instance
        
    Returns:
        InequalitiesSolver: The initialized inequalities solver instance
        
    Raises:
        RuntimeError: If required managers are not initialized
    """
    missing_managers = []
    if bc.data_manager is None:
        missing_managers.append("DataManager")
    if bc.feature_manager is None:
        missing_managers.append("FeatureManager")
    if bc.subproblem_manager is None:
        missing_managers.append("SubproblemManager")
    if bc.config is None or bc.config.dimensions is None:
        missing_managers.append("DimensionsConfig")
    if missing_managers:
        raise RuntimeError(
            "DataManager, FeatureManager, SubproblemManager, and DimensionsConfig must be set in config "
            "before initializing inequalities manager. Missing managers: "
            f"{', '.join(missing_managers)}"
        )

    bc.inequalities_manager = InequalitiesSolver(
        comm_manager=bc.comm_manager,
        dimensions_cfg=bc.config.dimensions,
        data_manager=bc.data_manager,
        feature_manager=bc.feature_manager,
        subproblem_manager=None
    )
    return bc.inequalities_manager

