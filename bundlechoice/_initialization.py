"""
Initialization methods for BundleChoice components.

This module contains all the _try_init_* methods that lazily initialize
the various managers and solvers used by BundleChoice.
"""

from typing import Any, Optional
from bundlechoice.data_manager import DataManager
from bundlechoice.feature_manager import FeatureManager
from bundlechoice.subproblems.subproblem_manager import SubproblemManager
from bundlechoice.estimation import ColumnGenerationManager, RowGenerationManager
from bundlechoice.estimation.ellipsoid import EllipsoidManager
from bundlechoice.estimation.inequalities import InequalitiesManager


# ============================================================================
# Initialization Functions
# ============================================================================

def try_init_data_manager(bc: Any) -> DataManager:
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


def try_init_feature_manager(bc: Any) -> FeatureManager:
    """Initialize FeatureManager if dimensions_cfg is set."""
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


def try_init_subproblem_manager(bc: Any) -> SubproblemManager:
    """Initialize SubproblemManager if subproblem_cfg is set."""
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
    # Don't auto-load here - will auto-load on first use (init_and_solve)
    return bc.subproblem_manager


def try_init_row_generation_manager(bc: Any) -> RowGenerationManager:
    """Initialize RowGenerationManager if not already present."""
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
            "\n\nRun bc.print_status() to see your current setup state."
        )

    bc.row_generation_manager = RowGenerationManager(
        comm_manager=bc.comm_manager,
        dimensions_cfg=bc.config.dimensions,
        row_generation_cfg=bc.config.row_generation,
        data_manager=bc.data_manager,
        feature_manager=bc.feature_manager,
        subproblem_manager=bc.subproblem_manager
    )
    return bc.row_generation_manager


def try_init_ellipsoid_manager(bc: Any, theta_init: Optional[Any] = None) -> EllipsoidManager:
    """Initialize EllipsoidManager if not already present."""
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
            "\n\nRun bc.print_status() to see your current setup state."
        )

    bc.ellipsoid_manager = EllipsoidManager(
        comm_manager=bc.comm_manager,
        dimensions_cfg=bc.config.dimensions,
        ellipsoid_cfg=bc.config.ellipsoid,
        data_manager=bc.data_manager,
        feature_manager=bc.feature_manager,
        subproblem_manager=bc.subproblem_manager,
        theta_init=theta_init
    )
    return bc.ellipsoid_manager


def try_init_inequalities_manager(bc: Any) -> InequalitiesManager:
    """Initialize InequalitiesManager if required managers are set."""
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

    bc.inequalities_manager = InequalitiesManager(
        comm_manager=bc.comm_manager,
        dimensions_cfg=bc.config.dimensions,
        data_manager=bc.data_manager,
        feature_manager=bc.feature_manager,
        subproblem_manager=None
    )
    return bc.inequalities_manager


def try_init_column_generation_manager(bc: Any, theta_init: Optional[Any] = None) -> ColumnGenerationManager:
    """Initialize ColumnGenerationManager if not already present."""
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
            "Cannot initialize column generation solver - missing setup:\n  "
            + "\n  ".join(missing)
            + "\n\nRun bc.print_status() to see your current setup state."
        )

    bc.column_generation_manager = ColumnGenerationManager(
        comm_manager=bc.comm_manager,
        dimensions_cfg=bc.config.dimensions,
        row_generation_cfg=bc.config.row_generation,
        data_manager=bc.data_manager,
        feature_manager=bc.feature_manager,
        subproblem_manager=bc.subproblem_manager,
        theta_init=theta_init,
    )
    return bc.column_generation_manager

