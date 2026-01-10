"""
Initialization methods for BundleChoice components.

This module contains all the _try_init_* methods that lazily initialize
the various managers and solvers used by BundleChoice.
"""

from typing import Any, Optional
from bundlechoice.data_manager import DataManager
from bundlechoice.feature_manager import FeatureManager
from bundlechoice.subproblems.subproblem_manager import SubproblemManager
from bundlechoice.estimation import ColumnGenerationManager, RowGenerationManager, StandardErrorsManager
from bundlechoice.estimation.ellipsoid import EllipsoidManager
from bundlechoice.errors import SetupError


def try_init_data_manager(bc: Any) -> DataManager:
    """Initialize DataManager if dimensions_cfg is set."""
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
    # Check required components (except features which can be auto-built)
    if bc.data_manager is None or bc.config is None or bc.config.subproblem is None:
        missing = []
        if bc.config is None or bc.config.subproblem is None:
            missing.append("config with 'subproblem' section")
        if bc.data_manager is None:
            missing.append("data")
        
        raise SetupError(
            f"Cannot initialize subproblem manager - missing: {', '.join(missing)}",
            suggestion=(
                "Complete these steps:\n"
                "  1. bc.load_config(config_dict)  # Include 'subproblem' section\n"
                "  2. bc.data.load_and_scatter(input_data)"
            )
        )

    # Auto-initialize feature_manager if not set
    if bc.feature_manager is None:
        try_init_feature_manager(bc)

    # Auto-build oracles if not already set
    if bc.feature_manager._features_oracle is None:
        bc.feature_manager.build_from_data()
    elif bc.feature_manager._error_oracle is None:
        # Features oracle was set manually but error oracle wasn't - build error oracle
        bc.feature_manager.build_error_oracle_from_data()

    bc.subproblem_manager = SubproblemManager(
        dimensions_cfg=bc.config.dimensions,
        comm_manager=bc.comm_manager,
        data_manager=bc.data_manager,
        feature_manager=bc.feature_manager,
        subproblem_cfg=bc.config.subproblem
    )
    # Note: load() is called by core.py's _try_init_subproblem_manager for consistency
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
        raise SetupError(
            "Cannot initialize row generation solver - missing setup",
            suggestion="\n  ".join(f"- {m}" for m in missing) + "\n\nRun bc.print_status() to see your current setup state.",
            missing=missing
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
        raise SetupError(
            "Cannot initialize ellipsoid solver - missing setup",
            suggestion="\n  ".join(f"- {m}" for m in missing) + "\n\nRun bc.print_status() to see your current setup state.",
            missing=missing
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
        raise SetupError(
            "Cannot initialize column generation solver - missing setup",
            suggestion="\n  ".join(f"- {m}" for m in missing) + "\n\nRun bc.print_status() to see your current setup state.",
            missing=missing
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


def try_init_standard_errors_manager(bc: Any) -> StandardErrorsManager:
    """Initialize StandardErrorsManager if not already present."""
    if bc.data_manager is None or bc.feature_manager is None or bc.subproblem_manager is None or bc.config is None:
        missing = []
        if bc.data_manager is None:
            missing.append("data (call bc.data.load_and_scatter(input_data))")
        if bc.feature_manager is None:
            missing.append("features (call bc.features.set_oracle(fn) or bc.features.build_from_data())")
        if bc.subproblem_manager is None:
            missing.append("subproblem (call bc.subproblems.load())")
        if bc.config is None:
            missing.append("config (call bc.load_config(config_dict))")
        raise SetupError(
            "Cannot initialize standard errors manager - missing setup",
            suggestion="\n  ".join(f"- {m}" for m in missing) + "\n\nRun bc.print_status() to see your current setup state.",
            missing=missing
        )

    bc.standard_errors_manager = StandardErrorsManager(
        comm_manager=bc.comm_manager,
        dimensions_cfg=bc.config.dimensions,
        data_manager=bc.data_manager,
        feature_manager=bc.feature_manager,
        subproblem_manager=bc.subproblem_manager,
        se_cfg=bc.config.standard_errors,
    )
    return bc.standard_errors_manager

