"""
Initialization methods for BundleChoice components.

Provides lazy initialization of managers with clear error messages.
"""

from typing import Any, Optional, List, TYPE_CHECKING
from bundlechoice.errors import SetupError

if TYPE_CHECKING:
    from bundlechoice.data_manager import DataManager
    from bundlechoice.feature_manager import FeatureManager
    from bundlechoice.subproblems.subproblem_manager import SubproblemManager
    from bundlechoice.estimation import RowGenerationManager, StandardErrorsManager, ColumnGenerationManager
    from bundlechoice.estimation.ellipsoid import EllipsoidManager


def _check_requirements(bc: Any, requirements: List[str], manager_name: str) -> None:
    """Check that required components are initialized."""
    missing = []
    hints = {
        "config": ("config", "Call bc.load_config(config_dict)"),
        "dimensions": ("config.dimensions", "Add 'dimensions' section to config"),
        "subproblem": ("config.subproblem", "Add 'subproblem' section to config"),
        "row_generation": ("config.row_generation", "Add 'row_generation' to your config"),
        "ellipsoid": ("config.ellipsoid", "Add 'ellipsoid' to your config"),
        "data": ("data_manager", "Call bc.data.load_and_scatter(input_data)"),
        "features": ("feature_manager", "Call bc.features.build_from_data() or set_oracle()"),
        "subproblems": ("subproblem_manager", "Call bc.subproblems.load()"),
    }
    
    for req in requirements:
        attr, _ = hints.get(req, (req, ""))
        parts = attr.split(".")
        obj = bc
        for part in parts:
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is None:
            _, hint = hints.get(req, (req, f"Initialize {req}"))
            missing.append(f"{req} ({hint})")
    
    if missing:
        raise SetupError(
            f"Cannot initialize {manager_name} - missing: {', '.join(r.split(' (')[0] for r in missing)}",
            suggestion="\n  ".join(f"- {m}" for m in missing) + "\n\nRun bc.print_status() to see current setup.",
            missing=[m.split(" (")[0] for m in missing],
        )


def try_init_data_manager(bc: Any) -> 'DataManager':
    """Initialize DataManager."""
    from bundlechoice.data_manager import DataManager
    _check_requirements(bc, ["config", "dimensions"], "data manager")
    bc.data_manager = DataManager(bc.config.dimensions, bc.comm_manager)
    return bc.data_manager


def try_init_feature_manager(bc: Any) -> 'FeatureManager':
    """Initialize FeatureManager."""
    from bundlechoice.feature_manager import FeatureManager
    _check_requirements(bc, ["config", "dimensions"], "feature manager")
    bc.feature_manager = FeatureManager(bc.config.dimensions, bc.comm_manager, bc.data_manager)
    return bc.feature_manager


def try_init_subproblem_manager(bc: Any) -> 'SubproblemManager':
    """Initialize SubproblemManager."""
    from bundlechoice.subproblems.subproblem_manager import SubproblemManager
    _check_requirements(bc, ["config", "subproblem", "data", "features"], "subproblem manager")
    bc.subproblem_manager = SubproblemManager(
        bc.config.dimensions, bc.comm_manager, bc.data_manager, bc.feature_manager, bc.config.subproblem
    )
    return bc.subproblem_manager


def try_init_row_generation_manager(bc: Any) -> 'RowGenerationManager':
    """Initialize RowGenerationManager."""
    from bundlechoice.estimation import RowGenerationManager
    _check_requirements(bc, ["data", "features", "subproblems", "row_generation"], "row generation solver")
    bc.row_generation_manager = RowGenerationManager(
        bc.comm_manager, bc.config.dimensions, bc.config.row_generation,
        bc.data_manager, bc.feature_manager, bc.subproblem_manager
    )
    return bc.row_generation_manager


def try_init_ellipsoid_manager(bc: Any, theta_init: Optional[Any] = None) -> 'EllipsoidManager':
    """Initialize EllipsoidManager."""
    from bundlechoice.estimation.ellipsoid import EllipsoidManager
    _check_requirements(bc, ["data", "features", "subproblems", "ellipsoid"], "ellipsoid solver")
    bc.ellipsoid_manager = EllipsoidManager(
        bc.comm_manager, bc.config.dimensions, bc.config.ellipsoid,
        bc.data_manager, bc.feature_manager, bc.subproblem_manager, theta_init
    )
    return bc.ellipsoid_manager


def try_init_column_generation_manager(bc: Any, theta_init: Optional[Any] = None) -> 'ColumnGenerationManager':
    """Initialize ColumnGenerationManager."""
    from bundlechoice.estimation import ColumnGenerationManager
    _check_requirements(bc, ["data", "features", "subproblems", "row_generation"], "column generation solver")
    bc.column_generation_manager = ColumnGenerationManager(
        bc.comm_manager, bc.config.dimensions, bc.config.row_generation,
        bc.data_manager, bc.feature_manager, bc.subproblem_manager, theta_init
    )
    return bc.column_generation_manager


def try_init_standard_errors_manager(bc: Any) -> 'StandardErrorsManager':
    """Initialize StandardErrorsManager."""
    from bundlechoice.estimation import StandardErrorsManager
    _check_requirements(bc, ["config", "data", "features", "subproblems"], "standard errors manager")
    bc.standard_errors_manager = StandardErrorsManager(
        bc.comm_manager, bc.config.dimensions, bc.data_manager,
        bc.feature_manager, bc.subproblem_manager, bc.config.standard_errors
    )
    return bc.standard_errors_manager
