from typing import Any, Optional, List, TYPE_CHECKING
if TYPE_CHECKING:
    from bundlechoice.data_manager import DataManager
    from bundlechoice.oracles_manager import OraclesManager
    from bundlechoice.subproblems.subproblem_manager import SubproblemManager
    from bundlechoice.estimation import RowGenerationManager, StandardErrorsManager, ColumnGenerationManager
    from bundlechoice.estimation.ellipsoid import EllipsoidManager

def _check_requirements(bc, requirements, manager_name):
    checks = {'config': (bc.config, 'Call bc.load_config()'), 'dimensions': (bc.config and bc.config.dimensions, "Add 'dimensions' to config"), 'subproblem': (bc.config and bc.config.subproblem, "Add 'subproblem' to config"), 'row_generation': (bc.config and bc.config.row_generation, "Add 'row_generation' to config"), 'ellipsoid': (bc.config and bc.config.ellipsoid, "Add 'ellipsoid' to config"), 'data': (bc.data_manager and bc.data_manager.local_data, 'Call bc.data.load_input_data()'), 'features': (bc.oracles_manager and bc.oracles_manager._features_oracle, 'Call bc.oracles.build_quadratic_features_from_data()'), 'subproblems': (bc.subproblem_manager, 'Initialize subproblems')}
    missing = [(req, checks[req][1]) for req in requirements if not checks.get(req, (True,))[0]]
    if missing:
        msg_parts = [f"Cannot initialize {manager_name} - missing: {', '.join((r for r, _ in missing))}"]
        msg_parts.append('\n  '.join((f'- {r}: {h}' for r, h in missing)))
        raise RuntimeError('\n  '.join(msg_parts))

def try_init_data_manager(bc):
    from bundlechoice.data_manager import DataManager
    _check_requirements(bc, ['config', 'dimensions'], 'data manager')
    bc.data_manager = DataManager(bc.config.dimensions, bc.comm_manager)
    return bc.data_manager

def try_init_oracles_manager(bc):
    from bundlechoice.oracles_manager import OraclesManager
    _check_requirements(bc, ['config', 'dimensions'], 'oracles manager')
    bc.oracles_manager = OraclesManager(bc.config.dimensions, bc.comm_manager, bc.data_manager)
    return bc.oracles_manager

def try_init_subproblem_manager(bc):
    from bundlechoice.subproblems.subproblem_manager import SubproblemManager
    if bc.data_manager is None or bc.config is None or bc.config.subproblem is None:
        missing = []
        if bc.config is None or bc.config.subproblem is None:
            missing.append("config with 'subproblem' section")
        if bc.data_manager is None:
            missing.append('data')
        raise RuntimeError(f"Cannot initialize subproblem manager - missing: {', '.join(missing)}\nComplete these steps:\n  1. bc.load_config(config_dict)  # Include 'subproblem' section\n  2. bc.data.load_input_data(input_data)")
    if bc.oracles_manager is None:
        try_init_oracles_manager(bc)
    if bc.oracles_manager._features_oracle is None:
        bc.oracles_manager.build_quadratic_features_from_data()
    elif bc.oracles_manager._error_oracle is None:
        bc.oracles_manager.build_error_oracle_from_data()
    bc.subproblem_manager = SubproblemManager(bc.config.dimensions, bc.comm_manager, bc.data_manager, bc.oracles_manager, bc.config.subproblem)
    return bc.subproblem_manager

def try_init_row_generation_manager(bc):
    from bundlechoice.estimation import RowGenerationManager
    _check_requirements(bc, ['data', 'features', 'subproblems', 'row_generation'], 'row generation solver')
    bc.row_generation_manager = RowGenerationManager(bc.comm_manager, bc.config.dimensions, bc.config.row_generation, bc.data_manager, bc.oracles_manager, bc.subproblem_manager)
    return bc.row_generation_manager

def try_init_ellipsoid_manager(bc, theta_init=None):
    from bundlechoice.estimation.ellipsoid import EllipsoidManager
    _check_requirements(bc, ['data', 'features', 'subproblems', 'ellipsoid'], 'ellipsoid solver')
    bc.ellipsoid_manager = EllipsoidManager(bc.comm_manager, bc.config.dimensions, bc.config.ellipsoid, bc.data_manager, bc.oracles_manager, bc.subproblem_manager, theta_init)
    return bc.ellipsoid_manager

def try_init_column_generation_manager(bc, theta_init=None):
    from bundlechoice.estimation import ColumnGenerationManager
    _check_requirements(bc, ['data', 'features', 'subproblems', 'row_generation'], 'column generation solver')
    bc.column_generation_manager = ColumnGenerationManager(bc.comm_manager, bc.config.dimensions, bc.config.row_generation, bc.data_manager, bc.oracles_manager, bc.subproblem_manager, theta_init)
    return bc.column_generation_manager

def try_init_standard_errors_manager(bc):
    from bundlechoice.estimation import StandardErrorsManager
    _check_requirements(bc, ['config', 'data', 'features', 'subproblems'], 'standard errors manager')
    bc.standard_errors_manager = StandardErrorsManager(bc.comm_manager, bc.config.dimensions, bc.data_manager, bc.oracles_manager, bc.subproblem_manager, bc.config.standard_errors)
    return bc.standard_errors_manager