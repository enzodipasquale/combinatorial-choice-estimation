from .solver_registry import SOLVER_REGISTRY
import numpy as np
from combest.utils import get_logger

logger = get_logger(__name__)


def _empty_bundles(dimensions_cfg):
    return np.zeros((0, dimensions_cfg.n_items), dtype=bool)

class SubproblemManager:

    def __init__(self, comm_manager, config, data_manager, features_manager):
        self.config = config
        self.comm_manager = comm_manager
        self.data_manager = data_manager
        self.features_manager = features_manager
        self.subproblem_solver = None
        self._solver_is_initialized = False

    def load_solver(self, solver_spec=None):
        solver_spec = solver_spec or self.config.subproblem.name

        if isinstance(solver_spec, type):
            cls = solver_spec
        elif callable(solver_spec) and not isinstance(solver_spec, str):
            self.subproblem_solver = solver_spec(self.comm_manager,
                                 self.data_manager,
                                 self.features_manager,
                                 self.config.subproblem,
                                 self.config.dimensions)
            return self.subproblem_solver
        else:
            cls = SOLVER_REGISTRY.get(solver_spec)
            if cls is None:
                raise ValueError(f"Unknown solver: '{solver_spec}'. "
                               f"Available: {', '.join(SOLVER_REGISTRY.keys())}")

        self.subproblem_solver = cls(self.comm_manager,
                         self.data_manager,
                         self.features_manager,
                         self.config.subproblem,
                         self.config.dimensions)

        return self.subproblem_solver

    def initialize_solver(self):
        if self.subproblem_solver is None:
            self.load_solver()
        self.subproblem_solver.initialize()
        self._solver_is_initialized = True

    def initialize_and_solve(self, theta):
        theta = self.comm_manager.Bcast(theta)
        self.initialize_solver()
        if self.comm_manager.num_local_agent == 0:
            return _empty_bundles(self.config.dimensions)
        local_bundles = self.subproblem_solver.solve(theta)
        return local_bundles

    def solve(self, theta):
        if self.subproblem_solver is None:
            raise ValueError("Solver not initialized")
        if self.comm_manager.num_local_agent == 0:
            return _empty_bundles(self.config.dimensions)
        return self.subproblem_solver.solve(theta)

    def generate_obs_bundles(self, theta):
        local_bundles = self.initialize_and_solve(theta)
        self.data_manager.local_data.id_data["obs_bundles"] = local_bundles
        obs_bundles = self.comm_manager.Gatherv_by_row(local_bundles, row_counts=self.comm_manager.agent_counts)
        return obs_bundles

    def update_gurobi_settings(self, settings_dict):
        self.config.subproblem.gurobi_params.update(settings_dict)
        if self.subproblem_solver is not None:
            self.subproblem_solver.update_solver_settings(settings_dict)
