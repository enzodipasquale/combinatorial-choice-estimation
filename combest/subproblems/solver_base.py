from abc import ABC, abstractmethod


class SubproblemSolver(ABC):

    def __init__(self, comm_manager, data_manager, features_manager, subproblem_cfg, dimensions_cfg):
        self.comm_manager = comm_manager
        self.data_manager = data_manager
        self.features_manager = features_manager
        self.subproblem_cfg = subproblem_cfg
        self.dimensions_cfg = dimensions_cfg

    def initialize(self):
        pass

    @abstractmethod
    def solve(self, theta):
        pass

    def update_solver_settings(self, settings_dict):
        pass
