from abc import ABC, abstractmethod


class SubproblemSolver(ABC):

    def __init__(self, data_manager, oracles_manager, subproblem_cfg, dimensions_cfg):
        self.data_manager = data_manager
        self.oracles_manager = oracles_manager
        self.subproblem_cfg = subproblem_cfg
        self.dimensions_cfg = dimensions_cfg

    def initialize(self):
        pass

    @abstractmethod
    def solve(self, theta):
        pass

    def update_solver_settings(self, settings_dict):
        pass
