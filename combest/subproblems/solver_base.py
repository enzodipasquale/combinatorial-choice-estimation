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


class GurobiMixin:

    def _create_gurobi_model(self):
        import gurobipy as gp
        from combest.utils import suppress_output
        with suppress_output():
            model = gp.Model()
            model.setParam('OutputFlag', 0)
            model.setParam('Threads', 1)
            for k, v in self.subproblem_cfg.gurobi_params.items():
                if v is not None:
                    model.setParam(k, v)
            model.setAttr('ModelSense', gp.GRB.MAXIMIZE)
        return model

    def update_solver_settings(self, settings_dict):
        for model in self.local_problems:
            for param, value in settings_dict.items():
                model.setParam(param, value)
