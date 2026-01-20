from .subproblem_registry import SUBPROBLEM_REGISTRY
import numpy as np
from bundlechoice.utils import get_logger

logger = get_logger(__name__)

class SubproblemManager:

    def __init__(self, comm_manager, config, data_manager, oracles_manager):
        self.config = config
        self.comm_manager = comm_manager
        self.data_manager = data_manager
        self.oracles_manager = oracles_manager
        self.subproblem = None

    def load_subproblem(self, subproblem=None):
        subproblem = subproblem or self.config.subproblem.name
        
        if isinstance(subproblem, type):
            cls = subproblem
        elif callable(subproblem) and not isinstance(subproblem, str):
            self.subproblem = subproblem(self.data_manager, 
                                         self.oracles_manager, 
                                         self.config.subproblem, 
                                         self.config.dimensions)
            return self.subproblem
        else:
            cls = SUBPROBLEM_REGISTRY.get(subproblem)
            if cls is None:
                raise ValueError(f"Unknown subproblem: '{subproblem}'. "
                               f"Available: {', '.join(SUBPROBLEM_REGISTRY.keys())}")
        
        self.subproblem = cls(self.data_manager, 
                             self.oracles_manager, 
                             self.config.subproblem, 
                             self.config.dimensions)
        
        return self.subproblem

    def initialize_subproblems(self):
        if self.subproblem is None:
            self.load_subproblem()
        self.subproblem.initialize()

    def initialize_and_solve_subproblems(self, theta):
        theta = self.comm_manager.Bcast(theta)
        self.initialize_subproblems()
        local_bundles = self.subproblem.solve(theta)
        return local_bundles

    def solve_subproblems(self, theta):
        if self.subproblem is None:
            raise ValueError("Subproblem not initialized")
        return self.subproblem.solve(theta)
    
    def initialize_and_solve_subproblems(self, theta):
        theta = self.comm_manager.Bcast(theta)
        self.initialize_subproblems()
        local_bundles = self.subproblem.solve(theta)
        return local_bundles

    def generate_obs_bundles(self, theta):
        local_bundles = self.initialize_and_solve_subproblems(theta)
        self.data_manager.local_data["id_data"]["obs_bundles"] = local_bundles.astype(bool)
        obs_bundles = self.comm_manager.Gatherv_by_row(local_bundles, row_counts=self.data_manager.agent_counts)
        return obs_bundles


    def update_gurobi_settings(self, settings_dict):
        self.config.subproblem.settings.update(settings_dict)
        
        if self.subproblem is None:
            return
        
        if hasattr(self.subproblem, 'local_pbs') and self.subproblem.local_pbs is not None:
            import gurobipy as gp
            for model in self.subproblem.local_pbs:
                if model is not None:
                    for param, value in settings_dict.items():
                        model.setParam(param, value)