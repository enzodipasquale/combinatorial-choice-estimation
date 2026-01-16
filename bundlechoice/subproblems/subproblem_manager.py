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

    def load(self, subproblem=None):
        subproblem = subproblem or self.config.subproblem.name
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
            self.load()
        self.subproblem.initialize()

    def initialize_and_solve_subproblems(self, theta):
        theta = self.comm_manager.Bcast(theta)
        self.initialize_subproblems()
        local_bundles = self.subproblem.solve(theta)
        return local_bundles

    def solve_subproblems(self, theta):
        return self.subproblem.solve(theta)
   