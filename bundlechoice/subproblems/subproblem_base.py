from abc import ABC, abstractmethod
import numpy as np

class BaseSubproblem(ABC):

    def __init__(self, data_manager, oracles_manager, subproblem_cfg, dimensions_cfg):
        self.data_manager = data_manager
        self.oracles_manager = oracles_manager
        self.subproblem_cfg = subproblem_cfg
        self.dimensions_cfg = dimensions_cfg


class BatchSubproblemBase(BaseSubproblem, ABC):

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def solve(self, theta):
        pass

class SerialSubproblemBase(BaseSubproblem, ABC):

    @abstractmethod
    def initialize_single_pb(self, agent_id):
        pass

    @abstractmethod
    def solve_single_pb(self, agent_id, theta, pb=None):
        pass

    def initialize(self):
        self.local_pbs = [self.initialize_single_pb(i) for i in range(self.data_manager.num_local_agent)]
        return self.local_pbs

    def solve(self, theta):
        return np.array([self.solve_single_pb(i, theta, pb) for i, pb in enumerate(self.local_pbs)])
