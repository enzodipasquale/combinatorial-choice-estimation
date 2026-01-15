from abc import ABC, abstractmethod
from typing import Any, Optional, List
import numpy as np
from numpy.typing import NDArray

class BaseSubproblem(ABC):

    def __init__(self, data_manager, oracles_manager, subproblem_cfg, dimensions_cfg):
        self.data_manager = data_manager
        self.oracles_manager = oracles_manager
        self.subproblem_cfg = subproblem_cfg
        self.config = subproblem_cfg
        self.dimensions_cfg = dimensions_cfg

    def features_oracle(self, agent_id, bundle, data_override=None):
        return self.oracles_manager.features_oracle(agent_id, bundle, data_override)

    def error_oracle(self, agent_id, bundle, data_override=None):
        return self.oracles_manager.error_oracle(agent_id, bundle, data_override)

    @abstractmethod
    def initialize_all(self):
        pass

    @abstractmethod
    def solve_all(self, theta, subproblems=None):
        pass

class BatchSubproblemBase(BaseSubproblem, ABC):

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def solve(self, theta):
        pass

    def initialize_all(self):
        self.initialize()
        return None

    def solve_all(self, theta, subproblems=None):
        return self.solve(theta)

class SerialSubproblemBase(BaseSubproblem, ABC):

    @abstractmethod
    def initialize(self, agent_id):
        pass

    @abstractmethod
    def solve(self, agent_id, theta, problem_state=None):
        pass

    def solve_serial(self, theta, problems):
        return np.array([self.solve(id, theta, pb) for id, pb in enumerate(problems)])

    def initialize_all(self):
        return [self.initialize(id) for id in range(self.data_manager.num_local_agents)]

    def solve_all(self, theta, subproblems=None):
        if subproblems is None:
            raise RuntimeError('subproblems is required for serial subproblems')
        return self.solve_serial(theta, subproblems)