from abc import ABC, abstractmethod
from typing import Any, Optional
import numpy as np

class BaseSubproblemABC(ABC):
    """
    Abstract base class for all subproblems. Defines the required interface.
    """
    solve_all_local_problems: bool

    @abstractmethod
    def initialize(self, *args, **kwargs) -> Any:
        """Prepare the subproblem for solving."""
        pass

    @abstractmethod
    def solve(self, *args, **kwargs) -> Any:
        """Solve the subproblem."""
        pass

class BaseSubproblem:
    def __init__(self, data_manager, feature_manager, subproblem_cfg):
        self.data_manager = data_manager
        self.feature_manager = feature_manager
        self.subproblem_cfg = subproblem_cfg
        self.config = subproblem_cfg  # Fix: use subproblem_cfg, not data_manager.config

    @property
    def num_items(self):
        return self.data_manager.num_items

    @property
    def num_agents(self):
        return self.data_manager.num_agents

    @property
    def num_simuls(self):
        return self.data_manager.num_simuls

    @property
    def num_features(self):
        return self.data_manager.num_features

    def get_features(self, local_id, bundle, data_override=None):
        return self.feature_manager.get_features(local_id, bundle, data_override)

    @property
    def local_data(self):
        return self.data_manager.local_data

    @property
    def num_local_agents(self):
        return self.data_manager.num_local_agents

class BatchSubproblemBase(BaseSubproblemABC, BaseSubproblem):
    """
    Base class for batch (vectorized) subproblems.
    """
    
    @abstractmethod
    def initialize(self) -> Any:
        pass

    @abstractmethod
    def solve(self, lambda_k: Any) -> Any:
        pass

class SerialSubproblemBase(BaseSubproblemABC, BaseSubproblem):
    """
    Base class for serial (per-agent) subproblems.
    """
    
    @abstractmethod
    def initialize(self, local_id: int) -> Any:
        pass

    @abstractmethod
    def solve(self, local_id: int, lambda_k: Any, pb: Any = None) -> Any:
        pass