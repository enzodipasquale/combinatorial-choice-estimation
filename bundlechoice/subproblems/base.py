from abc import ABC, abstractmethod
from typing import Any, Optional
import numpy as np
from bundlechoice.base import HasDimensions, HasData

class BaseSubproblem(HasDimensions, HasData, ABC):
    def __init__(self, data_manager, feature_manager, subproblem_cfg, dimensions_cfg=None):
        self.data_manager = data_manager
        self.feature_manager = feature_manager
        self.subproblem_cfg = subproblem_cfg
        self.config = subproblem_cfg
        self.dimensions_cfg = dimensions_cfg

    def get_features(self, id, bundle, data_override=None):
        return self.feature_manager.get_features(id, bundle, data_override)

class BatchSubproblemBase(BaseSubproblem, ABC):
    """
    Base class for batch (vectorized) subproblems.
    """
    @abstractmethod
    def initialize(self) -> Any:
        pass

    @abstractmethod
    def solve(self, theta: Any) -> Any:
        pass

class SerialSubproblemBase(BaseSubproblem, ABC):
    """
    Base class for serial (per-agent) subproblems.
    """
    @abstractmethod
    def initialize(self, id: int) -> Any:
        pass

    @abstractmethod
    def solve(self, id: int, theta: Any, pb: Any = None) -> Any:
        pass
    
    def solve_all(self, theta: Any, problems: list[Any]) -> Any:
        return [self.solve(id, theta, pb) for id, pb in enumerate(problems)]
        