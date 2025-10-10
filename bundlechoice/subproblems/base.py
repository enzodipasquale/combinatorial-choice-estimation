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

    def features_oracle(self, agent_id, bundle, data_override=None):
        return self.feature_manager.features_oracle(agent_id, bundle, data_override)
    
    @abstractmethod
    def initialize_all(self) -> Any:
        """
        Initialize all subproblems for this rank. Implementation depends on whether this is batch or serial.
        
        Returns:
            None for batch subproblems, or list of subproblem instances for serial
        """
        pass
    
    @abstractmethod
    def solve_all(self, theta: Any, subproblems: Any = None) -> Any:
        """
        Solve all subproblems for this rank.
        
        Args:
            theta: Parameter vector for subproblem solving
            subproblems: Subproblem instances (only used by serial)
            
        Returns:
            Results from solving all subproblems
        """
        pass

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
    
    def initialize_all(self) -> Any:
        """Initialize batch subproblem (no agent-specific initialization needed)."""
        self.initialize()
        return None
    
    def solve_all(self, theta: Any, subproblems: Any = None) -> Any:
        """Solve batch subproblem."""
        return self.solve(theta)

class SerialSubproblemBase(BaseSubproblem, ABC):
    """
    Base class for serial (per-agent) subproblems.
    """
    @abstractmethod
    def initialize(self, agent_id: int) -> Any:
        pass

    @abstractmethod
    def solve(self, agent_id: int, theta: Any, pb: Any = None) -> Any:
        pass
    
    def solve_serial(self, theta: Any, problems: list[Any]) -> Any:
        return np.array([self.solve(agent_id, theta, pb) for agent_id, pb in enumerate(problems)])
    
    def initialize_all(self) -> Any:
        """Initialize serial subproblems for all agents on this rank."""
        return [self.initialize(agent_id) for agent_id in range(self.num_local_agents)]
    
    def solve_all(self, theta: Any, subproblems: Any = None) -> Any:
        """Solve serial subproblems using subproblems."""
        if subproblems is None:
            raise RuntimeError("subproblems is required for serial subproblems")
        return self.solve_serial(theta, subproblems)
        