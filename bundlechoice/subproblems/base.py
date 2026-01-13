"""
Base classes for subproblem solvers.

Supports both batch (vectorized) and serial (per-agent) implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, List
import numpy as np
from numpy.typing import NDArray
from bundlechoice.base import HasDimensions, HasData


class BaseSubproblem(HasDimensions, HasData, ABC):
    """Base class for all subproblem solvers."""
    
    def __init__(self, data_manager: Any, oracles_manager: Any, 
                 subproblem_cfg: Any, dimensions_cfg: Optional[Any] = None) -> None:
        self.data_manager = data_manager
        self.oracles_manager = oracles_manager
        self.subproblem_cfg = subproblem_cfg
        self.config = subproblem_cfg
        self.dimensions_cfg = dimensions_cfg

    def features_oracle(self, agent_id: int, bundle: NDArray[np.float64], 
                       data_override: Optional[Any] = None) -> NDArray[np.float64]:
        """Compute features for agent/bundle."""
        return self.oracles_manager.features_oracle(agent_id, bundle, data_override)
    
    def error_oracle(self, agent_id: int, bundle: NDArray[np.float64],
                    data_override: Optional[Any] = None) -> float:
        """Compute error for agent/bundle."""
        return self.oracles_manager.error_oracle(agent_id, bundle, data_override)
    
    @abstractmethod
    def initialize_all(self) -> Any:
        """Initialize all subproblems for this rank. Returns None (batch) or list (serial)."""
        pass
    
    @abstractmethod
    def solve_all(self, theta: NDArray[np.float64], subproblems: Optional[Any] = None) -> NDArray[np.float64]:
        """Solve all subproblems for this rank."""
        pass

class BatchSubproblemBase(BaseSubproblem, ABC):
    """Base class for batch (vectorized) subproblems."""
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize batch subproblem (no agent-specific state needed)."""
        pass

    @abstractmethod
    def solve(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Solve batch subproblem for all agents on this rank."""
        pass
    
    def initialize_all(self) -> None:
        """Initialize batch subproblem."""
        self.initialize()
        return None
    
    def solve_all(self, theta: NDArray[np.float64], subproblems: Optional[Any] = None) -> NDArray[np.float64]:
        """Solve batch subproblem."""
        return self.solve(theta)

class SerialSubproblemBase(BaseSubproblem, ABC):
    """Base class for serial (per-agent) subproblems."""
    
    @abstractmethod
    def initialize(self, agent_id: int) -> Any:
        """Initialize subproblem for one agent. Returns problem state."""
        pass

    @abstractmethod
    def solve(self, agent_id: int, theta: NDArray[np.float64], problem_state: Optional[Any] = None) -> NDArray[np.float64]:
        """Solve subproblem for one agent. Returns bundle."""
        pass
    
    def solve_serial(self, theta: NDArray[np.float64], problems: List[Any]) -> NDArray[np.float64]:
        """Solve all serial subproblems sequentially."""
        return np.array([self.solve(id, theta, pb) for id, pb in enumerate(problems)])
    
    def initialize_all(self) -> List[Any]:
        """Initialize serial subproblems for all agents on this rank."""
        return [self.initialize(id) for id in range(self.num_local_agents)]
    
    def solve_all(self, theta: NDArray[np.float64], subproblems: Optional[List[Any]] = None) -> NDArray[np.float64]:
        """Solve serial subproblems using problem states."""
        if subproblems is None:
            raise RuntimeError("subproblems is required for serial subproblems")
        return self.solve_serial(theta, subproblems)