"""
Base mixin classes for BundleChoice components.

Provides property accessors for shared dependencies.
"""

from typing import TYPE_CHECKING, Optional, Dict, Any

if TYPE_CHECKING:
    from mpi4py import MPI
    from .comm_manager import CommManager
    from .config import DimensionsConfig, BundleChoiceConfig
    from .data_manager import DataManager


class HasComm:
    """Mixin for MPI communicator access."""
    comm_manager: 'CommManager'

    @property
    def comm(self) -> 'MPI.Comm':
        return self.comm_manager.comm

    @property
    def rank(self) -> int:
        return self.comm_manager.rank

    @property
    def comm_size(self) -> int:
        return self.comm_manager.size
    
    def is_root(self) -> bool:
        return self.comm_manager.is_root()


class HasDimensions:
    """Mixin for problem dimensions access."""
    dimensions_cfg: 'DimensionsConfig'

    @property
    def num_agents(self) -> int:
        return self.dimensions_cfg.num_agents if self.dimensions_cfg else 0

    @property
    def num_items(self) -> int:
        return self.dimensions_cfg.num_items if self.dimensions_cfg else 0

    @property
    def num_features(self) -> int:
        return self.dimensions_cfg.num_features if self.dimensions_cfg else 0

    @property
    def num_simulations(self) -> int:
        return self.dimensions_cfg.num_simulations if self.dimensions_cfg else 1


class HasConfig:
    """Mixin for configuration access."""
    config: 'BundleChoiceConfig'

    @property
    def subproblem_cfg(self):
        return self.config.subproblem if self.config else None

    @property
    def row_generation_cfg(self):
        return self.config.row_generation if self.config else None

    @property
    def ellipsoid_cfg(self):
        return self.config.ellipsoid if self.config else None


class HasData:
    """Mixin for data access."""
    data_manager: 'DataManager'

    @property
    def input_data(self) -> Optional[Dict[str, Any]]:
        return self.data_manager.input_data if self.data_manager else None

    @property
    def local_data(self) -> Optional[Dict[str, Any]]:
        return self.data_manager.local_data if self.data_manager else None

    @property
    def num_local_agents(self) -> int:
        return self.data_manager.num_local_agents if self.data_manager else 0
