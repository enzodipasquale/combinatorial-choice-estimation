from typing import Optional, Dict, Any

# ============================================================================
# Mixin Classes
# ============================================================================

class HasComm:
    """Mixin for MPI communicator access (rank, size, is_root, comm)."""
    comm_manager: 'CommManager'  # type: ignore

    @property
    def comm(self) -> Optional['MPI.Comm']:  # type: ignore
        """MPI communicator object."""
        return self.comm_manager.comm if self.comm_manager is not None else None

    @property
    def rank(self) -> Optional[int]:
        """MPI rank of this process."""
        return self.comm_manager.rank if self.comm_manager is not None else None

    @property
    def comm_size(self) -> Optional[int]:
        """Total number of MPI processes."""
        return self.comm_manager.size if self.comm_manager is not None else None
    
    def is_root(self) -> bool:
        """Check if current rank is root (rank 0)."""
        return self.comm_manager.is_root() if self.comm_manager is not None else False

class HasDimensions:
    """Mixin for problem dimensions access (num_agents, num_items, etc.)."""
    dimensions_cfg: 'DimensionsConfig'  # type: ignore

    @property
    def num_agents(self) -> Optional[int]:
        """Number of agents in the problem."""
        return self.dimensions_cfg.num_agents if self.dimensions_cfg else None

    @property
    def num_items(self) -> Optional[int]:
        """Number of items available for choice."""
        return self.dimensions_cfg.num_items if self.dimensions_cfg else None

    @property
    def num_features(self) -> Optional[int]:
        """Number of features per agent-item combination."""
        return self.dimensions_cfg.num_features if self.dimensions_cfg else None

    @property
    def num_simulations(self) -> int:
        """Number of simulation runs."""
        return self.dimensions_cfg.num_simulations if self.dimensions_cfg else 1

class HasConfig:
    """Mixin for configuration access (subproblem_cfg, row_generation_cfg, etc.)."""
    config: 'BundleChoiceConfig'  # type: ignore

    @property
    def subproblem_cfg(self) -> Optional['SubproblemConfig']:
        """Subproblem algorithm configuration."""
        return self.config.subproblem if self.config else None

    @property
    def row_generation_cfg(self) -> Optional['RowGenerationConfig']:
        """Row generation solver configuration."""
        return self.config.row_generation if self.config else None

    @property
    def ellipsoid_cfg(self) -> Optional['EllipsoidConfig']:
        """Ellipsoid method solver configuration."""
        return self.config.ellipsoid if self.config else None

class HasData:
    """Mixin for data access (input_data, local_data, num_local_agents)."""
    data_manager: 'DataManager'  # type: ignore

    @property
    def input_data(self) -> Optional[Any]:
        """Input data dictionary containing all problem data."""
        return self.data_manager.input_data if self.data_manager is not None else None

    @property
    def local_data(self) -> Optional[Any]:
        """Local data dictionary for this MPI rank."""
        return self.data_manager.local_data if self.data_manager is not None else None

    @property
    def num_local_agents(self) -> Optional[int]:
        """Number of agents assigned to this MPI rank."""
        return self.data_manager.num_local_agents if self.data_manager is not None else None
