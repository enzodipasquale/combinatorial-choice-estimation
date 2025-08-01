

class HasComm:
    """
    Mixin for classes that provide an MPI communicator.
    
    This mixin provides convenient access to MPI rank and communicator size.
    Classes using this mixin must define a `comm_manager` attribute of type `CommManager`.
    
    Attributes:
        comm_manager: Communication manager instance
    """
    comm_manager: 'CommManager'  # type: ignore

    @property
    def rank(self):
        """MPI rank of this process."""
        return self.comm_manager.rank if self.comm_manager is not None else None

    @property
    def comm_size(self):
        """Total number of MPI processes in the communicator."""
        return self.comm_manager.size if self.comm_manager is not None else None
    
    def is_root(self) -> bool:
        """Check if the current rank is the root rank (rank 0)."""
        return self.comm_manager.is_root() if self.comm_manager is not None else False

class HasDimensions:
    """
    Mixin for classes that provide access to problem dimensions.
    
    This mixin provides convenient properties for accessing problem dimensions
    from a `dimensions_cfg` attribute. Classes using this mixin must define
    a `dimensions_cfg` attribute of type `DimensionsConfig`.
    
    Use this mixin for classes that only have access to dimensions configuration
    but not the full BundleChoiceConfig. For classes that have access to the
    full config object, use HasConfig instead.
    
    Attributes:
        dimensions_cfg: Configuration object containing problem dimensions
    """
    dimensions_cfg: 'DimensionsConfig'  # type: ignore

    @property
    def num_agents(self):
        """Number of agents in the problem."""
        return self.dimensions_cfg.num_agents if self.dimensions_cfg else None

    @property
    def num_items(self):
        """Number of items available for choice."""
        return self.dimensions_cfg.num_items if self.dimensions_cfg else None

    @property
    def num_features(self):
        """Number of features per agent-item combination."""
        return self.dimensions_cfg.num_features if self.dimensions_cfg else None

    @property
    def num_simuls(self):
        """Number of simulation runs."""
        return self.dimensions_cfg.num_simuls if self.dimensions_cfg else None

class HasConfig:
    """
    Mixin for classes that provide access to configuration components.
    
    This mixin provides convenient properties for accessing configuration
    components from a `config` attribute. Classes using this mixin must define
    a `config` attribute of type `BundleChoiceConfig`.
    
    Note: This mixin does NOT provide dimension properties (num_agents, etc.).
    Use HasDimensions for dimension access, or implement both mixins together.
    
    Attributes:
        config: Main configuration object containing all components
    """
    config: 'BundleChoiceConfig'  # type: ignore

    @property
    def subproblem_cfg(self):
        """Subproblem algorithm configuration."""
        return self.config.subproblem if self.config else None

    @property
    def row_generation_cfg(self):
        """Row generation solver configuration."""
        return self.config.row_generation if self.config else None

    @property
    def ellipsoid_cfg(self):
        """Ellipsoid method solver configuration."""
        return self.config.ellipsoid if self.config else None

class HasData:
    """
    Mixin for classes that provide access to data management.
    
    This mixin provides convenient properties for accessing data from a
    `data_manager` attribute. Classes using this mixin must define a
    `data_manager` attribute of type `DataManager`.
    
    Attributes:
        data_manager: Data manager instance
    """
    data_manager: 'DataManager'  # type: ignore

    @property
    def input_data(self):
        """Input data dictionary containing all problem data."""
        return self.data_manager.input_data

    @property
    def local_data(self):
        """Local data dictionary for this MPI rank."""
        return self.data_manager.local_data

    @property
    def num_local_agents(self):
        """Number of agents assigned to this MPI rank."""
        return self.data_manager.num_local_agents
