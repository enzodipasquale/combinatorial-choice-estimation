import mpi4py.MPI as MPI

class HasComm:
    """
    Mixin for classes that provide an MPI communicator.
    Expects the subclass to define self.comm (of type MPI.Comm).
    Provides convenient properties for rank and comm_size.
    """
    comm: 'MPI.Comm'  # type: ignore

    @property
    def rank(self):
        """MPI rank of this process."""
        return self.comm.Get_rank() if self.comm is not None else None

    @property
    def comm_size(self):
        """Total number of MPI processes."""
        return self.comm.Get_size() if self.comm is not None else None

class HasDimensions:
    """
    Mixin for classes that provide a dimensions_cfg attribute.
    Expects the subclass to define self.dimensions_cfg (of type DimensionsConfig).
    Provides convenient properties for num_agents, num_items, num_features, and num_simuls.
    """
    dimensions_cfg: 'DimensionsConfig'  # type: ignore

    @property
    def num_agents(self):
        """Number of agents (from dimensions_cfg)."""
        return self.dimensions_cfg.num_agents if self.dimensions_cfg else None

    @property
    def num_items(self):
        """Number of items (from dimensions_cfg)."""
        return self.dimensions_cfg.num_items if self.dimensions_cfg else None

    @property
    def num_features(self):
        """Number of features (from dimensions_cfg)."""
        return self.dimensions_cfg.num_features if self.dimensions_cfg else None

    @property
    def num_simuls(self):
        """Number of simulations (from dimensions_cfg)."""
        return self.dimensions_cfg.num_simuls if self.dimensions_cfg else None

class HasData:
    """
    Mixin for classes that provide a data_manager attribute.
    Expects the subclass to define self.data_manager (of type DataManager).
    Provides convenient properties for input_data, local_data, and num_local_agents.
    """
    data_manager: 'DataManager'  # type: ignore

    @property
    def input_data(self):
        """Input data dictionary (from data_manager)."""
        return self.data_manager.input_data

    @property
    def local_data(self):
        """Local data dictionary for this rank (from data_manager)."""
        return self.data_manager.local_data

    @property
    def num_local_agents(self):
        """Number of local agents for this rank (from data_manager)."""
        return self.data_manager.num_local_agents
