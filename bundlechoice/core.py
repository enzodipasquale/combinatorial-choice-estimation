from .config import DimensionsConfig, RowGenerationConfig, SubproblemConfig, BundleChoiceConfig
from .data_manager import DataManager
from .feature_manager import FeatureManager
from .subproblems import SUBPROBLEM_REGISTRY
from .subproblems.subproblem_manager import SubproblemManager, SubproblemProtocol
from mpi4py import MPI
from typing import Optional, Callable
from bundlechoice.utils import get_logger
from bundlechoice.estimation import RowGenerationSolver
from bundlechoice.base import HasDimensions, HasData, HasComm
logger = get_logger(__name__)

class BundleChoice(HasDimensions, HasComm, HasData):
    """
    Main orchestrator for modular bundle choice estimation.
    
    This class provides a clean API for distributed bundle choice estimation using MPI.
    It manages the lifecycle of data loading, feature extraction, subproblem solving,
    and parameter estimation. All distributed computation is handled transparently.
    
    The class follows a lazy initialization pattern where components are created
    only when needed, allowing for flexible configuration and setup.
    
    Typical usage:
        bc = BundleChoice()
        bc.load_config(cfg)
        bc.load_data(data, scatter=True)
        bc.build_feature_oracle_from_data()
        results = bc.init_and_solve_subproblems(theta)
        
    Attributes:
        config: Main configuration object
        dimensions_cfg: Problem dimensions configuration
        subproblem_cfg: Subproblem algorithm configuration
        row_generation_cfg: Row generation solver configuration
        data_manager: Data management component
        feature_manager: Feature extraction component
        subproblem_manager: Subproblem solving component
        row_generation_manager: Row generation estimation component
        comm: MPI communicator
    """
    config: Optional[BundleChoiceConfig]
    dimensions_cfg: Optional[DimensionsConfig]
    subproblem_cfg: Optional[SubproblemConfig]
    row_generation_cfg: Optional[RowGenerationConfig]
    data_manager: Optional[DataManager]
    feature_manager: Optional[FeatureManager]
    subproblem_manager: Optional[SubproblemManager]
    row_generation_manager: Optional[RowGenerationSolver]
    comm: MPI.Comm

    def __init__(self):
        """
        Initialize an empty BundleChoice instance.
        
        All configuration, data, and features must be loaded explicitly via the
        provided methods. The MPI communicator is automatically initialized to
        COMM_WORLD.
        """
        self.config = None
        self.dimensions_cfg = None
        self.subproblem_cfg = None
        self.row_generation_cfg = None
        self.comm = MPI.COMM_WORLD
        self.data_manager = None
        self.feature_manager = None
        self.subproblem_manager = None
        self.row_generation_manager = None

    # --- Initialization ---
    def _try_init_data_manager(self):
        """
        Initialize the DataManager if dimensions_cfg is set and not already initialized.
        
        This method creates a new DataManager instance using the current dimensions
        configuration and MPI communicator.
        
        Returns:
            DataManager: The initialized DataManager instance
            
        Raises:
            ValueError: If dimensions_cfg is not set
        """
        if self.dimensions_cfg is None:
            raise ValueError("dimensions_cfg must be set before initializing data manager.")
        self.data_manager = DataManager(
            dimensions_cfg=self.dimensions_cfg,
            comm=self.comm
        )
        return self.data_manager
        
    def _try_init_feature_manager(self):
        """
        Initialize the FeatureManager if dimensions_cfg is set.
        
        This method creates a new FeatureManager instance using the current
        dimensions configuration, MPI communicator, and data manager.
        
        Returns:
            FeatureManager: The initialized FeatureManager instance
            
        Raises:
            ValueError: If dimensions_cfg is not set
        """
        if self.dimensions_cfg is None:
            logger.error("dimensions_cfg must be set before initializing feature manager.")
        self.feature_manager = FeatureManager(
            dimensions_cfg=self.dimensions_cfg,
            comm=self.comm,
            data_manager=self.data_manager
        )
        return self.feature_manager
        
    def _try_init_subproblem_manager(self):
        """
        Initialize the subproblem manager if subproblem_cfg is set.
        
        This method creates a new SubproblemManager instance and loads the
        specified subproblem algorithm.
        
        Returns:
            SubproblemManager: The initialized SubproblemManager instance
            
        Raises:
            RuntimeError: If required managers or configs are not set
        """
        if self.data_manager is None or self.feature_manager is None or self.subproblem_cfg is None:
            raise RuntimeError("DataManager, FeatureManager, and SubproblemConfig must be set before initializing subproblem manager.")

        self.subproblem_manager = SubproblemManager(
            dimensions_cfg=self.dimensions_cfg,
            comm=self.comm,
            data_manager=self.data_manager,
            feature_manager=self.feature_manager,
            subproblem_cfg=self.subproblem_cfg
        )
        self.subproblem_manager.load()
        return self.subproblem_manager
    
    def _try_init_row_generation_manager(self):
        """
        Initialize the RowGenerationSolver if not already present.
        
        This method creates a new RowGenerationSolver instance using the current
        configuration and manager components.
        
        Returns:
            RowGenerationSolver: The initialized RowGenerationSolver instance
            
        Raises:
            RuntimeError: If required managers are not set
        """
        if self.data_manager is None or self.feature_manager is None or self.subproblem_manager is None:
            # raise error with missing managers
            missing_managers = []
            if self.data_manager is None:
                missing_managers.append("DataManager")
            if self.feature_manager is None:
                missing_managers.append("FeatureManager")
            if self.subproblem_manager is None:
                missing_managers.append("SubproblemManager")
            raise RuntimeError(
                "DataManager, FeatureManager, and SubproblemManager must be set "
                "before initializing row generation manager. Missing managers: "
                f"{', '.join(missing_managers)}"
            )

        self.row_generation_manager = RowGenerationSolver(
            comm=self.comm,
            dimensions_cfg=self.dimensions_cfg,
            rowgen_cfg=self.row_generation_cfg,
            data_manager=self.data_manager,
            feature_manager=self.feature_manager,
            subproblem_manager=self.subproblem_manager
        )
        return self.row_generation_manager

    # --- Properties ---
    @property
    def data(self):
        """
        Access the data manager component.
        
        Returns:
            DataManager: The data manager instance
        """
        if self.data_manager is None:
            self._try_init_data_manager()
        return self.data_manager

    @property
    def features(self):
        """
        Access the feature manager component.
        
        Returns:
            FeatureManager: The feature manager instance
        """
        if self.feature_manager is None:
            self._try_init_feature_manager()
        return self.feature_manager

    @property
    def subproblems(self):
        """
        Access the subproblem manager component.
        
        Returns:
            SubproblemManager: The subproblem manager instance
        """
        if self.subproblem_manager is None:
            self._try_init_subproblem_manager()
        return self.subproblem_manager

    @property
    def row_generation(self):
        """
        Access the row generation manager component.
        
        Returns:
            RowGenerationSolver: The row generation solver instance
        """
        if self.row_generation_manager is None:
            self._try_init_row_generation_manager()
        return self.row_generation_manager
        
    def load_config(self, cfg):
        """
        Load configuration from a dictionary or YAML file.
        
        This method loads the complete configuration and resets all manager
        components to ensure consistency with the new configuration.
        
        Args:
            cfg: Dictionary or YAML file path with configuration
            
        Returns:
            BundleChoice: self for method chaining
            
        Raises:
            ValueError: If cfg is not a dictionary or string
        """
        if isinstance(cfg, str):
            self.config = BundleChoiceConfig.from_yaml(cfg)
        elif isinstance(cfg, dict):
            self.config = BundleChoiceConfig.from_dict(cfg)
        else:
            raise ValueError("cfg must be a dictionary or a YAML path.")
        self.dimensions_cfg = self.config.dimensions
        self.row_generation_cfg = self.config.row_generation
        self.subproblem_cfg = self.config.subproblem

        self.data_manager = None
        self.feature_manager = None
        self.subproblem_manager = None
        self.row_generation_manager = None

        return self
