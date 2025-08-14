from .config import DimensionsConfig, RowGenerationConfig, SubproblemConfig, BundleChoiceConfig, EllipsoidConfig
from .data_manager import DataManager
from .feature_manager import FeatureManager
from .subproblems.subproblem_manager import SubproblemManager
from mpi4py import MPI
from typing import Optional, Callable, Any
from bundlechoice.utils import get_logger
from bundlechoice.estimation import row_generationerationSolver
from bundlechoice.estimation.ellipsoid import EllipsoidSolver
from bundlechoice.estimation.inequalities import InequalitiesSolver
from bundlechoice.base import HasComm, HasConfig
from .comm_manager import CommManager
logger = get_logger(__name__)


class BundleChoice(HasComm, HasConfig):
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
        config: Main configuration object containing all components
        data_manager: Data management component
        feature_manager: Feature extraction component
        subproblem_manager: Subproblem solving component
        row_generation_manager: Row generation estimation component
        ellipsoid_manager: Ellipsoid method estimation component
        comm: MPI communicator
    """
    config: Optional[BundleChoiceConfig]
    data_manager: Optional[DataManager]
    feature_manager: Optional[FeatureManager]
    subproblem_manager: Optional[SubproblemManager]
    row_generation_manager: Optional[row_generationerationSolver]
    ellipsoid_manager: Optional[EllipsoidSolver]
    inequalities_manager: Optional[InequalitiesSolver]
    comm: MPI.Comm
    comm_manager: Optional[CommManager]

    def __init__(self):
        """
        Initialize an empty BundleChoice instance.
        
        All configuration, data, and features must be loaded explicitly via the
        provided methods. The MPI communicator is automatically initialized to
        COMM_WORLD.
        """
        self.config = None
        self.comm_manager = CommManager(MPI.COMM_WORLD)
        self.data_manager = None
        self.feature_manager = None
        self.subproblem_manager = None
        self.row_generation_manager = None
        self.ellipsoid_manager = None
        self.inequalities_manager = None

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
        if self.config is None or self.config.dimensions is None:
            raise ValueError("dimensions_cfg must be set in config before initializing data manager.")
        
        self.data_manager = DataManager(
            dimensions_cfg=self.config.dimensions,
            comm_manager=self.comm_manager
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
        if self.config is None or self.config.dimensions is None:
            logger.error("dimensions_cfg must be set in config before initializing feature manager.")
        
        self.feature_manager = FeatureManager(
            dimensions_cfg=self.config.dimensions,
            comm_manager=self.comm_manager,
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
        if self.data_manager is None or self.feature_manager is None or self.config is None or self.config.subproblem is None:
            raise RuntimeError("DataManager, FeatureManager, and SubproblemConfig must be set in config before initializing subproblem manager.")

        self.subproblem_manager = SubproblemManager(
            dimensions_cfg=self.config.dimensions,
            comm_manager=self.comm_manager,
            data_manager=self.data_manager,
            feature_manager=self.feature_manager,
            subproblem_cfg=self.config.subproblem
        )
        self.subproblem_manager.load()
        return self.subproblem_manager
    
    def _try_init_row_generation_manager(self):
        """
        Initialize the row_generationerationSolver if not already present.
        
        This method creates a new row_generationerationSolver instance using the current
        configuration and manager components.
        
        Returns:
            row_generationerationSolver: The initialized row_generationerationSolver instance
            
        Raises:
            RuntimeError: If required managers are not set
        """
        if self.data_manager is None or self.feature_manager is None or self.subproblem_manager is None or self.config is None or self.config.row_generation is None:
            missing_managers = []
            if self.data_manager is None:
                missing_managers.append("DataManager")
            if self.feature_manager is None:
                missing_managers.append("FeatureManager")
            if self.subproblem_manager is None:
                missing_managers.append("SubproblemManager")
            if self.config is None or self.config.row_generation is None:
                missing_managers.append("RowGenerationConfig")
            raise RuntimeError(
                "DataManager, FeatureManager, SubproblemManager, and RowGenerationConfig must be set in config "
                "before initializing row generation manager. Missing managers: "
                f"{', '.join(missing_managers)}"
            )

        self.row_generation_manager = row_generationerationSolver(
            comm_manager=self.comm_manager,
            dimensions_cfg=self.config.dimensions,
            row_generation_cfg=self.config.row_generation,
            data_manager=self.data_manager,
            feature_manager=self.feature_manager,
            subproblem_manager=self.subproblem_manager
        )
        return self.row_generation_manager

    

    def _try_init_ellipsoid_manager(self):
        """
        Initialize the EllipsoidSolver if not already present.
        
        This method creates a new EllipsoidSolver instance using the current
        configuration and manager components.
        
        Returns:
            EllipsoidSolver: The initialized EllipsoidSolver instance
            
        Raises:
            RuntimeError: If required managers are not set
        """
        if self.data_manager is None or self.feature_manager is None or self.subproblem_manager is None or self.config is None or self.config.ellipsoid is None:
            missing_managers = []
            if self.data_manager is None:
                missing_managers.append("DataManager")
            if self.feature_manager is None:
                missing_managers.append("FeatureManager")
            if self.subproblem_manager is None:
                missing_managers.append("SubproblemManager")
            if self.config is None or self.config.ellipsoid is None:
                missing_managers.append("EllipsoidConfig")
            raise RuntimeError(
                "DataManager, FeatureManager, SubproblemManager, and EllipsoidConfig must be set in config "
                "before initializing ellipsoid manager. Missing managers: "
                f"{', '.join(missing_managers)}"
            )

        self.ellipsoid_manager = EllipsoidSolver(
            comm_manager=self.comm_manager,
            dimensions_cfg=self.config.dimensions,
            ellipsoid_cfg=self.config.ellipsoid,
            data_manager=self.data_manager,
            feature_manager=self.feature_manager,
            subproblem_manager=self.subproblem_manager
        )
        return self.ellipsoid_manager

    def _try_init_inequalities_manager(self):
        """
        Initialize the InequalitiesSolver if required managers are set.
        
        This method creates a new InequalitiesSolver instance using the current
        configuration and managers.
        
        Returns:
            InequalitiesSolver: The initialized inequalities solver instance
            
        Raises:
            RuntimeError: If required managers are not initialized
        """
        missing_managers = []
        if self.data_manager is None:
            missing_managers.append("DataManager")
        if self.feature_manager is None:
            missing_managers.append("FeatureManager")
        if self.subproblem_manager is None:
            missing_managers.append("SubproblemManager")
        if self.config is None or self.config.dimensions is None:
            missing_managers.append("DimensionsConfig")
        if missing_managers:
            raise RuntimeError(
                "DataManager, FeatureManager, SubproblemManager, and DimensionsConfig must be set in config "
                "before initializing inequalities manager. Missing managers: "
                f"{', '.join(missing_managers)}"
            )

        self.inequalities_manager = InequalitiesSolver(
            dimensions_cfg=self.config.dimensions,
            data_manager=self.data_manager,
            feature_manager=self.feature_manager,
        )
        return self.inequalities_manager

    # --- Properties ---
    @property
    def data(self):
        """
        Access the data manager component.
        
        Returns:
            DataManager: The data manager instance
        """
        if self.data_manager is None:
            # if self.comm_manager.is_root():
            #     print("*"*100)
            #     print("Initializing data manager")
            #     print("*"*100)
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
            # if self.comm_manager.is_root():
            #     print("*"*100)
            #     print("Initializing feature manager")
            #     print("*"*100)
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
            # if self.comm_manager.is_root():
            #     print("*"*100)
            #     print("Initializing subproblem manager")
            #     print("*"*100)
            self._try_init_subproblem_manager()
        return self.subproblem_manager

    @property
    def row_generation(self):
        """
        Access the row generation manager component.
        
        Returns:
            row_generationerationSolver: The row generation solver instance
        """
        if self.row_generation_manager is None:
            self._try_init_row_generation_manager()
        return self.row_generation_manager
        
    @property
    def ellipsoid(self):
        """
        Access the ellipsoid manager component.
        
        Returns:
            EllipsoidSolver: The ellipsoid solver instance
        """
        if self.ellipsoid_manager is None:
            self._try_init_ellipsoid_manager()
        return self.ellipsoid_manager
        
    @property
    def inequalities(self):
        """
        Access the inequalities manager component.
        
        Returns:
            InequalitiesSolver: The inequalities solver instance
        """
        if self.inequalities_manager is None:
            self._try_init_inequalities_manager()
        return self.inequalities_manager
        
    def load_config(self, cfg):
        """
        Load configuration from a dictionary or YAML file.
        
        This method merges the new configuration with existing configuration,
        preserving component references and only updating specified fields.
        
        Args:
            cfg: Dictionary or YAML file path with configuration
            
        Returns:
            BundleChoice: self for method chaining
            
        Raises:
            ValueError: If cfg is not a dictionary or string
        """
        if isinstance(cfg, str):
            new_config = BundleChoiceConfig.from_yaml(cfg)
        elif isinstance(cfg, dict):
            new_config = BundleChoiceConfig.from_dict(cfg)
        else:
            raise ValueError("cfg must be a dictionary or a YAML path.")

        # Merge with existing config instead of overwriting
        if self.config is None:
            self.config = new_config
        else:
            self.config.update_in_place(new_config)

        # Build informative configuration summary
        logger.info("BundleChoice configured:")
        
        # Add dimensions information
        if self.config.dimensions.num_agents is not None:
            logger.info(f"  • {self.config.dimensions.num_agents} agents")
        if self.config.dimensions.num_items is not None:
            logger.info(f"  • {self.config.dimensions.num_items} items")
        if self.config.dimensions.num_features is not None:
            logger.info(f"  • {self.config.dimensions.num_features} features")
        if self.config.dimensions.num_simuls > 1:
            logger.info(f"  • {self.config.dimensions.num_simuls} simulations")
        
        # Add algorithm information
        if self.config.subproblem.name:
            logger.info(f"  • Algorithm: {self.config.subproblem.name}")
        
        # Add solver information if configured
        if hasattr(self.config, 'row_generation') and self.config.row_generation.max_iters != float('inf'):
            logger.info(f"  • Max iterations: {self.config.row_generation.max_iters}")
        if hasattr(self.config, 'ellipsoid') and self.config.ellipsoid.max_iterations != 1000:
            logger.info(f"  • Ellipsoid iterations: {self.config.ellipsoid.max_iterations}")
        return self
