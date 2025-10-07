from .config import DimensionsConfig, RowGenerationConfig, SubproblemConfig, BundleChoiceConfig, EllipsoidConfig
from .data_manager import DataManager
from .feature_manager import FeatureManager
from .subproblems.subproblem_manager import SubproblemManager
from mpi4py import MPI
from typing import Optional, Callable, Any
from bundlechoice.utils import get_logger
from bundlechoice.estimation import RowGenerationSolver
from bundlechoice.estimation.ellipsoid import EllipsoidSolver
from bundlechoice.estimation.inequalities import InequalitiesSolver
from bundlechoice.base import HasComm, HasConfig
from .comm_manager import CommManager
from contextlib import contextmanager
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
    row_generation_manager: Optional[RowGenerationSolver]
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
            raise ValueError("dimensions_cfg must be set in config before initializing feature manager.")
        
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
            missing = []
            if self.data_manager is None:
                missing.append("data (call bc.data.load_and_scatter(input_data))")
            if self.feature_manager is None:
                missing.append("features (call bc.features.set_oracle(fn) or bc.features.build_from_data())")
            if self.config is None or self.config.subproblem is None:
                missing.append("subproblem config (add 'subproblem' to your config)")
            raise RuntimeError(
                "Cannot initialize subproblem manager - missing setup:\n  " +
                "\n  ".join(missing)
            )

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
        Initialize the RowGenerationSolver if not already present.
        
        This method creates a new RowGenerationSolver instance using the current
        configuration and manager components.
        
        Returns:
            RowGenerationSolver: The initialized RowGenerationSolver instance
            
        Raises:
            RuntimeError: If required managers are not set
        """
        if self.data_manager is None or self.feature_manager is None or self.subproblem_manager is None or self.config is None or self.config.row_generation is None:
            missing = []
            if self.data_manager is None:
                missing.append("data (call bc.data.load_and_scatter(input_data))")
            if self.feature_manager is None:
                missing.append("features (call bc.features.set_oracle(fn) or bc.features.build_from_data())")
            if self.subproblem_manager is None:
                missing.append("subproblem (call bc.subproblems.load())")
            if self.config is None or self.config.row_generation is None:
                missing.append("row_generation config (add 'row_generation' to your config)")
            raise RuntimeError(
                "Cannot initialize row generation solver - missing setup:\n  " +
                "\n  ".join(missing) +
                "\n\nRun bc.validate_setup('row_generation') to check your configuration."
            )

        self.row_generation_manager = RowGenerationSolver(
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
            missing = []
            if self.data_manager is None:
                missing.append("data (call bc.data.load_and_scatter(input_data))")
            if self.feature_manager is None:
                missing.append("features (call bc.features.set_oracle(fn) or bc.features.build_from_data())")
            if self.subproblem_manager is None:
                missing.append("subproblem (call bc.subproblems.load())")
            if self.config is None or self.config.ellipsoid is None:
                missing.append("ellipsoid config (add 'ellipsoid' to your config)")
            raise RuntimeError(
                "Cannot initialize ellipsoid solver - missing setup:\n  " +
                "\n  ".join(missing) +
                "\n\nRun bc.validate_setup('ellipsoid') to check your configuration."
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
            comm_manager=self.comm_manager,
            dimensions_cfg=self.config.dimensions,
            data_manager=self.data_manager,
            feature_manager=self.feature_manager,
            subproblem_manager=None
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
    
    def validate_setup(self, for_method='row_generation'):
        """
        Validate that all components are initialized for the specified estimation method.
        
        Args:
            for_method: Estimation method to validate for ('row_generation', 'ellipsoid', or 'inequalities')
        
        Raises:
            RuntimeError: If setup is incomplete with helpful guidance
        
        Returns:
            bool: True if setup is valid
        """
        missing = []
        
        if self.config is None:
            missing.append("config (call bc.load_config(...))")
        if self.data_manager is None:
            missing.append("data (call bc.data.load_and_scatter(input_data))")
        if self.feature_manager is None or self.feature_manager._features_oracle is None:
            missing.append("features (call bc.features.set_oracle(fn) or bc.features.build_from_data())")
        if self.subproblem_manager is None and for_method in ['row_generation', 'ellipsoid']:
            missing.append("subproblem (call bc.subproblems.load())")
        
        if for_method == 'row_generation' and (self.config is None or self.config.row_generation is None):
            missing.append("row_generation config (add 'row_generation' to your config)")
        elif for_method == 'ellipsoid' and (self.config is None or self.config.ellipsoid is None):
            missing.append("ellipsoid config (add 'ellipsoid' to your config)")
        
        if missing:
            raise RuntimeError(
                f"Setup incomplete for {for_method}:\n  " +
                "\n  ".join(f"- {m}" for m in missing)
            )
        
        logger.info("✅ Setup validated for %s", for_method)
        return True
    
    def generate_observations(self, theta_true):
        """
        Generate observed bundles from true parameters and reload data.
        Handles the common workflow pattern automatically.
        
        Args:
            theta_true: True parameter vector for generating observations
        
        Returns:
            Observed bundles (rank 0 only, None on other ranks)
        
        Example:
            >>> bc.data.load_and_scatter(input_data)
            >>> bc.features.build_from_data()
            >>> bc.generate_observations(theta_true)
            >>> theta_hat = bc.row_generation.solve()
        """
        obs_bundles = self.subproblems.init_and_solve(theta_true)
        
        # Prepare input_data on rank 0
        if self.is_root():
            if self.data_manager.input_data is None:
                raise RuntimeError("Cannot generate observations without input_data")
            self.data_manager.input_data["obs_bundle"] = obs_bundles
            updated_data = self.data_manager.input_data
        else:
            updated_data = None
        
        # All ranks participate in scatter
        self.data.load_and_scatter(updated_data)
        
        # Rebuild features if using auto-generated oracle
        if self.feature_manager._features_oracle is not None:
            oracle_code = self.feature_manager._features_oracle.__code__
            if 'features_oracle' in oracle_code.co_name:
                self.features.build_from_data()
        
        return obs_bundles
    
    @contextmanager
    def temp_config(self, **updates):
        """
        Temporarily modify configuration.
        
        Args:
            **updates: Configuration updates to apply temporarily
        
        Example:
            >>> with bc.temp_config(row_generation={'max_iters': 5}):
            ...     quick_theta = bc.row_generation.solve()
            >>> # Config restored to original
            >>> final_theta = bc.row_generation.solve()
        """
        import copy
        original_config = copy.deepcopy(self.config)
        try:
            self.load_config(updates)
            yield self
        finally:
            self.config = original_config
    
    def quick_setup(self, config, input_data, features_oracle=None):
        """
        Quick setup for common workflow.
        Combines load_config, load_and_scatter, features, and subproblems.load().
        
        Args:
            config: Configuration dict or YAML path
            input_data: Input data dictionary
            features_oracle: Feature function or None to auto-generate
        
        Returns:
            self for method chaining
        
        Example:
            >>> bc = BundleChoice().quick_setup(cfg, data, my_features)
            >>> theta = bc.row_generation.solve()
        """
        self.load_config(config)
        self.data.load_and_scatter(input_data)
        
        if features_oracle is not None:
            self.features.set_oracle(features_oracle)
        else:
            self.features.build_from_data()
        
        self.subproblems.load()
        return self
        
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
        
        # Validate configuration
        self.config.validate()

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
