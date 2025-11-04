from .config import DimensionsConfig, RowGenerationConfig, SubproblemConfig, BundleChoiceConfig, EllipsoidConfig
from .data_manager import DataManager
from .feature_manager import FeatureManager
from .subproblems.subproblem_manager import SubproblemManager
from mpi4py import MPI
from typing import Optional, Callable, Any, Dict, Union
import numpy as np
from bundlechoice.utils import get_logger
from bundlechoice.estimation import RowGenerationSolver
from bundlechoice.estimation.ellipsoid import EllipsoidSolver
from bundlechoice.estimation.inequalities import InequalitiesSolver
from bundlechoice.base import HasComm, HasConfig
from .comm_manager import CommManager
from contextlib import contextmanager
logger = get_logger(__name__)


# ============================================================================
# Main BundleChoice Class
# ============================================================================

class BundleChoice(HasComm, HasConfig):
    """
    Main orchestrator for bundle choice estimation.
    
    Manages data loading, feature extraction, subproblem solving, and parameter
    estimation with MPI distribution. Components are lazily initialized on access.
    
    Example:
        bc = BundleChoice()
        bc.load_config(cfg)
        bc.data.load_and_scatter(input_data)
        bc.features.build_from_data()
        theta = bc.row_generation.solve()
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

    def __init__(self) -> None:
        """Initialize empty BundleChoice instance. Config and data must be loaded separately."""
        self.config = None
        self.comm_manager = CommManager(MPI.COMM_WORLD)
        self.data_manager = None
        self.feature_manager = None
        self.subproblem_manager = None
        self.row_generation_manager = None
        self.ellipsoid_manager = None
        self.inequalities_manager = None

    # ============================================================================
    # Component Initialization
    # ============================================================================
    def _try_init_data_manager(self) -> DataManager:
        """Initialize DataManager from dimensions config."""
        if self.config is None or self.config.dimensions is None:
            raise ValueError("dimensions_cfg must be set in config before initializing data manager.")
        
        self.data_manager = DataManager(
            dimensions_cfg=self.config.dimensions,
            comm_manager=self.comm_manager
        )
        return self.data_manager
        
    def _try_init_feature_manager(self) -> FeatureManager:
        """Initialize FeatureManager from dimensions config."""
        if self.config is None or self.config.dimensions is None:
            raise ValueError("dimensions_cfg must be set in config before initializing feature manager.")
        
        self.feature_manager = FeatureManager(
            dimensions_cfg=self.config.dimensions,
            comm_manager=self.comm_manager,
            data_manager=self.data_manager
        )
        return self.feature_manager
        
    def _try_init_subproblem_manager(self) -> SubproblemManager:
        """Initialize SubproblemManager and load algorithm."""
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
                "\n  ".join(f"✗ {m}" for m in missing) +
                "\n\nRun bc.print_status() to see your current setup state."
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
    
    def _try_init_row_generation_manager(self) -> RowGenerationSolver:
        """Initialize RowGenerationSolver."""
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

    def _try_init_ellipsoid_manager(self) -> EllipsoidSolver:
        """Initialize EllipsoidSolver."""
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

    def _try_init_inequalities_manager(self) -> InequalitiesSolver:
        """Initialize InequalitiesSolver."""
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

    # ============================================================================
    # Property Accessors (Lazy Initialization)
    # ============================================================================

    @property
    def data(self) -> DataManager:
        """Access data manager (initialized on first access)."""
        if self.data_manager is None:
            self._try_init_data_manager()
        return self.data_manager

    @property
    def features(self) -> FeatureManager:
        """Access feature manager (initialized on first access)."""
        if self.feature_manager is None:
            self._try_init_feature_manager()
        return self.feature_manager

    @property
    def subproblems(self) -> SubproblemManager:
        """Access subproblem manager (initialized on first access)."""
        if self.subproblem_manager is None:
            self._try_init_subproblem_manager()
        return self.subproblem_manager

    @property
    def row_generation(self) -> RowGenerationSolver:
        """Access row generation solver (initialized on first access)."""
        if self.row_generation_manager is None:
            self._try_init_row_generation_manager()
        return self.row_generation_manager
        
    @property
    def ellipsoid(self) -> EllipsoidSolver:
        """Access ellipsoid solver (initialized on first access)."""
        if self.ellipsoid_manager is None:
            self._try_init_ellipsoid_manager()
        return self.ellipsoid_manager
        
    @property
    def inequalities(self) -> InequalitiesSolver:
        """Access inequalities solver (initialized on first access)."""
        if self.inequalities_manager is None:
            self._try_init_inequalities_manager()
        return self.inequalities_manager
    
    # ============================================================================
    # Setup Validation & Status
    # ============================================================================

    def validate_setup(self, for_method: str = 'row_generation') -> bool:
        """Validate setup for estimation method. Raises RuntimeError if incomplete."""
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
    
    def status(self) -> Dict[str, Any]:
        """Get setup status dictionary. Returns component initialization state."""
        return {
            'config_loaded': self.config is not None,
            'data_loaded': self.data_manager is not None and self.data_manager.local_data is not None,
            'features_set': self.feature_manager is not None and self.feature_manager._features_oracle is not None,
            'subproblems_ready': self.subproblem_manager is not None and self.subproblem_manager.demand_oracle is not None,
            'dimensions': f"agents={self.config.dimensions.num_agents}, items={self.config.dimensions.num_items}, features={self.config.dimensions.num_features}" if self.config and self.config.dimensions else 'Not set',
            'subproblem': self.config.subproblem.name if self.config and self.config.subproblem and self.config.subproblem.name else 'Not set',
            'mpi_rank': self.rank,
            'mpi_size': self.comm_size,
        }
    
    def print_status(self) -> None:
        """Print formatted setup status to stdout."""
        status = self.status()
        print("\n=== BundleChoice Status ===")
        print(f"Config:      {'✓' if status['config_loaded'] else '✗'}")
        print(f"Data:        {'✓' if status['data_loaded'] else '✗'}")
        print(f"Features:    {'✓' if status['features_set'] else '✗'}")
        print(f"Subproblems: {'✓' if status['subproblems_ready'] else '✗'}")
        print(f"\nDimensions:  {status['dimensions']}")
        print(f"Algorithm:   {status['subproblem']}")
        print(f"MPI:         rank {status['mpi_rank']}/{status['mpi_size']}")

    # ============================================================================
    # Workflow Methods
    # ============================================================================

    def generate_observations(self, theta_true: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate observed bundles from true parameters, then reload data.
        
        Args:
            theta_true: True parameter vector
            
        Returns:
            Observed bundles (rank 0 only, None on other ranks)
        """
        obs_bundles = self.subproblems.init_and_solve(theta_true)
        
        if self.is_root():
            if self.data_manager.input_data is None:
                raise RuntimeError("Cannot generate observations without input_data")
            self.data_manager.input_data["obs_bundle"] = obs_bundles
            updated_data = self.data_manager.input_data
        else:
            updated_data = None
        
        self.data.load_and_scatter(updated_data)
        
        # Rebuild features if using auto-generated oracle
        if self.feature_manager._features_oracle is not None:
            oracle_code = self.feature_manager._features_oracle.__code__
            if 'features_oracle' in oracle_code.co_name:
                self.features.build_from_data()
        
        return obs_bundles
    
    @contextmanager
    def temp_config(self, **updates: Dict[str, Any]):
        """Temporarily modify configuration (restored after context)."""
        import copy
        original_config = copy.deepcopy(self.config)
        try:
            self.load_config(updates)
            yield self
        finally:
            self.config = original_config
    
    def quick_setup(self, config: Union[Dict[str, Any], str], input_data: Dict[str, Any], 
                   features_oracle: Optional[Callable] = None) -> 'BundleChoice':
        """Quick setup: config + data + features + subproblems in one call."""
        self.load_config(config)
        self.data.load_and_scatter(input_data)
        
        if features_oracle is not None:
            self.features.set_oracle(features_oracle)
        else:
            self.features.build_from_data()
        
        self.subproblems.load()
        return self

    # ============================================================================
    # Configuration Management
    # ============================================================================
        
    def load_config(self, cfg: Union[Dict[str, Any], str]) -> 'BundleChoice':
        """Load configuration from dict or YAML file. Merges with existing config."""
        if isinstance(cfg, str):
            new_config = BundleChoiceConfig.from_yaml(cfg)
        elif isinstance(cfg, dict):
            new_config = BundleChoiceConfig.from_dict(cfg)
        else:
            raise ValueError("cfg must be a dictionary or a YAML path.")

        if self.config is None:
            self.config = new_config
        else:
            self.config.update_in_place(new_config)
        
        self.config.validate()

        logger.info("BundleChoice configured:")
        if self.config.dimensions.num_agents is not None:
            logger.info(f"  • {self.config.dimensions.num_agents} agents")
        if self.config.dimensions.num_items is not None:
            logger.info(f"  • {self.config.dimensions.num_items} items")
        if self.config.dimensions.num_features is not None:
            logger.info(f"  • {self.config.dimensions.num_features} features")
        if self.config.dimensions.num_simuls > 1:
            logger.info(f"  • {self.config.dimensions.num_simuls} simulations")
        if self.config.subproblem.name:
            logger.info(f"  • Algorithm: {self.config.subproblem.name}")
        if hasattr(self.config, 'row_generation') and self.config.row_generation.max_iters != float('inf'):
            logger.info(f"  • Max iterations: {self.config.row_generation.max_iters}")
        if hasattr(self.config, 'ellipsoid') and self.config.ellipsoid.max_iterations != 1000:
            logger.info(f"  • Ellipsoid iterations: {self.config.ellipsoid.max_iterations}")
        return self
