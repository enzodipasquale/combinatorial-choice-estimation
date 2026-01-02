from .config import BundleChoiceConfig
from .data_manager import DataManager
from .feature_manager import FeatureManager
from .subproblems.subproblem_manager import SubproblemManager
from mpi4py import MPI
from typing import Optional, Any, Dict, Union
import numpy as np
from bundlechoice.estimation import RowGenerationManager, StandardErrorsManager, ColumnGenerationManager
from bundlechoice.estimation.ellipsoid import EllipsoidManager
from bundlechoice.estimation.inequalities import InequalitiesManager
from bundlechoice.base import HasComm, HasConfig
from .comm_manager import CommManager


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
    row_generation_manager: Optional[RowGenerationManager]
    column_generation_manager: Optional[ColumnGenerationManager]
    ellipsoid_manager: Optional[EllipsoidManager]
    inequalities_manager: Optional[InequalitiesManager]
    standard_errors_manager: Optional[StandardErrorsManager]
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
        self.column_generation_manager = None
        self.ellipsoid_manager = None
        self.inequalities_manager = None
        self.standard_errors_manager = None

    # ============================================================================
    # Component Initialization
    # ============================================================================
    def _try_init_data_manager(self) -> DataManager:
        """Initialize DataManager from dimensions config."""
        from bundlechoice._initialization import try_init_data_manager
        return try_init_data_manager(self)
        
    def _try_init_feature_manager(self) -> FeatureManager:
        """Initialize FeatureManager from dimensions config."""
        from bundlechoice._initialization import try_init_feature_manager
        return try_init_feature_manager(self)
        
    def _try_init_subproblem_manager(self) -> SubproblemManager:
        """Initialize SubproblemManager and load algorithm."""
        from bundlechoice._initialization import try_init_subproblem_manager
        manager = try_init_subproblem_manager(self)
        # Auto-load subproblem on initialization (consistent with original behavior)
        manager.load()
        return manager
    
    def _try_init_row_generation_manager(self, theta_init: Optional[np.ndarray] = None) -> RowGenerationManager:
        """Initialize RowGenerationManager."""
        from bundlechoice._initialization import try_init_row_generation_manager
        return try_init_row_generation_manager(self)

    def _try_init_column_generation_manager(self, theta_init: Optional[np.ndarray] = None) -> ColumnGenerationManager:
        """Initialize ColumnGenerationManager."""
        from bundlechoice._initialization import try_init_column_generation_manager
        return try_init_column_generation_manager(self, theta_init)

    def _try_init_ellipsoid_manager(self, theta_init: Optional[np.ndarray] = None) -> EllipsoidManager:
        """Initialize EllipsoidManager."""
        from bundlechoice._initialization import try_init_ellipsoid_manager
        return try_init_ellipsoid_manager(self, theta_init)

    def _try_init_inequalities_manager(self) -> InequalitiesManager:
        """Initialize InequalitiesManager."""
        from bundlechoice._initialization import try_init_inequalities_manager
        return try_init_inequalities_manager(self)

    def _try_init_standard_errors_manager(self) -> StandardErrorsManager:
        """Initialize StandardErrorsManager."""
        from bundlechoice._initialization import try_init_standard_errors_manager
        return try_init_standard_errors_manager(self)

    # ============================================================================
    # Property Accessors (Lazy Initialization)
    # ============================================================================

    @property
    def data(self) -> DataManager:
        if self.data_manager is None:
            self._try_init_data_manager()
        return self.data_manager

    @property
    def features(self) -> FeatureManager:
        if self.feature_manager is None:
            self._try_init_feature_manager()
        return self.feature_manager

    @property
    def subproblems(self) -> SubproblemManager:
        if self.subproblem_manager is None:
            self._try_init_subproblem_manager()
        return self.subproblem_manager

    @property
    def row_generation(self) -> RowGenerationManager:
        if self.row_generation_manager is None:
            self._try_init_row_generation_manager()
        return self.row_generation_manager

    @property
    def column_generation(self) -> ColumnGenerationManager:
        if self.column_generation_manager is None:
            self._try_init_column_generation_manager()
        return self.column_generation_manager
        
    @property
    def ellipsoid(self) -> EllipsoidManager:
        if self.ellipsoid_manager is None:
            self._try_init_ellipsoid_manager()
        return self.ellipsoid_manager
        
    @property
    def inequalities(self) -> InequalitiesManager:
        if self.inequalities_manager is None:
            self._try_init_inequalities_manager()
        return self.inequalities_manager

    @property
    def standard_errors(self) -> StandardErrorsManager:
        if self.standard_errors_manager is None:
            self._try_init_standard_errors_manager()
        return self.standard_errors_manager
    
    # ============================================================================
    # Setup Status
    # ============================================================================
    
    def print_status(self) -> None:
        """Print formatted setup status to stdout."""
        print("\n=== BundleChoice Status ===")
        print(f"Config:      {'OK' if self.config is not None else 'Not set'}")
        print(f"Data:        {'OK' if self.data_manager is not None and self.data_manager.local_data is not None else 'Not set'}")
        print(f"Features:    {'OK' if self.feature_manager is not None and self.feature_manager._features_oracle is not None else 'Not set'}")
        print(f"Subproblems: {'OK' if self.subproblem_manager is not None and self.subproblem_manager.subproblem_instance is not None else 'Not set'}")
        
        if self.config and self.config.dimensions:
            dimensions_str = f"agents={self.config.dimensions.num_agents}, items={self.config.dimensions.num_items}, features={self.config.dimensions.num_features}"
        else:
            dimensions_str = 'Not set'
        print(f"\nDimensions:  {dimensions_str}")
        
        if self.config and self.config.subproblem and self.config.subproblem.name:
            subproblem_str = self.config.subproblem.name
        else:
            subproblem_str = 'Not set'
        print(f"Algorithm:   {subproblem_str}")
        print(f"MPI:         rank {self.rank}/{self.comm_size}")

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

        # Use print for configuration to avoid logging prefix clutter
        if self.is_root():
            print("=" * 70)
            print("BUNDLECHOICE CONFIGURATION")
            print("=" * 70)
            
            # Problem dimensions
            print("Problem Dimensions:")
            if self.config.dimensions.num_agents is not None:
                print(f"  • Agents: {self.config.dimensions.num_agents}")
            if self.config.dimensions.num_items is not None:
                print(f"  • Items: {self.config.dimensions.num_items}")
            if self.config.dimensions.num_features is not None:
                print(f"  • Features: {self.config.dimensions.num_features}")
            if self.config.dimensions.num_simulations > 1:
                print(f"  • Simulations: {self.config.dimensions.num_simulations}")
            
            # Subproblem configuration
            if self.config.subproblem.name:
                print("\nSubproblem:")
                print(f"  • Algorithm: {self.config.subproblem.name}")
                settings = self.config.subproblem.settings
                if settings:
                    if 'TimeLimit' in settings:
                        print(f"  • TimeLimit: {settings['TimeLimit']}s")
                    if 'MIPGap_tol' in settings:
                        print(f"  • MIPGap tolerance: {settings['MIPGap_tol']}")
                    if 'OutputFlag' in settings:
                        print(f"  • Gurobi output: {'enabled' if settings['OutputFlag'] == 1 else 'disabled'}")
            
            # Ellipsoid configuration
            if hasattr(self.config, 'ellipsoid') and self.config.ellipsoid.max_iterations != 1000:
                print("\nEllipsoid:")
                print(f"  • Max iterations: {self.config.ellipsoid.max_iterations}")
                if self.config.ellipsoid.tolerance != 1e-6:
                    print(f"  • Tolerance: {self.config.ellipsoid.tolerance}")
            
            # MPI information
            if self.comm_manager is not None:
                print("\nParallelization:")
                print(f"  • MPI workers: {self.comm_manager.comm.Get_size()}")
            
            print()  # Blank line to separate from next section
        return self
