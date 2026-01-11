"""
BundleChoice: Main orchestrator for bundle choice estimation.
"""

from typing import Optional, Dict, Any, Union, TYPE_CHECKING
import numpy as np
from mpi4py import MPI
from bundlechoice.config import BundleChoiceConfig
from bundlechoice.base import HasComm, HasConfig
from bundlechoice.comm_manager import CommManager
from bundlechoice.utils import get_logger

if TYPE_CHECKING:
    from bundlechoice.data_manager import DataManager
    from bundlechoice.feature_manager import FeatureManager
    from bundlechoice.subproblems.subproblem_manager import SubproblemManager
    from bundlechoice.estimation import RowGenerationManager, StandardErrorsManager, ColumnGenerationManager
    from bundlechoice.estimation.ellipsoid import EllipsoidManager

logger = get_logger(__name__)


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
    
    def __init__(self) -> None:
        """Initialize empty BundleChoice instance."""
        self.config: Optional[BundleChoiceConfig] = None
        self.comm_manager: CommManager = CommManager(MPI.COMM_WORLD)
        self.data_manager: Optional['DataManager'] = None
        self.feature_manager: Optional['FeatureManager'] = None
        self.subproblem_manager: Optional['SubproblemManager'] = None
        self.row_generation_manager: Optional['RowGenerationManager'] = None
        self.column_generation_manager: Optional['ColumnGenerationManager'] = None
        self.ellipsoid_manager: Optional['EllipsoidManager'] = None
        self.standard_errors_manager: Optional['StandardErrorsManager'] = None

    # ========================================================================
    # Component Initialization
    # ========================================================================
    
    def _try_init_data_manager(self) -> 'DataManager':
        from bundlechoice._initialization import try_init_data_manager
        return try_init_data_manager(self)
        
    def _try_init_feature_manager(self) -> 'FeatureManager':
        from bundlechoice._initialization import try_init_feature_manager
        return try_init_feature_manager(self)
        
    def _try_init_subproblem_manager(self) -> 'SubproblemManager':
        from bundlechoice._initialization import try_init_subproblem_manager
        manager = try_init_subproblem_manager(self)
        manager.load()
        return manager
    
    def _try_init_row_generation_manager(self) -> 'RowGenerationManager':
        from bundlechoice._initialization import try_init_row_generation_manager
        return try_init_row_generation_manager(self)

    def _try_init_ellipsoid_manager(self, theta_init: Optional[np.ndarray] = None) -> 'EllipsoidManager':
        from bundlechoice._initialization import try_init_ellipsoid_manager
        return try_init_ellipsoid_manager(self, theta_init)

    def _try_init_column_generation_manager(self, theta_init: Optional[np.ndarray] = None) -> 'ColumnGenerationManager':
        from bundlechoice._initialization import try_init_column_generation_manager
        return try_init_column_generation_manager(self, theta_init)

    def _try_init_standard_errors_manager(self) -> 'StandardErrorsManager':
        from bundlechoice._initialization import try_init_standard_errors_manager
        return try_init_standard_errors_manager(self)

    # ========================================================================
    # Property Accessors (Lazy Initialization)
    # ========================================================================

    @property
    def data(self) -> 'DataManager':
        if self.data_manager is None:
            self._try_init_data_manager()
        return self.data_manager

    @property
    def features(self) -> 'FeatureManager':
        if self.feature_manager is None:
            self._try_init_feature_manager()
        return self.feature_manager

    @property
    def subproblems(self) -> 'SubproblemManager':
        if self.subproblem_manager is None:
            self._try_init_subproblem_manager()
        return self.subproblem_manager

    @property
    def row_generation(self) -> 'RowGenerationManager':
        if self.row_generation_manager is None:
            self._try_init_row_generation_manager()
        return self.row_generation_manager
        
    @property
    def ellipsoid(self) -> 'EllipsoidManager':
        if self.ellipsoid_manager is None:
            self._try_init_ellipsoid_manager()
        return self.ellipsoid_manager

    @property
    def column_generation(self) -> 'ColumnGenerationManager':
        if self.column_generation_manager is None:
            self._try_init_column_generation_manager()
        return self.column_generation_manager
        
    @property
    def standard_errors(self) -> 'StandardErrorsManager':
        if self.standard_errors_manager is None:
            self._try_init_standard_errors_manager()
        return self.standard_errors_manager
    
    # ========================================================================
    # Setup Status
    # ========================================================================
    
    def print_status(self) -> None:
        """Print formatted setup status."""
        lines = ["\n=== BundleChoice Status ==="]
        lines.append(f"Config:      {'OK' if self.config else 'Not set'}")
        lines.append(f"Data:        {'OK' if self.data_manager and self.data_manager.local_data else 'Not set'}")
        lines.append(f"Features:    {'OK' if self.feature_manager and self.feature_manager._features_oracle else 'Not set'}")
        lines.append(f"Subproblems: {'OK' if self.subproblem_manager and self.subproblem_manager.subproblem_instance else 'Not set'}")
        
        if self.config and self.config.dimensions:
            d = self.config.dimensions
            lines.append(f"\nDimensions:  agents={d.num_agents}, items={d.num_items}, features={d.num_features}")
        else:
            lines.append("\nDimensions:  Not set")
        
        algo = self.config.subproblem.name if self.config and self.config.subproblem else 'Not set'
        lines.append(f"Algorithm:   {algo}")
        lines.append(f"MPI:         rank {self.rank}/{self.comm_size}")
        
        logger.info("\n".join(lines))

    # ========================================================================
    # Workflow Methods
    # ========================================================================

    def generate_observations(self, theta_true: np.ndarray) -> Optional[np.ndarray]:
        """Generate observed bundles from true parameters, then reload data."""
        obs_bundles = self.subproblems.init_and_solve(theta_true)
        
        if self.is_root():
            if self.data_manager.input_data is None:
                raise RuntimeError("Cannot generate observations without input_data")
            self.data_manager.input_data["obs_bundle"] = obs_bundles
            updated_data = self.data_manager.input_data
        else:
            updated_data = None
        
        self.data.load_and_scatter(updated_data)
        
        if self.feature_manager._features_oracle is not None:
            oracle_code = self.feature_manager._features_oracle.__code__
            if 'features_oracle' in oracle_code.co_name:
                self.features.build_from_data()
        
        return obs_bundles

    # ========================================================================
    # Configuration Management
    # ========================================================================
        
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
        
        if self.is_root():
            self._log_config()
        return self
    
    def _log_config(self) -> None:
        """Log configuration summary."""
        lines = ["=" * 70, "BUNDLECHOICE CONFIGURATION", "=" * 70]
        
        # Problem dimensions
        lines.append("Problem Dimensions:")
        d = self.config.dimensions
        if d.num_agents is not None:
            lines.append(f"  • Agents: {d.num_agents}")
        if d.num_items is not None:
            lines.append(f"  • Items: {d.num_items}")
        if d.num_features is not None:
            lines.append(f"  • Features: {d.num_features}")
        if d.num_simulations > 1:
            lines.append(f"  • Simulations: {d.num_simulations}")
        
        # Subproblem configuration
        if self.config.subproblem.name:
            lines.append("\nSubproblem:")
            lines.append(f"  • Algorithm: {self.config.subproblem.name}")
            settings = self.config.subproblem.settings
            if settings:
                if 'TimeLimit' in settings:
                    lines.append(f"  • TimeLimit: {settings['TimeLimit']}s")
                if 'MIPGap_tol' in settings:
                    lines.append(f"  • MIPGap tolerance: {settings['MIPGap_tol']}")
        
        # Ellipsoid configuration (only if non-default)
        if hasattr(self.config, 'ellipsoid') and self.config.ellipsoid.max_iterations != 1000:
            lines.append("\nEllipsoid:")
            lines.append(f"  • Max iterations: {self.config.ellipsoid.max_iterations}")
            if self.config.ellipsoid.tolerance != 1e-6:
                lines.append(f"  • Tolerance: {self.config.ellipsoid.tolerance}")
        
        # MPI information
        if self.comm_manager:
            lines.append("\nParallelization:")
            lines.append(f"  • MPI workers: {self.comm_manager.comm.Get_size()}")
        
        lines.append("")
        logger.info("\n".join(lines))
