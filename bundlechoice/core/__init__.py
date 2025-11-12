"""
BundleChoice core module.

This module contains the main BundleChoice orchestrator class and its components.
"""

from typing import Optional, Any
from mpi4py import MPI

from bundlechoice.config import BundleChoiceConfig
from bundlechoice.data_manager import DataManager
from bundlechoice.feature_manager import FeatureManager
from bundlechoice.subproblems.subproblem_manager import SubproblemManager
from bundlechoice.estimation import RowGenerationSolver
from bundlechoice.estimation.ellipsoid import EllipsoidSolver
from bundlechoice.estimation.inequalities import InequalitiesSolver
from bundlechoice.base import HasComm, HasConfig
from bundlechoice.comm_manager import CommManager
from bundlechoice.utils import get_logger

from . import _initialization as init_module
from . import _validation as validation_module
from . import _workflow as workflow_module

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
    column_generation_manager: Optional[Any]
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
        self.column_generation_manager = None

    # --- Initialization methods (delegated to _initialization module) ---
    
    def _try_init_data_manager(self):
        return init_module.try_init_data_manager(self)
    
    def _try_init_feature_manager(self):
        return init_module.try_init_feature_manager(self)
    
    def _try_init_subproblem_manager(self):
        return init_module.try_init_subproblem_manager(self)
    
    def _try_init_row_generation_manager(self):
        return init_module.try_init_row_generation_manager(self)
    
    def _try_init_ellipsoid_manager(self):
        return init_module.try_init_ellipsoid_manager(self)
    
    def _try_init_inequalities_manager(self):
        return init_module.try_init_inequalities_manager(self)

    def _try_init_column_generation_manager(self, theta_init: Optional[Any] = None):
        return init_module.try_init_column_generation_manager(self, theta_init)

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
            # Ensure subproblem_manager is initialized first
            _ = self.subproblems
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
            # Ensure subproblem_manager is initialized first
            _ = self.subproblems
            self._try_init_ellipsoid_manager()
        return self.ellipsoid_manager

    @property
    def column_generation(self):
        """
        Access the column generation solver component.

        Returns:
            ColumnGenerationSolver: The column generation solver instance
        """
        if self.column_generation_manager is None:
            _ = self.subproblems
            self._try_init_column_generation_manager()
        return self.column_generation_manager
        
    @property
    def inequalities(self):
        """
        Access the inequalities manager component.
        
        Returns:
            InequalitiesSolver: The inequalities solver instance
        """
        if self.inequalities_manager is None:
            # Ensure subproblem_manager is initialized first (if needed by config)
            if self.config and self.config.subproblem:
                _ = self.subproblems
            self._try_init_inequalities_manager()
        return self.inequalities_manager

    # --- Validation methods (delegated to _validation module) ---
    
    def validate_setup(self, for_method='row_generation'):
        """Validate that all components are initialized for the specified estimation method."""
        return validation_module.validate_setup(self, for_method)
    
    def status(self) -> dict:
        """Get setup status summary."""
        return validation_module.status(self)
    
    def print_status(self):
        """Print formatted setup status."""
        validation_module.print_status(self)

    # --- Workflow methods (delegated to _workflow module) ---
    
    def generate_observations(self, theta_true):
        """Generate observed bundles from true parameters and reload data."""
        return workflow_module.generate_observations(self, theta_true)
    
    def temp_config(self, **updates):
        """Temporarily modify configuration."""
        return workflow_module.temp_config(self, **updates)
    
    def quick_setup(self, config, input_data, features_oracle=None):
        """Quick setup for common workflow."""
        return workflow_module.quick_setup(self, config, input_data, features_oracle)

    # --- Configuration ---
    
    def configure(self, cfg):
        """
        Configure BundleChoice from a dictionary, YAML file, or Config object.
        
        This method merges the new configuration with existing configuration,
        preserving component references and only updating specified fields.
        
        Args:
            cfg: Dictionary, YAML file path, or BundleChoiceConfig object
            
        Returns:
            BundleChoice: self for method chaining
            
        Raises:
            ValueError: If cfg is not a valid configuration type
        """
        if isinstance(cfg, str):
            new_config = BundleChoiceConfig.from_yaml(cfg)
        elif isinstance(cfg, dict):
            new_config = BundleChoiceConfig.from_dict(cfg)
        elif isinstance(cfg, BundleChoiceConfig):
            new_config = cfg
        else:
            raise ValueError("cfg must be a dictionary, YAML path, or BundleChoiceConfig object.")

        # Merge with existing config instead of overwriting
        if self.config is None:
            self.config = new_config
        else:
            self.config.update_in_place(new_config)
        
        # Validate configuration
        self.config.validate()
        
        # Validate subproblem name early if specified
        if self.config.subproblem and self.config.subproblem.name:
            from bundlechoice.subproblems.subproblem_registry import SUBPROBLEM_REGISTRY
            if self.config.subproblem.name not in SUBPROBLEM_REGISTRY:
                available = ', '.join(SUBPROBLEM_REGISTRY.keys())
                logger.warning(
                    f"⚠️  Config specifies unknown subproblem '{self.config.subproblem.name}'. "
                    f"Available: {available}"
                )

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
    
    # Backward compatibility alias
    load_config = configure

