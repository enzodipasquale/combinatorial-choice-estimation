"""BundleChoice: Main orchestrator for bundle choice estimation."""

from typing import Optional, Dict, Any, Union, TYPE_CHECKING
import numpy as np
from mpi4py import MPI
from bundlechoice.config import BundleChoiceConfig
from bundlechoice.comm_manager import CommManager
from bundlechoice.utils import get_logger

if TYPE_CHECKING:
    from bundlechoice.data_manager import DataManager
    from bundlechoice.oracles_manager import OraclesManager
    from bundlechoice.subproblems.subproblem_manager import SubproblemManager
    from bundlechoice.estimation import RowGenerationManager, StandardErrorsManager, ColumnGenerationManager
    from bundlechoice.estimation.ellipsoid import EllipsoidManager

logger = get_logger(__name__)


class BundleChoice:
    """
    Main orchestrator for bundle choice estimation.
    
    Manages data loading, feature extraction, subproblem solving, and parameter
    estimation with MPI distribution. Components are lazily initialized on access.
    
    Example:
        bc = BundleChoice()
        bc.load_config(cfg)
        bc.data.load_and_scatter(input_data)
        bc.oracles.build_from_data()
        theta = bc.row_generation.solve()
    """
    
    def __init__(self) -> None:
        """Initialize empty BundleChoice instance."""
        self.config: Optional[BundleChoiceConfig] = None
        self.comm_manager: CommManager = CommManager(MPI.COMM_WORLD)
        self.data_manager: Optional['DataManager'] = None
        self.oracles_manager: Optional['OraclesManager'] = None
        self.subproblem_manager: Optional['SubproblemManager'] = None
        self.row_generation_manager: Optional['RowGenerationManager'] = None
        self.column_generation_manager: Optional['ColumnGenerationManager'] = None
        self.ellipsoid_manager: Optional['EllipsoidManager'] = None
        self.standard_errors_manager: Optional['StandardErrorsManager'] = None

    def _try_init_data_manager(self) -> 'DataManager':
        from bundlechoice._initialization import try_init_data_manager
        return try_init_data_manager(self)
        
    def _try_init_oracles_manager(self) -> 'OraclesManager':
        from bundlechoice._initialization import try_init_oracles_manager
        return try_init_oracles_manager(self)
        
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

    @property
    def data(self) -> 'DataManager':
        if self.data_manager is None:
            self._try_init_data_manager()
        return self.data_manager

    @property
    def oracles(self) -> 'OraclesManager':
        """Access the oracles manager (feature and error oracles)."""
        if self.oracles_manager is None:
            self._try_init_oracles_manager()
        return self.oracles_manager

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

    # Convenience properties for user-facing API
    @property
    def num_agents(self) -> int:
        return self.config.dimensions.num_agents

    @property
    def num_items(self) -> int:
        return self.config.dimensions.num_items

    @property
    def num_features(self) -> int:
        return self.config.dimensions.num_features

    @property
    def num_simulations(self) -> int:
        return self.config.dimensions.num_simulations

    @property
    def rank(self) -> int:
        return self.comm_manager.rank

    def is_root(self) -> bool:
        return self.comm_manager.is_root()

    def generate_observations(self, theta_true: np.ndarray) -> Optional[np.ndarray]:
        """Generate observed bundles from true parameters, then reload data."""
        obs_bundles = self.subproblems.init_and_solve(theta_true)
        
        local_errors_backup = None
        if self.data_manager.local_data is not None:
            local_errors_backup = self.data_manager.local_data.get("errors")
        
        if self.comm_manager.is_root():
            if self.data_manager.input_data is None:
                raise RuntimeError("Cannot generate observations without input_data")
            self.data_manager.input_data["obs_bundle"] = obs_bundles
            updated_data = self.data_manager.input_data
            has_errors_in_input = updated_data.get("errors") is not None
        else:
            updated_data = None
            has_errors_in_input = None
        
        has_errors_in_input = self.comm_manager.broadcast_from_root(has_errors_in_input, root=0)
        
        self.data.load_and_scatter(updated_data, errors_required=has_errors_in_input)
        
        if not has_errors_in_input and local_errors_backup is not None:
            self.data_manager.local_data["errors"] = local_errors_backup
        
        if (self.oracles_manager._features_oracle is not None and 
            self.oracles_manager._vectorized_features is not None):
            self.oracles.build_from_data()
        
        return obs_bundles
        
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
        
        if self.comm_manager.is_root():
            self._log_config()
        return self
    
    def _log_config(self) -> None:
        """Log configuration summary."""
        d = self.config.dimensions
        parts = []
        if d.num_agents is not None:
            parts.append(f"{d.num_agents} agents")
        if d.num_items is not None:
            parts.append(f"{d.num_items} items")
        if d.num_features is not None:
            parts.append(f"{d.num_features} features")
        if d.num_simulations > 1:
            parts.append(f"{d.num_simulations} simulations")
        
        algo = self.config.subproblem.name or "not set"
        workers = self.comm_manager.comm.Get_size()
        
        logger.info("Config: %s | algo=%s | %d MPI workers", ", ".join(parts), algo, workers)
