from dataclasses import dataclass, field, fields
from typing import Optional, List, Any, Dict, Union, Callable
from pathlib import Path
import numpy as np
import yaml


# ============================================================================
# Configuration Mixins & Base Classes
# ============================================================================

class AutoUpdateMixin:
    """Mixin for automatic in-place config updates using dataclass reflection."""
    
    def update_in_place(self, other: Any) -> None:
        """Update config in place with values from other (merges nested configs)."""
        for field in fields(self):
            if hasattr(self, field.name) and hasattr(other, field.name):
                current_value = getattr(self, field.name)
                other_value = getattr(other, field.name)
                
                if other_value is not None:
                    if isinstance(current_value, dict):
                        current_value.update(other_value)
                    elif hasattr(current_value, 'update_in_place') and hasattr(other_value, 'update_in_place'):
                        current_value.update_in_place(other_value)
                    else:
                        setattr(self, field.name, other_value)


# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class DimensionsConfig(AutoUpdateMixin):
    """Problem dimensions: agents, items, features, simulations."""
    num_agents: Optional[int] = None
    num_items: Optional[int] = None
    num_features: Optional[int] = None
    num_simulations: int = 1
    feature_names: Optional[List[str]] = None
    # Feature structure for grouping (modular, fixed_effects, quadratic, etc.)
    _feature_groups: Dict[str, List[int]] = field(default_factory=dict)
    
    # Backward compatibility alias
    @property
    def num_simuls(self) -> int:
        """Backward compatibility alias for num_simulations."""
        return self.num_simulations
    
    # ============================================================================
    # Feature Naming API
    # ============================================================================
    
    def set_feature_names(self, names: List[str]) -> 'DimensionsConfig':
        """Set feature names. Length must match num_features."""
        if self.num_features is not None and len(names) != self.num_features:
            raise ValueError(f"Expected {self.num_features} names, got {len(names)}")
        self.feature_names = names
        return self
    
    def get_feature_name(self, index: int) -> str:
        """Get name for feature at index. Returns 'theta_{index}' if no names set."""
        if self.feature_names and 0 <= index < len(self.feature_names):
            return self.feature_names[index]
        return f"theta_{index}"
    
    def get_feature_index(self, name: str) -> int:
        """Get index for feature by name. Raises KeyError if not found."""
        if not self.feature_names:
            raise KeyError(f"No feature names set. Cannot find '{name}'")
        try:
            return self.feature_names.index(name)
        except ValueError:
            raise KeyError(f"Feature '{name}' not found in {self.feature_names}")
    
    def get_indices_by_pattern(self, pattern: str) -> List[int]:
        """Get indices for features matching glob pattern (e.g., 'FE_*')."""
        import fnmatch
        if not self.feature_names:
            return []
        return [i for i, name in enumerate(self.feature_names) if fnmatch.fnmatch(name, pattern)]
    
    def set_feature_groups(
        self,
        modular: Optional[List[str]] = None,
        fixed_effects: Optional[Union[int, List[str]]] = None,
        quadratic: Optional[List[str]] = None,
    ) -> 'DimensionsConfig':
        """
        Define feature groups for structured access.
        
        Args:
            modular: Names for modular features (first block)
            fixed_effects: Number of FE (auto-generates FE_0, FE_1, ...) or list of names
            quadratic: Names for quadratic features (last block)
        """
        names = []
        groups = {}
        
        if modular:
            groups['modular'] = list(range(len(names), len(names) + len(modular)))
            names.extend(modular)
        
        if fixed_effects:
            if isinstance(fixed_effects, int):
                fe_names = [f"FE_{i}" for i in range(fixed_effects)]
            else:
                fe_names = fixed_effects
            groups['fixed_effects'] = list(range(len(names), len(names) + len(fe_names)))
            names.extend(fe_names)
        
        if quadratic:
            groups['quadratic'] = list(range(len(names), len(names) + len(quadratic)))
            names.extend(quadratic)
        
        self.feature_names = names
        self._feature_groups = groups
        
        if self.num_features is None:
            self.num_features = len(names)
        elif self.num_features != len(names):
            raise ValueError(f"Feature groups define {len(names)} features, but num_features={self.num_features}")
        
        return self
    
    def get_group_indices(self, group: str) -> List[int]:
        """Get indices for a feature group (e.g., 'modular', 'fixed_effects', 'quadratic')."""
        return self._feature_groups.get(group, [])
    
    def get_structural_indices(self) -> List[int]:
        """Get indices for non-FE (structural) parameters."""
        fe_indices = set(self._feature_groups.get('fixed_effects', []))
        return [i for i in range(self.num_features or 0) if i not in fe_indices]


@dataclass
class SubproblemConfig(AutoUpdateMixin):
    """Subproblem solver configuration: algorithm name and settings."""
    name: Optional[str] = None
    settings: dict = field(default_factory=dict)


class BoundsManager:
    """
    Manages parameter bounds with name-based access.
    
    Example:
        bounds = BoundsManager(dims_cfg)
        bounds.set("bidder_elig_pop", lower=75)
        bounds.set("pop_distance", lower=400, upper=650)
        bounds.set_pattern("FE_*", lower=0, upper=1000)
    """
    
    def __init__(self, dimensions_cfg: 'DimensionsConfig'):
        self.dimensions_cfg = dimensions_cfg
        self._lower: Dict[int, float] = {}
        self._upper: Dict[int, float] = {}
    
    def set(
        self,
        name_or_index: Union[str, int],
        lower: Optional[float] = None,
        upper: Optional[float] = None,
    ) -> 'BoundsManager':
        """Set bounds for a parameter by name or index."""
        if isinstance(name_or_index, str):
            idx = self.dimensions_cfg.get_feature_index(name_or_index)
        else:
            idx = name_or_index
        
        if lower is not None:
            self._lower[idx] = lower
        if upper is not None:
            self._upper[idx] = upper
        return self
    
    def set_pattern(
        self,
        pattern: str,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
    ) -> 'BoundsManager':
        """Set bounds for all features matching a glob pattern (e.g., 'FE_*')."""
        for idx in self.dimensions_cfg.get_indices_by_pattern(pattern):
            self.set(idx, lower=lower, upper=upper)
        return self
    
    def get_arrays(
        self,
        num_features: int,
        default_lower: float = 0.0,
        default_upper: float = 1000.0,
    ) -> tuple:
        """Return (theta_lbs, theta_ubs) as numpy arrays."""
        lbs = np.full(num_features, default_lower)
        ubs = np.full(num_features, default_upper)
        for idx, val in self._lower.items():
            if idx < num_features:
                lbs[idx] = val
        for idx, val in self._upper.items():
            if idx < num_features:
                ubs[idx] = val
        return lbs, ubs


@dataclass
class RowGenerationConfig(AutoUpdateMixin):
    """Row generation solver parameters: tolerances, iteration limits, bounds."""
    tolerance_optimality: float = 1e-6
    max_slack_counter: float = float('inf')
    tol_row_generation: float = 0.0
    row_generation_decay: float = 0.0
    max_iters: float = float('inf')
    min_iters: int = 0
    gurobi_settings: dict = field(default_factory=dict)
    theta_ubs: Any = 1000
    theta_lbs: Any = None
    parameters_to_log: Optional[List[int]] = None
    subproblem_callback: Optional[Callable[[int, Any, Optional[Any]], None]] = None
    _bounds_manager: Optional[BoundsManager] = None
    
    def bounds(self, dimensions_cfg: 'DimensionsConfig') -> BoundsManager:
        """Get or create bounds manager for name-based bounds setting."""
        if self._bounds_manager is None:
            self._bounds_manager = BoundsManager(dimensions_cfg)
        return self._bounds_manager
    
    def apply_bounds(self, num_features: int) -> None:
        """Apply bounds from BoundsManager to theta_lbs/theta_ubs arrays."""
        if self._bounds_manager is not None:
            default_lower = 0.0 if self.theta_lbs is None else float(self.theta_lbs) if np.isscalar(self.theta_lbs) else 0.0
            default_upper = float(self.theta_ubs) if np.isscalar(self.theta_ubs) else 1000.0
            self.theta_lbs, self.theta_ubs = self._bounds_manager.get_arrays(
                num_features, default_lower, default_upper
            )


@dataclass
class EllipsoidConfig(AutoUpdateMixin):
    """Ellipsoid method solver parameters: convergence, radius, decay."""
    max_iterations: int = 1000
    num_iters: Optional[int] = None
    solver_precision: float = 1e-6
    tolerance: float = 1e-6
    initial_radius: float = 1.0
    decay_factor: float = 0.95
    min_volume: float = 1e-12
    verbose: bool = True


@dataclass
class StandardErrorsConfig(AutoUpdateMixin):
    """Standard errors computation parameters."""
    num_simulations: int = 10
    step_size: float = 1e-2
    seed: Optional[int] = None
    beta_indices: Optional[List[int]] = None  # Which parameters to report SE for (default: all)
    error_sigma: float = 1.0  # Standard deviation of errors (should match estimation errors)


@dataclass
class BundleChoiceConfig(AutoUpdateMixin):
    """Unified configuration container for all BundleChoice components."""
    dimensions: DimensionsConfig = field(default_factory=DimensionsConfig)
    subproblem: SubproblemConfig = field(default_factory=SubproblemConfig)
    row_generation: RowGenerationConfig = field(default_factory=RowGenerationConfig)
    ellipsoid: EllipsoidConfig = field(default_factory=EllipsoidConfig)
    standard_errors: StandardErrorsConfig = field(default_factory=StandardErrorsConfig)

    # ============================================================================
    # Factory Methods
    # ============================================================================

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> 'BundleChoiceConfig':
        """Create configuration from dictionary."""
        dims_cfg = cfg.get("dimensions", {}).copy()
        # Backward compatibility: convert num_simuls to num_simulations
        if "num_simuls" in dims_cfg and "num_simulations" not in dims_cfg:
            dims_cfg["num_simulations"] = dims_cfg.pop("num_simuls")
        # Handle _feature_groups (internal field)
        dims_cfg.pop("_feature_groups", None)
        return cls(
            dimensions=DimensionsConfig(**dims_cfg),
            subproblem=SubproblemConfig(**cfg.get("subproblem", {})),
            row_generation=RowGenerationConfig(**cfg.get("row_generation", {})),
            ellipsoid=EllipsoidConfig(**cfg.get("ellipsoid", {})),
            standard_errors=StandardErrorsConfig(**cfg.get("standard_errors", {})),
        )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'BundleChoiceConfig':
        """Create configuration from YAML file."""
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        return cls.from_dict(cfg) 

    @classmethod
    def load(cls, cfg: Union[str, Path, Dict[str, Any]]) -> 'BundleChoiceConfig':
        """Load configuration from YAML file or dictionary."""
        if isinstance(cfg, (str, Path)):
            return cls.from_yaml(str(cfg))
        elif isinstance(cfg, dict):
            return cls.from_dict(cfg)
        else:
            raise ValueError("cfg must be a string, Path, or a dictionary.")

    # ============================================================================
    # Validation
    # ============================================================================

    def validate(self) -> None:
        """Validate all configuration parameters. Raises ValueError if invalid."""
        if self.dimensions.num_agents is not None and self.dimensions.num_agents <= 0:
            raise ValueError("num_agents must be positive")
        if self.dimensions.num_items is not None and self.dimensions.num_items <= 0:
            raise ValueError("num_items must be positive")
        if self.dimensions.num_features is not None and self.dimensions.num_features <= 0:
            raise ValueError("num_features must be positive")
        if self.dimensions.num_simulations <= 0:
            raise ValueError("num_simulations must be positive")
        
        if self.row_generation.tolerance_optimality <= 0:
            raise ValueError("tolerance_optimality must be positive")
        if self.row_generation.max_iters <= 0:
            raise ValueError("max_iters must be positive")
        if self.row_generation.min_iters < 0:
            raise ValueError("min_iters must be non-negative")
        
        if self.ellipsoid.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.ellipsoid.solver_precision <= 0:
            raise ValueError("solver_precision must be positive")
        if self.ellipsoid.tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if self.ellipsoid.initial_radius <= 0:
            raise ValueError("initial_radius must be positive")
        if not 0 < self.ellipsoid.decay_factor < 1:
            raise ValueError("decay_factor must be between 0 and 1")
        if self.ellipsoid.min_volume <= 0:
            raise ValueError("min_volume must be positive")


