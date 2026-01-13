from dataclasses import dataclass, field, fields
from typing import Optional, List, Any, Dict, Union, Callable
from pathlib import Path
import yaml


class AutoUpdateMixin:
    """Mixin for automatic in-place config updates using dataclass reflection."""
    
    def update_in_place(self, other: Any) -> None:
        """Update config in place with values from other (merges nested configs)."""
        for f in fields(self):
            if hasattr(self, f.name) and hasattr(other, f.name):
                current_value = getattr(self, f.name)
                other_value = getattr(other, f.name)
                
                if other_value is not None:
                    if isinstance(current_value, dict):
                        current_value.update(other_value)
                    elif hasattr(current_value, 'update_in_place') and hasattr(other_value, 'update_in_place'):
                        current_value.update_in_place(other_value)
                    else:
                        setattr(self, f.name, other_value)


@dataclass
class DimensionsConfig(AutoUpdateMixin):
    """Problem dimensions: agents, items, features, simulations."""
    num_agents: Optional[int] = None
    num_items: Optional[int] = None
    num_features: Optional[int] = None
    num_simulations: int = 1
    feature_names: Optional[List[str]] = None
    _feature_groups: Dict[str, List[int]] = field(default_factory=dict)
    
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
        """Define feature groups for structured access."""
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
    beta_indices: Optional[List[int]] = None
    error_sigma: float = 1.0


@dataclass
class BundleChoiceConfig(AutoUpdateMixin):
    """Unified configuration container for all BundleChoice components."""
    dimensions: DimensionsConfig = field(default_factory=DimensionsConfig)
    subproblem: SubproblemConfig = field(default_factory=SubproblemConfig)
    row_generation: RowGenerationConfig = field(default_factory=RowGenerationConfig)
    ellipsoid: EllipsoidConfig = field(default_factory=EllipsoidConfig)
    standard_errors: StandardErrorsConfig = field(default_factory=StandardErrorsConfig)

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> 'BundleChoiceConfig':
        """Create configuration from dictionary."""
        dims_cfg = cfg.get("dimensions", {}).copy()
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
