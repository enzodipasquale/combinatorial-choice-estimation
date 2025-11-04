from dataclasses import dataclass, field, fields
from typing import Optional, List, Any, Dict, Union
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
    num_simuls: int = 1


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
class BundleChoiceConfig(AutoUpdateMixin):
    """Unified configuration container for all BundleChoice components."""
    dimensions: DimensionsConfig = field(default_factory=DimensionsConfig)
    subproblem: SubproblemConfig = field(default_factory=SubproblemConfig)
    row_generation: RowGenerationConfig = field(default_factory=RowGenerationConfig)
    ellipsoid: EllipsoidConfig = field(default_factory=EllipsoidConfig)

    # ============================================================================
    # Factory Methods
    # ============================================================================

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> 'BundleChoiceConfig':
        """Create configuration from dictionary."""
        return cls(
            dimensions=DimensionsConfig(**cfg.get("dimensions", {})),
            subproblem=SubproblemConfig(**cfg.get("subproblem", {})),
            row_generation=RowGenerationConfig(**cfg.get("row_generation", {})),
            ellipsoid=EllipsoidConfig(**cfg.get("ellipsoid", {})),
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
        if self.dimensions.num_simuls <= 0:
            raise ValueError("num_simuls must be positive")
        
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


