from dataclasses import dataclass, field, fields
from typing import Optional, List, Any, Dict, Union, Callable
from pathlib import Path
import yaml

class ConfigMixin:

    def update_in_place(self, other):
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
class DimensionsConfig(ConfigMixin):
    num_obs: Optional[int] = None
    num_items: Optional[int] = None
    num_features: Optional[int] = None
    num_simulations: int = 1
    feature_names: Optional[List[str]] = None

    @property
    def num_agents(self):
        if self.num_obs is None or self.num_simulations is None:
            return None
        return self.num_simulations * self.num_obs
        
@dataclass
class SubproblemConfig(ConfigMixin):
    name: Optional[str] = None
    settings: dict = field(default_factory=dict)

@dataclass
class RowGenerationConfig(ConfigMixin):
    tolerance_optimality: float = 1e-06
    max_slack_counter: float = float('inf')
    tol_row_generation: float = 0.0
    row_generation_decay: float = 0.0
    max_iters: float = float('inf')
    min_iters: int = 0
    gurobi_settings: dict = field(default_factory=dict)
    theta_ubs: Any = 1000
    theta_lbs: Any = None
    parameters_to_log: Optional[List[int]] = None
    verbose: bool = True
    subproblem_callback: Optional[Callable[[int, Any, Optional[Any]], None]] = None
    master_init_callback: Optional[Callable[[Any, Any, Any], None]] = None

@dataclass
class EllipsoidConfig(ConfigMixin):
    max_iterations: int = 1000
    num_iters: Optional[int] = None
    solver_precision: float = 1e-06
    initial_radius: float = 1.0
    verbose: bool = True

@dataclass
class StandardErrorsConfig(ConfigMixin):
    num_simulations: int = 10
    step_size: float = 0.01
    seed: Optional[int] = None
    beta_indices: Optional[List[int]] = None
    error_sigma: float = 1.0

@dataclass
class BundleChoiceConfig(ConfigMixin):
    dimensions: DimensionsConfig = field(default_factory=DimensionsConfig)
    subproblem: SubproblemConfig = field(default_factory=SubproblemConfig)
    row_generation: RowGenerationConfig = field(default_factory=RowGenerationConfig)
    ellipsoid: EllipsoidConfig = field(default_factory=EllipsoidConfig)
    standard_errors: StandardErrorsConfig = field(default_factory=StandardErrorsConfig)

    @classmethod
    def from_dict(cls, cfg):
        return cls(dimensions=DimensionsConfig(**cfg.get('dimensions', {})), subproblem=SubproblemConfig(**cfg.get('subproblem', {})), row_generation=RowGenerationConfig(**cfg.get('row_generation', {})), ellipsoid=EllipsoidConfig(**cfg.get('ellipsoid', {})), standard_errors=StandardErrorsConfig(**cfg.get('standard_errors', {})))

    @classmethod
    def from_yaml(cls, path):
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
        return cls.from_dict(cfg)