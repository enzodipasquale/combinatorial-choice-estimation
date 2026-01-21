from dataclasses import dataclass, field, fields
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
    n_obs: int = None
    n_items: int = None
    n_features: int = None
    n_simulations: int = 1
    feature_names: list = None

    @property
    def num_agents(self):
        if self.n_obs is None or self.n_simulations is None:
            return None
        return self.n_simulations * self.n_obs
        
        
@dataclass
class SubproblemConfig(ConfigMixin):
    name: str = None
    settings: dict = field(default_factory=dict)

@dataclass
class RowGenerationConfig(ConfigMixin):
    max_slack_counter: float = float('inf')
    tol_row_generation: float = 1e-6
    row_generation_decay: float = 1.0
    max_iters: float = float('inf')
    min_iters: int = 0
    gurobi_settings: dict = field(default_factory=dict)
    theta_ubs: float = 1000
    theta_lbs: float = 0
    parameters_to_log: list = None
    verbose: bool = True

@dataclass
class EllipsoidConfig(ConfigMixin):
    max_iterations: int = 1000
    num_iters: int = None
    solver_precision: float = 1e-06
    initial_radius: float = 1.0
    verbose: bool = True

@dataclass
class StandardErrorsConfig(ConfigMixin):
    n_simulations: int = 10
    step_size: float = 0.01
    seed: int = None
    beta_indices: list = None
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
