from dataclasses import dataclass, field, fields
from pathlib import Path
import yaml
import numpy as np

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

MAX_LABEL_WIDTH = 14

@dataclass
class DimensionsConfig(ConfigMixin):
    n_obs: int = None
    n_items: int = None
    n_covariates: int = None
    n_simulations: int = 1
    covariate_names: dict = None

    def __post_init__(self):
        self._build_labels()

    def update_in_place(self, other):
        super().update_in_place(other)
        self._build_labels()

    def _build_labels(self):
        if self.n_covariates is not None:
            names = self.covariate_names or {}
            self.covariate_labels = [names.get(i, f"θ[{i}]")[:MAX_LABEL_WIDTH]
                                     for i in range(self.n_covariates)]
            self.named_covariate_indices = list(names.keys())
            self.covariate_label_width = max((len(l) for l in self.covariate_labels), default=5)
        else:
            self.covariate_labels = None
            self.named_covariate_indices = None
            self.covariate_label_width = 5

    @property
    def n_agents(self):
        if self.n_obs is None or self.n_simulations is None:
            return None
        return self.n_simulations * self.n_obs

    def get_display_indices(self, parameters_to_log=None, max_default=5):
        return (parameters_to_log or self.named_covariate_indices
                or list(range(min(max_default, self.n_covariates))))

@dataclass
class SubproblemConfig(ConfigMixin):
    name: str = None
    gurobi_params: dict = field(default_factory=dict)

def theta_bounds_arrays(theta_bounds, n_covariates, default_lb=0, default_ub=10000):
    if not theta_bounds:
        return default_lb, default_ub

    lb = theta_bounds.get("lb", default_lb)
    ub = theta_bounds.get("ub", default_ub)

    theta_lbs = np.full(n_covariates, float(lb))
    theta_ubs = np.full(n_covariates, float(ub))

    for k, v in (theta_bounds.get("lbs") or {}).items():
        theta_lbs[int(k)] = float(v)
    for k, v in (theta_bounds.get("ubs") or {}).items():
        theta_ubs[int(k)] = float(v)

    return theta_lbs, theta_ubs

@dataclass
class RowGenerationConfig(ConfigMixin):
    max_slack_counter: float = float('inf')
    tolerance: float = 1e-6
    max_iters: float = float('inf')
    min_iters: int = 0
    master_gurobi_params: dict = field(default_factory=dict)
    theta_ubs: float = 1000
    theta_lbs: float = 0
    theta_bounds: dict = None
    parameters_to_log: list = None
    verbose: bool = True
    save_master_model_dir: str = None

    def theta_bounds_arrays(self, n_covariates: int):
        return theta_bounds_arrays(self.theta_bounds, n_covariates, self.theta_lbs, self.theta_ubs)

@dataclass
class EllipsoidConfig(ConfigMixin):
    max_iterations: int = 1000
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
    rowgen_tol: float = 1e-6
    rowgen_max_iters: int = 1000
    rowgen_min_iters: int = 0
    master_gurobi_params: dict = field(default_factory=dict)
    parameters_to_log: list = None
    theta_bounds: dict = None

    def theta_bounds_arrays(self, n_covariates: int):
        return theta_bounds_arrays(self.theta_bounds, n_covariates)

@dataclass
class ModelConfig(ConfigMixin):
    dimensions: DimensionsConfig = field(default_factory=DimensionsConfig)
    subproblem: SubproblemConfig = field(default_factory=SubproblemConfig)
    row_generation: RowGenerationConfig = field(default_factory=RowGenerationConfig)
    ellipsoid: EllipsoidConfig = field(default_factory=EllipsoidConfig)
    standard_errors: StandardErrorsConfig = field(default_factory=StandardErrorsConfig)

    @classmethod
    def from_dict(cls, cfg):
        return cls(
            dimensions=DimensionsConfig(**cfg.get('dimensions', {})),
            subproblem=SubproblemConfig(**cfg.get('subproblem', {})),
            row_generation=RowGenerationConfig(**cfg.get('row_generation', {})),
            ellipsoid=EllipsoidConfig(**cfg.get('ellipsoid', {})),
            standard_errors=StandardErrorsConfig(**cfg.get('standard_errors', {})),
        )

    @classmethod
    def from_yaml(cls, path):
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
        return cls.from_dict(cfg)
