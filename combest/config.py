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

MAX_DISPLAY_WIDTH = 14

@dataclass
class DimensionsConfig(ConfigMixin):
    n_obs: int = None
    n_items: int = None
    n_covariates: int = None
    n_simulations: int = 1
    covariate_names: dict = None
    display_indices: list = None

    def __post_init__(self):
        self._build_labels()

    def update_in_place(self, other):
        super().update_in_place(other)
        self._build_labels()

    def _build_labels(self):
        if self.n_covariates is not None:
            names = self.covariate_names or {}
            self.covariate_labels = [names.get(i, f"θ[{i}]") for i in range(self.n_covariates)]
            self.named_covariate_indices = list(names.keys())
            self.covariate_label_width = min(
                max((len(l) for l in self.covariate_labels), default=5),
                MAX_DISPLAY_WIDTH,
            )
            if self.display_indices is None:
                self.display_indices = (self.named_covariate_indices
                                        or list(range(min(5, self.n_covariates))))
        else:
            self.covariate_labels = None
            self.named_covariate_indices = None
            self.covariate_label_width = 5

    @property
    def n_agents(self):
        if self.n_obs is None or self.n_simulations is None:
            return None
        return self.n_simulations * self.n_obs

    def display_label(self, i):
        return self.covariate_labels[i][:MAX_DISPLAY_WIDTH]

@dataclass
class SubproblemConfig(ConfigMixin):
    name: str = None
    gurobi_params: dict = field(default_factory=dict)

def _resolve_bound_key(k, name_to_idx):
    try:
        return int(k)
    except (ValueError, TypeError):
        if name_to_idx and k in name_to_idx:
            return name_to_idx[k]
        raise KeyError(f"Unknown covariate name in bounds: '{k}'")

def theta_bounds_arrays(theta_bounds, n_covariates, default_lb=0, default_ub=10000,
                        covariate_names=None):
    if not theta_bounds:
        return default_lb, default_ub

    name_to_idx = {v: k for k, v in (covariate_names or {}).items()}

    lb = theta_bounds.get("lb", default_lb)
    ub = theta_bounds.get("ub", default_ub)

    theta_lbs = np.full(n_covariates, float(lb))
    theta_ubs = np.full(n_covariates, float(ub))

    for k, v in (theta_bounds.get("lbs") or {}).items():
        try:
            theta_lbs[_resolve_bound_key(k, name_to_idx)] = float(v)
        except KeyError:
            pass  # skip bounds for covariates not in the model
    for k, v in (theta_bounds.get("ubs") or {}).items():
        try:
            theta_ubs[_resolve_bound_key(k, name_to_idx)] = float(v)
        except KeyError:
            pass  # skip bounds for covariates not in the model

    return theta_lbs, theta_ubs

@dataclass
class QuadraticPenaltyConfig(ConfigMixin):
    initial_weight: float = 1.0
    decay_iterations: int = 50

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

    quadratic_penalty: QuadraticPenaltyConfig = None

    verbose: bool = True
    save_master_model_dir: str = None

    def __post_init__(self):
        if isinstance(self.quadratic_penalty, dict):
            self.quadratic_penalty = QuadraticPenaltyConfig(**self.quadratic_penalty)

    def theta_bounds_arrays(self, n_covariates: int, covariate_names=None):
        return theta_bounds_arrays(self.theta_bounds, n_covariates, self.theta_lbs, self.theta_ubs,
                                   covariate_names)

@dataclass
class EllipsoidConfig(ConfigMixin):
    max_iterations: int = 1000
    solver_precision: float = 1e-06
    initial_radius: float = 1.0
    verbose: bool = True

@dataclass
class StandardErrorsConfig(ConfigMixin):
    n_simulations: int = 10
    seed: int = None
    beta_indices: list = None
    error_sigma: float = 1.0
    rowgen_tol: float = 1e-3
    rowgen_max_iters: int = 100
    rowgen_min_iters: int = 0
    master_gurobi_params: dict = field(default_factory=dict)

    theta_bounds: dict = None

    def theta_bounds_arrays(self, n_covariates: int, covariate_names=None):
        return theta_bounds_arrays(self.theta_bounds, n_covariates, covariate_names=covariate_names)

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
