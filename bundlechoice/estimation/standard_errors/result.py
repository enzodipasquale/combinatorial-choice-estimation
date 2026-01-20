from dataclasses import dataclass
import numpy as np

@dataclass
class StandardErrorsResultBase:
    mean: np.ndarray
    se: np.ndarray
    t_stats: np.ndarray
    n_samples: int

@dataclass
class BayesianBootstrapResult(StandardErrorsResultBase):
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    confidence: float
    samples: np.ndarray = None  

@dataclass
class SandwichResult(StandardErrorsResultBase):
    variance: np.ndarray  # full covariance matrix
    A_matrix: np.ndarray
    B_matrix: np.ndarray
    beta_indices: np.ndarray

