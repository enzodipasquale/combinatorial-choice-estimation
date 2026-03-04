from dataclasses import dataclass
import numpy as np

@dataclass
class BayesianBootstrapResult:
    mean: np.ndarray
    se: np.ndarray
    t_stats: np.ndarray
    n_samples: int
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    confidence: float
    samples: np.ndarray = None
