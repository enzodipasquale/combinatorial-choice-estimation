from dataclasses import dataclass
import numpy as np

@dataclass
class StandardErrorsResult:
    se: np.ndarray
    se_all: np.ndarray
    theta_beta: np.ndarray
    beta_indices: np.ndarray
    variance: np.ndarray
    A_matrix: np.ndarray
    B_matrix: np.ndarray
    t_stats: np.ndarray
