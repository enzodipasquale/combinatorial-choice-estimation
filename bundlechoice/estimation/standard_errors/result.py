from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

@dataclass
class StandardErrorsResult:
    se: NDArray[np.float64]
    se_all: NDArray[np.float64]
    theta_beta: NDArray[np.float64]
    beta_indices: NDArray[np.int64]
    variance: NDArray[np.float64]
    A_matrix: NDArray[np.float64]
    B_matrix: NDArray[np.float64]
    t_stats: NDArray[np.float64]