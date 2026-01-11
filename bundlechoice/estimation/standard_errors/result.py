"""Standard errors result dataclass."""

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass
class StandardErrorsResult:
    """Result of standard errors computation."""
    se: NDArray[np.float64]           # SE for selected parameters
    se_all: NDArray[np.float64]       # SE for all parameters
    theta_beta: NDArray[np.float64]   # Parameter values for selected
    beta_indices: NDArray[np.int64]   # Which parameters were selected
    variance: NDArray[np.float64]     # Variance matrix
    A_matrix: NDArray[np.float64]     # Jacobian matrix (or identity for resampling)
    B_matrix: NDArray[np.float64]     # Outer product matrix (or identity for resampling)
    t_stats: NDArray[np.float64]      # t-statistics for selected parameters
