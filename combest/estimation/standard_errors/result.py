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
    u_samples: np.ndarray = None
    converged: np.ndarray = None  # bool array, per sample

    def welfare_decomposition(self, pt_result):
        S, N, xbar = pt_result.n_simulations, pt_result.n_obs, pt_result.xbar
        mask = self.converged
        surplus = self.u_samples[mask].reshape(-1, N, S).mean(axis=2).sum(axis=1)
        contributions = self.samples[mask] * xbar[None, :]
        entropy = surplus - contributions.sum(axis=1)
        return surplus, contributions, entropy
