import numpy as np
from .result import BayesianBootstrapResult
from .serial_bootstrap import SerialBootstrapMixin
from .distributed_bootstrap import DistributedBootstrapMixin
from combest.utils import get_logger

logger = get_logger(__name__)


class StandardErrorsManager(SerialBootstrapMixin, DistributedBootstrapMixin):

    def __init__(self, pt_estimation_manager):
        self.pt_estimation_manager = pt_estimation_manager
        self.comm_manager = pt_estimation_manager.comm_manager
        self.config = pt_estimation_manager.config
        self.data_manager = pt_estimation_manager.data_manager
        self.features_manager = pt_estimation_manager.features_manager
        self.subproblem_manager = pt_estimation_manager.subproblem_manager
        self.row_generation_manager = pt_estimation_manager.n_slack

        self.se_cfg = self.config.standard_errors
        self.dim = self.config.dimensions

    # ------------------------------------------------------------------
    # Weight generation (shared by both bootstrap methods)
    # ------------------------------------------------------------------

    def _gather_obs_quantity_on_root(self):
        full = self.comm_manager.Gatherv_by_row(
            self.data_manager.local_obs_quantity,
            row_counts=self.comm_manager.agent_counts)
        if self.comm_manager.is_root():
            return full[:self.dim.n_obs]
        return None

    def generate_weights_bayesian_bootstrap(self, seed, num_bootstrap):
        rng = np.random.default_rng(seed)
        obs_quantity = self._gather_obs_quantity_on_root()
        if self.comm_manager.is_root():
            weights = rng.gamma(obs_quantity[:, None], 1.0, (self.dim.n_obs, num_bootstrap))
            weights /= weights.mean(axis=0, keepdims=True)
            weights = np.tile(weights, (self.dim.n_simulations, 1))
        else:
            weights = None
        return weights

    def generate_weights_standard_bootstrap(self, seed, num_bootstrap):
        rng = np.random.default_rng(seed)
        obs_quantity = self._gather_obs_quantity_on_root()
        if self.comm_manager.is_root():
            n_total = int(obs_quantity.sum())
            pvals = obs_quantity / obs_quantity.sum()
            weights = rng.multinomial(n_total, pvals=pvals, size=num_bootstrap).T.astype(np.float64)
            weights = np.tile(weights, (self.dim.n_simulations, 1))
        else:
            weights = None
        return weights

    # ------------------------------------------------------------------
    # Statistics (shared by both bootstrap methods)
    # ------------------------------------------------------------------

    def create_bootstrap_result(self, theta_boots, u_boots, converged_mask=None, confidence=0.95):
        if not self.comm_manager.is_root():
            return None
        theta_boots = np.asarray(theta_boots)
        u_boots = np.asarray(u_boots)
        theta_for_stats = theta_boots if converged_mask is None else theta_boots[converged_mask]
        n_samples, _ = theta_for_stats.shape
        mean = theta_for_stats.mean(axis=0)
        se = theta_for_stats.std(axis=0, ddof=1)
        t_stats = np.where(se > 1e-16, mean / se, np.nan)
        alpha = 1 - confidence
        ci_lower = np.percentile(theta_for_stats, 100 * alpha / 2, axis=0)
        ci_upper = np.percentile(theta_for_stats, 100 * (1 - alpha / 2), axis=0)
        return BayesianBootstrapResult(
            mean=mean, se=se, t_stats=t_stats, n_samples=n_samples,
            ci_lower=ci_lower, ci_upper=ci_upper,
            confidence=confidence, samples=theta_boots, u_samples=u_boots,
        )
