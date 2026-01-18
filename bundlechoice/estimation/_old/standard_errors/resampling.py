from typing import Optional, Callable, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray
from .result import StandardErrorsResult
from bundlechoice.utils import get_logger
if TYPE_CHECKING:
    from bundlechoice.estimation.row_generation import RowGenerationManager
logger = get_logger(__name__)

class ResamplingMixin:

    def compute_bootstrap(self, theta_hat, solve_fn, num_bootstrap=100, beta_indices=None, seed=None):
        if beta_indices is None:
            beta_indices = np.arange(self.dimensions_cfg.n_features, dtype=np.int64)
        if self.comm_manager._is_root():
            lines = ['=' * 70, 'STANDARD ERRORS (BOOTSTRAP)', '=' * 70]
            lines.append(f'  Samples: {num_bootstrap}, Parameters: {len(beta_indices)}')
            logger.info('\n'.join(lines))
            if seed is not None:
                np.random.seed(seed)
            obs_bundles = self.data_manager.input_data['obs_bundle']
            agent_data = self.data_manager.input_data["id_data"]
            item_data = self.data_manager.input_data.get("item_data")
            N = self.dimensions_cfg.n_obs
        theta_boots = []
        for b in range(num_bootstrap):
            if self.comm_manager._is_root() and (b + 1) % 20 == 0:
                logger.info('  Bootstrap %d/%d...', b + 1, num_bootstrap)
            if self.comm_manager._is_root():
                idx = np.random.choice(N, size=N, replace=True)
                boot_data = {'obs_bundle': obs_bundles[idx], "id_data": {k: v[idx] for k, v in agent_data.items()}, 'errors': np.random.randn(N, self.dimensions_cfg.n_items)}
                if item_data is not None:
                    boot_data["item_data"] = item_data
            else:
                boot_data = None
            theta_b = solve_fn(boot_data)
            if self.comm_manager._is_root() and theta_b is not None:
                theta_boots.append(theta_b)
        if not self.comm_manager._is_root():
            return None
        return self._finalize_resampling_result(theta_hat, theta_boots, beta_indices, 'Bootstrap')

    def compute_subsampling(self, theta_hat, solve_fn, subsample_size=None, num_subsamples=100, beta_indices=None, seed=None):
        if beta_indices is None:
            beta_indices = np.arange(self.dimensions_cfg.n_features, dtype=np.int64)
        N = self.dimensions_cfg.n_obs
        b = min(subsample_size or int(N ** 0.7), N - 1)
        if self.comm_manager._is_root():
            lines = ['=' * 70, 'STANDARD ERRORS (SUBSAMPLING)', '=' * 70]
            lines.append(f'  Subsamples: {num_subsamples}, size: {b} (N={N})')
            logger.info('\n'.join(lines))
            if seed is not None:
                np.random.seed(seed)
            obs_bundles = self.data_manager.input_data['obs_bundle']
            agent_data = self.data_manager.input_data["id_data"]
            item_data = self.data_manager.input_data.get("item_data")
        theta_subs = []
        for s in range(num_subsamples):
            if self.comm_manager._is_root() and (s + 1) % 20 == 0:
                logger.info('  Subsample %d/%d...', s + 1, num_subsamples)
            if self.comm_manager._is_root():
                idx = np.random.choice(N, size=b, replace=False)
                sub_data = {'obs_bundle': obs_bundles[idx], "id_data": {k: v[idx] for k, v in agent_data.items()}, 'errors': np.random.randn(b, self.dimensions_cfg.n_items)}
                if item_data is not None:
                    sub_data["item_data"] = item_data
            else:
                sub_data = None
            theta_s = solve_fn(sub_data)
            if self.comm_manager._is_root() and theta_s is not None:
                theta_subs.append(theta_s)
        if not self.comm_manager._is_root():
            return None
        return self._finalize_resampling_result(theta_hat, theta_subs, beta_indices, 'Subsampling', scale_factor=np.sqrt(b / N))

    def compute_bayesian_bootstrap(self, row_generation, num_bootstrap=100, beta_indices=None, seed=None, warmstart='model', theta_hat=None, initial_estimation=False):
        if initial_estimation and theta_hat is None:
            raise ValueError('initial_estimation=True requires theta_hat to be provided')
        if beta_indices is None:
            beta_indices = np.arange(self.dimensions_cfg.n_features, dtype=np.int64)
        if self.comm_manager._is_root():
            lines = ['=' * 70, 'STANDARD ERRORS (BAYESIAN BOOTSTRAP)', '=' * 70]
            lines.append(f'  Samples: {num_bootstrap}, Parameters: {len(beta_indices)}')
            lines.append(f'  Warm-start: {warmstart}')
            lines.append(f'  Initial estimation: {initial_estimation}')
            logger.info('\n'.join(lines))
        N = self.dimensions_cfg.n_obs
        theta_boots = []
        constraints = None
        prev_theta = theta_hat.copy() if theta_hat is not None else None
        if seed is not None:
            np.random.seed(seed)
        for b in range(num_bootstrap):
            if self.comm_manager._is_root() and (b + 1) % 20 == 0:
                logger.info('  Bayesian bootstrap %d/%d...', b + 1, num_bootstrap)
            if self.comm_manager._is_root():
                weights = np.random.exponential(1.0, N)
                weights = weights / weights.mean()
            else:
                weights = None
            is_first = b == 0
            if is_first:
                result = row_generation.solve(obs_weights=weights, theta_init=prev_theta)
            elif warmstart == 'model':
                result = row_generation.solve_reuse_model(obs_weights=weights, strip_slack=False, reset_lp=False)
            elif warmstart == 'model_reset':
                result = row_generation.solve_reuse_model(obs_weights=weights, strip_slack=False, reset_lp=True)
            elif warmstart == 'model_strip':
                result = row_generation.solve_reuse_model(obs_weights=weights, strip_slack=True, reset_lp=False)
            elif warmstart == 'constraints':
                result = row_generation.solve(obs_weights=weights, initial_constraints=constraints, theta_init=prev_theta)
            elif warmstart == 'theta':
                result = row_generation.solve(obs_weights=weights, theta_init=prev_theta)
            else:
                result = row_generation.solve(obs_weights=weights)
            if self.comm_manager._is_root():
                theta_boots.append(result.theta_hat)
                if warmstart == 'constraints':
                    constraints = row_generation.get_constraints()
            if warmstart == 'theta':
                prev_theta = self.comm_manager.comm.bcast(result.theta_hat.copy() if self.comm_manager._is_root() else None, root=0)
        if not self.comm_manager._is_root():
            return None
        if initial_estimation:
            final_theta_hat = theta_hat
        else:
            final_theta_hat = np.mean(theta_boots, axis=0)
        return self._finalize_resampling_result(final_theta_hat, theta_boots, beta_indices, 'Bayesian Bootstrap')

    def _finalize_resampling_result(self, theta_hat, theta_samples, beta_indices, method_name, scale_factor=1.0):
        if len(theta_samples) < 10:
            logger.error('  Too few successful samples (%d)', len(theta_samples))
            return None
        theta_samples = np.array(theta_samples)
        se_all = scale_factor * np.std(theta_samples, axis=0, ddof=1)
        se = se_all[beta_indices]
        theta_beta = theta_hat[beta_indices]
        t_stats = np.where(se > 1e-16, theta_beta / se, np.nan)
        self._log_se_table(theta_hat, se, beta_indices, t_stats, method_name)
        n_params = len(beta_indices)
        return StandardErrorsResult(se=se, se_all=se_all, theta_beta=theta_beta, beta_indices=beta_indices, variance=np.diag(se ** 2), A_matrix=np.eye(n_params), B_matrix=np.eye(n_params), t_stats=t_stats)