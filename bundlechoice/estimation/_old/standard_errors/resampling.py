import numpy as np
from .result import StandardErrorsResult
from bundlechoice.utils import get_logger
logger = get_logger(__name__)

class ResamplingMixin:

    def compute_bootstrap(self, theta_hat, solve_fn, num_bootstrap=100, beta_indices=None, seed=None):
        if beta_indices is None:
            beta_indices = np.arange(self.dim.n_features, dtype=np.int64)
        if self.comm_manager._is_root():
            logger.info('=== BOOTSTRAP SE ===')
            logger.info(f'  Samples: {num_bootstrap}, Parameters: {len(beta_indices)}')
            if seed is not None:
                np.random.seed(seed)
            obs_bundles = self.data_manager.input_data['obs_bundle']
            agent_data = self.data_manager.input_data["id_data"]
            item_data = self.data_manager.input_data.get("item_data")
            N = self.dim.n_obs
            
        theta_boots = []
        for b in range(num_bootstrap):
            if self.comm_manager._is_root() and (b + 1) % 20 == 0:
                logger.info('  Bootstrap %d/%d...', b + 1, num_bootstrap)
            if self.comm_manager._is_root():
                idx = np.random.choice(N, size=N, replace=True)
                boot_data = {'obs_bundle': obs_bundles[idx], 
                             "id_data": {k: v[idx] for k, v in agent_data.items()}, 
                             'errors': np.random.randn(N, self.dim.n_items)}
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
            beta_indices = np.arange(self.dim.n_features, dtype=np.int64)
        N = self.dim.n_obs
        b = min(subsample_size or int(N ** 0.7), N - 1)
        
        if self.comm_manager._is_root():
            logger.info('=== SUBSAMPLING SE ===')
            logger.info(f'  Subsamples: {num_subsamples}, size: {b} (N={N})')
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
                sub_data = {'obs_bundle': obs_bundles[idx], 
                            "id_data": {k: v[idx] for k, v in agent_data.items()}, 
                            'errors': np.random.randn(b, self.dim.n_items)}
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

    def compute_bayesian_bootstrap(self, row_generation, num_bootstrap=100, beta_indices=None, seed=None, 
                                   warmstart='model', theta_hat=None, initial_estimation=False):
        if initial_estimation and theta_hat is None:
            raise ValueError('initial_estimation=True requires theta_hat')
        if beta_indices is None:
            beta_indices = np.arange(self.dim.n_features, dtype=np.int64)
        if self.comm_manager._is_root():
            logger.info('=== BAYESIAN BOOTSTRAP SE ===')
            logger.info(f'  Samples: {num_bootstrap}, Warmstart: {warmstart}')
            
        N = self.dim.n_obs
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
        final_theta_hat = theta_hat if initial_estimation else np.mean(theta_boots, axis=0)
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
        return StandardErrorsResult(se=se, se_all=se_all, theta_beta=theta_beta, beta_indices=beta_indices, 
                                    variance=np.diag(se ** 2), A_matrix=np.eye(n_params), 
                                    B_matrix=np.eye(n_params), t_stats=t_stats)

    def _log_se_table(self, theta_hat, se, beta_indices, t_stats, method):
        lines = [f'Standard Errors ({method}):']
        for i, idx in enumerate(beta_indices):
            lines.append(f'  theta[{idx}] = {theta_hat[idx]:.6f}, SE = {se[i]:.6f}, t = {t_stats[i]:.2f}')
        logger.info('\n'.join(lines))
