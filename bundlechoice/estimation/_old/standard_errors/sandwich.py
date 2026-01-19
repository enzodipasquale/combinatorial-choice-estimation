import numpy as np
from mpi4py import MPI
from .result import StandardErrorsResult
from bundlechoice.utils import get_logger
logger = get_logger(__name__)

def compute_adaptive_step_size(theta_k, base_step=0.0001):
    eps = np.finfo(np.float64).eps
    min_step = np.sqrt(eps)
    scale = max(1.0, abs(theta_k))
    h = base_step * scale
    h = max(h, min_step * scale)
    if abs(theta_k) > 0:
        h = min(h, 0.1 * abs(theta_k))
    return h

class SandwichMixin:

    def compute(self, theta_hat, n_simulations=None, step_size=None, beta_indices=None, seed=None, 
                optimize_for_subset=True, error_sigma=None):
        n_simulations = n_simulations or self.se_cfg.n_simulations
        step_size = step_size or self.se_cfg.step_size
        seed = seed if seed is not None else self.se_cfg.seed
        error_sigma = error_sigma if error_sigma is not None else self.se_cfg.error_sigma
        theta_hat = self.comm_manager.comm.bcast(theta_hat, root=0)
        if beta_indices is None:
            beta_indices = np.arange(self.dim.n_features, dtype=np.int64)
        beta_indices = self.comm_manager.comm.bcast(beta_indices, root=0)
        errors_all_sims = self._generate_se_errors(n_simulations, seed, error_sigma)
        self._cache_obs_features()
        
        is_subset = len(beta_indices) < self.dim.n_features
        use_subset_opt = optimize_for_subset and is_subset
        if self.comm_manager._is_root():
            logger.info('=== SANDWICH SE ===')
            logger.info(f'  Simulations: {n_simulations}, Step: {step_size}')
            if use_subset_opt:
                logger.info(f'  Computing subset: {len(beta_indices)} params')
                
        if use_subset_opt:
            B = self._compute_B_matrix_subset(theta_hat, errors_all_sims, beta_indices)
            A = self._compute_A_matrix_subset(theta_hat, errors_all_sims, step_size, beta_indices)
        else:
            B = self._compute_B_matrix(theta_hat, errors_all_sims)
            A = self._compute_A_matrix(theta_hat, errors_all_sims, step_size)
            
        self.comm_manager.comm.Barrier()
        if not self.comm_manager._is_root():
            return None
        return self._finalize_sandwich(theta_hat, A, B, beta_indices, use_subset_opt)

    def compute_B_inverse(self, theta_hat, n_simulations=None, beta_indices=None, seed=None, error_sigma=None):
        n_simulations = n_simulations or self.se_cfg.n_simulations
        seed = seed if seed is not None else self.se_cfg.seed
        error_sigma = error_sigma if error_sigma is not None else self.se_cfg.error_sigma
        theta_hat = self.comm_manager.comm.bcast(theta_hat, root=0)
        if beta_indices is None:
            beta_indices = np.arange(self.dim.n_features, dtype=np.int64)
        beta_indices = self.comm_manager.comm.bcast(beta_indices, root=0)
        errors_all_sims = self._generate_se_errors(n_simulations, seed, error_sigma)
        self._cache_obs_features()
        
        is_subset = len(beta_indices) < self.dim.n_features
        if self.comm_manager._is_root():
            logger.info('=== B-INVERSE SE ===')
            logger.info(f'  Simulations: {n_simulations}, Parameters: {len(beta_indices)}')
            
        if is_subset:
            B = self._compute_B_matrix_subset(theta_hat, errors_all_sims, beta_indices)
        else:
            B = self._compute_B_matrix(theta_hat, errors_all_sims)
            
        self.comm_manager.comm.Barrier()
        if not self.comm_manager._is_root():
            return None
            
        B_cond = np.linalg.cond(B)
        logger.info('  B matrix: cond=%.2e', B_cond)
        if not np.isfinite(B_cond) or B_cond > 1e+16:
            logger.error('  B matrix is singular')
            return None
        try:
            B_inv = np.linalg.solve(B, np.eye(len(beta_indices)))
        except np.linalg.LinAlgError:
            logger.error('  B matrix singular')
            return None
            
        V = B_inv / self.dim.n_obs
        se = np.sqrt(np.maximum(np.diag(V), 0))
        theta_beta = theta_hat[beta_indices]
        t_stats = np.where(se > 1e-16, theta_beta / se, np.nan)
        self._log_se_table(theta_hat, se, beta_indices, t_stats, 'B-inverse')
        return StandardErrorsResult(se=se, se_all=se, theta_beta=theta_beta, beta_indices=beta_indices, 
                                    variance=V, A_matrix=np.eye(len(beta_indices)), B_matrix=B, t_stats=t_stats)

    def _finalize_sandwich(self, theta_hat, A, B, beta_indices, is_subset):
        A_cond, B_cond = np.linalg.cond(A), np.linalg.cond(B)
        logger.info('  A matrix: cond=%.2e, B matrix: cond=%.2e', A_cond, B_cond)
        if not np.isfinite(A_cond) or A_cond > 1e+16:
            logger.error('  A matrix singular/ill-conditioned')
            return None
        try:
            A_inv = np.linalg.solve(A, np.eye(len(beta_indices) if is_subset else self.dim.n_features))
        except np.linalg.LinAlgError:
            logger.error('  A matrix singular')
            return None
            
        V = 1.0 / self.dim.n_obs * (A_inv @ B @ A_inv.T)
        diag_V = np.diag(V)
        if np.any(diag_V < 0):
            logger.warning('  %d negative variances', np.sum(diag_V < 0))
        se_all = np.sqrt(np.maximum(diag_V, 0))
        se = se_all if is_subset else se_all[beta_indices]
        theta_beta = theta_hat[beta_indices]
        t_stats = np.where(se > 1e-16, theta_beta / se, np.nan)
        self._log_se_table(theta_hat, se, beta_indices, t_stats, 'Sandwich')
        return StandardErrorsResult(se=se, se_all=se_all, theta_beta=theta_beta, beta_indices=beta_indices, 
                                    variance=V, A_matrix=A, B_matrix=B, t_stats=t_stats)

    def _log_se_table(self, theta_hat, se, beta_indices, t_stats, method):
        lines = [f'Standard Errors ({method}):']
        for i, idx in enumerate(beta_indices):
            lines.append(f'  theta[{idx}] = {theta_hat[idx]:.6f}, SE = {se[i]:.6f}, t = {t_stats[i]:.2f}')
        logger.info('\n'.join(lines))

    def _compute_B_matrix(self, theta, errors_all_sims, beta_indices=None):
        num_sims = len(errors_all_sims)
        dim = len(beta_indices) if beta_indices is not None else self.dim.n_features
        if self.comm_manager._is_root():
            logger.info('Computing B matrix (%d x %d)...', dim, dim)
        all_features = []
        for s in range(num_sims):
            if self.comm_manager._is_root():
                logger.info('  Simulation %d/%d...', s + 1, num_sims)
            self.data_manager.update_errors(errors_all_sims[s] if self.comm_manager._is_root() else None)
            local_bundles = self._solve_local_or_empty(theta)
            features_local = self.oracles_manager.features_oracle(local_bundles)
            features = self.comm_manager.Gatherv_by_row(features_local, row_counts=self.data_manager.agent_counts)
            if self.comm_manager._is_root():
                all_features.append(features)
                
        self.comm_manager.comm.Barrier()
        if self.comm_manager._is_root():
            features_all = np.stack(all_features, axis=0)
            if beta_indices is not None:
                g_i = features_all.mean(axis=0)[:, beta_indices] - self._obs_features[:, beta_indices]
            else:
                g_i = features_all.mean(axis=0) - self._obs_features
            B = g_i.T @ g_i / self.dim.n_obs
            logger.info('  B matrix: cond=%.2e', np.linalg.cond(B))
            return B
        return None

    def _compute_B_matrix_subset(self, theta, errors_all_sims, beta_indices):
        return self._compute_B_matrix(theta, errors_all_sims, beta_indices)

    def _compute_A_matrix(self, theta, errors_all_sims, step_size, beta_indices=None):
        is_subset = beta_indices is not None
        indices = beta_indices if is_subset else np.arange(self.dim.n_features)
        dim = len(indices)
        if self.comm_manager._is_root():
            logger.info('Computing A matrix (%d x %d)...', dim, dim)
            A = np.zeros((dim, dim))
        else:
            A = None
            
        for col_idx, k in enumerate(indices):
            if self.comm_manager._is_root():
                logger.info('  Column %d/%d...', col_idx + 1, dim)
            h_k = compute_adaptive_step_size(theta[k], step_size)
            theta_plus, theta_minus = theta.copy(), theta.copy()
            theta_plus[k] += h_k
            theta_minus[k] -= h_k
            g_plus = self._compute_avg_subgradient(theta_plus, errors_all_sims, beta_indices)
            g_minus = self._compute_avg_subgradient(theta_minus, errors_all_sims, beta_indices)
            if self.comm_manager._is_root():
                A[:, col_idx] = (g_plus - g_minus) / (2 * h_k)
                
        self.comm_manager.comm.Barrier()
        return A

    def _compute_A_matrix_subset(self, theta, errors_all_sims, step_size, beta_indices):
        return self._compute_A_matrix(theta, errors_all_sims, step_size, beta_indices)

    def _compute_avg_subgradient(self, theta, errors_all_sims, beta_indices=None):
        num_sims = len(errors_all_sims)
        is_subset = beta_indices is not None
        dim = len(beta_indices) if is_subset else self.dim.n_features
        
        if is_subset:
            cache_key = tuple(beta_indices)
            if self._mean_obs_subset is None:
                self._mean_obs_subset = {}
            if cache_key not in self._mean_obs_subset:
                obs_local = self.data_manager.local_obs_bundles
                obs_feat = self.oracles_manager.features_oracle(obs_local)
                obs_sum = obs_feat[:, beta_indices].sum(axis=0) if obs_feat.size else np.zeros(dim)
                obs_sum_global = np.zeros(dim)
                self.comm_manager.comm.Allreduce(obs_sum, obs_sum_global, op=MPI.SUM)
                self._mean_obs_subset[cache_key] = obs_sum_global / self.dim.n_obs
            mean_obs = self._mean_obs_subset[cache_key]
        else:
            if self._mean_obs_full is None:
                self._cache_mean_obs_full()
            mean_obs = self._mean_obs_full
            
        sim_sum_local = np.zeros(dim)
        for s in range(num_sims):
            self.data_manager.update_errors(errors_all_sims[s] if self.comm_manager._is_root() else None)
            local_bundles = self._solve_local_or_empty(theta)
            feat_local = self.oracles_manager.features_oracle(local_bundles)
            if feat_local.size:
                if is_subset:
                    sim_sum_local += feat_local[:, beta_indices].sum(axis=0)
                else:
                    sim_sum_local += feat_local.sum(axis=0)
                    
        sim_sum_global = np.zeros(dim)
        self.comm_manager.comm.Allreduce(sim_sum_local, sim_sum_global, op=MPI.SUM)
        mean_sim = sim_sum_global / num_sims / self.dim.n_obs
        return mean_sim - mean_obs if self.comm_manager._is_root() else None

    def _solve_local_or_empty(self, theta):
        if self.data_manager.num_local_agent > 0:
            return self.subproblem_manager.solve_subproblems(theta)
        return np.empty((0, self.dim.n_items), dtype=bool)

    def _cache_obs_features(self):
        if self._obs_features is None:
            obs_bundles = self.data_manager.local_obs_bundles
            features_local = self.oracles_manager.features_oracle(obs_bundles)
            self._obs_features = self.comm_manager.Gatherv_by_row(features_local, row_counts=self.data_manager.agent_counts)

    def _cache_mean_obs_full(self):
        K = self.dim.n_features
        obs_local = self.data_manager.local_obs_bundles
        obs_feat = self.oracles_manager.features_oracle(obs_local)
        obs_sum = obs_feat.sum(axis=0) if obs_feat.size else np.zeros(K)
        obs_sum_global = np.zeros(K)
        self.comm_manager.comm.Allreduce(obs_sum, obs_sum_global, op=MPI.SUM)
        self._mean_obs_full = obs_sum_global / self.dim.n_obs

    def _generate_se_errors(self, n_simulations, seed, error_sigma=1.0):
        if self.comm_manager._is_root():
            if seed is not None:
                np.random.seed(seed)
            errors = error_sigma * np.random.randn(n_simulations, self.dim.n_obs, self.dim.n_items)
        else:
            errors = None
        return self.comm_manager.comm.bcast(errors, root=0)
