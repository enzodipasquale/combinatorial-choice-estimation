import numpy as np
from .result import BayesianBootstrapResult
from bundlechoice.utils import get_logger
logger = get_logger(__name__)

class ResamplingMixin:
    def compute_bayesian_bootstrap(self, num_bootstrap=100, seed=None, verbose_row_gen=False):
        theta_boots = []
        rng = np.random.default_rng(seed)
        
        for b in range(num_bootstrap):
            if self.comm_manager._is_root():
                weights = rng.exponential(1.0, self.dim.n_obs)
                weights = weights / weights.sum()
                weights = np.tile(weights, self.dim.n_simulations)
            else:
                weights = None
            local_weights = self.comm_manager.Scatterv_by_row(weights, row_counts=self.data_manager.agent_counts)
            
            if b == 0:
                result = self.row_generation_manager.solve(local_obs_weights=local_weights, verbose=verbose_row_gen)
            else:
                self.row_generation_manager.update_objective_for_weights(local_weights)
                if self.comm_manager._is_root():
                    self.row_generation_manager.master_model.optimize()
                    result = self.row_generation_manager._create_result(
                        0, self.row_generation_manager.master_model,
                        self.row_generation_manager.master_variables[0].X,
                        self.row_generation_manager.cfg)
                else:
                    result = None
            
            if self.comm_manager._is_root():
                theta_boots.append(result.theta_hat)
        
        if not self.comm_manager._is_root():
            return None

        return self.compute_bootstrap_stats(theta_boots)

    def compute_bootstrap_stats(self, theta_boots, theta_hat=None, confidence=0.95):
        if not self.comm_manager._is_root():
            return None
        
        theta_boots = np.asarray(theta_boots)
        n_samples, _ = theta_boots.shape
        
        mean = theta_boots.mean(axis=0)
        se = theta_boots.std(axis=0, ddof=1)
        point_est = theta_hat if theta_hat is not None else mean
        t_stats = np.where(se > 1e-16, point_est / se, np.nan)
        alpha = 1 - confidence
        ci_lower = np.percentile(theta_boots, 100 * alpha / 2, axis=0)
        ci_upper = np.percentile(theta_boots, 100 * (1 - alpha / 2), axis=0)
        return BayesianBootstrapResult(
            mean=mean,
            se=se,
            t_stats=t_stats,
            n_samples=n_samples,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence=confidence,
        )

    # def compute_bootstrap(self, theta_hat, solve_fn, num_bootstrap=100, beta_indices=None, seed=None):
    #         if beta_indices is None:
    #             beta_indices = np.arange(self.dim.n_features, dtype=np.int64)
    #         if self.comm_manager._is_root():
    #             logger.info('=== BOOTSTRAP SE ===')
    #             logger.info(f'  Samples: {num_bootstrap}, Parameters: {len(beta_indices)}')
    #             if seed is not None:
    #                 np.random.seed(seed)
    #             obs_bundles = self.data_manager.input_data['obs_bundle']
    #             agent_data = self.data_manager.input_data["id_data"]
    #             item_data = self.data_manager.input_data.get("item_data")
    #             N = self.dim.n_obs
                
    #         theta_boots = []
    #         for b in range(num_bootstrap):
    #             if self.comm_manager._is_root() and (b + 1) % 20 == 0:
    #                 logger.info('  Bootstrap %d/%d...', b + 1, num_bootstrap)
    #             if self.comm_manager._is_root():
    #                 idx = np.random.choice(N, size=N, replace=True)
    #                 boot_data = {'obs_bundle': obs_bundles[idx], 
    #                             "id_data": {k: v[idx] for k, v in agent_data.items()}, 
    #                             'errors': np.random.randn(N, self.dim.n_items)}
    #                 if item_data is not None:
    #                     boot_data["item_data"] = item_data
    #             else:
    #                 boot_data = None
    #             theta_b = solve_fn(boot_data)
    #             if self.comm_manager._is_root() and theta_b is not None:
    #                 theta_boots.append(theta_b)
                    
    #         if not self.comm_manager._is_root():
    #             return None
    #         return self._finalize_resampling_result(theta_hat, theta_boots, beta_indices, 'Bootstrap')

    # def compute_subsampling(self, theta_hat, solve_fn, subsample_size=None, num_subsamples=100, beta_indices=None, seed=None):
    #     if beta_indices is None:
    #         beta_indices = np.arange(self.dim.n_features, dtype=np.int64)
    #     N = self.dim.n_obs
    #     b = min(subsample_size or int(N ** 0.7), N - 1)
        
    #     if self.comm_manager._is_root():
    #         logger.info('=== SUBSAMPLING SE ===')
    #         logger.info(f'  Subsamples: {num_subsamples}, size: {b} (N={N})')
    #         if seed is not None:
    #             np.random.seed(seed)
    #         obs_bundles = self.data_manager.input_data['obs_bundle']
    #         agent_data = self.data_manager.input_data["id_data"]
    #         item_data = self.data_manager.input_data.get("item_data")
            
    #     theta_subs = []
    #     for s in range(num_subsamples):
    #         if self.comm_manager._is_root() and (s + 1) % 20 == 0:
    #             logger.info('  Subsample %d/%d...', s + 1, num_subsamples)
    #         if self.comm_manager._is_root():
    #             idx = np.random.choice(N, size=b, replace=False)
    #             sub_data = {'obs_bundle': obs_bundles[idx], 
    #                         "id_data": {k: v[idx] for k, v in agent_data.items()}, 
    #                         'errors': np.random.randn(b, self.dim.n_items)}
    #             if item_data is not None:
    #                 sub_data["item_data"] = item_data
    #         else:
    #             sub_data = None
    #         theta_s = solve_fn(sub_data)
    #         if self.comm_manager._is_root() and theta_s is not None:
    #             theta_subs.append(theta_s)
                
    #     if not self.comm_manager._is_root():
    #         return None
    #     return self._finalize_resampling_result(theta_hat, theta_subs, beta_indices, 'Subsampling', scale_factor=np.sqrt(b / N))

