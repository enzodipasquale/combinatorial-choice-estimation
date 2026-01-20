import numpy as np
from .result import BayesianBootstrapResult
from bundlechoice.utils import get_logger
logger = get_logger(__name__)
import time

class ResamplingMixin:
    def compute_bayesian_bootstrap(self, num_bootstrap=100, seed=None, verbose=False):
        theta_boots = []
        self.bootstrap_history = {}
        self.verbose = verbose
        rng = np.random.default_rng(seed)
        row_gen = self.row_generation_manager
        t0 = time.perf_counter()
        
        for b in range(num_bootstrap):
            t_boot = time.perf_counter()
            if self.comm_manager._is_root():
                weights = rng.exponential(1.0, self.dim.n_obs)
                weights /= weights.mean()
                weights = np.tile(weights, self.dim.n_simulations)
            else:
                weights = None
            local_weights = self.comm_manager.Scatterv_by_row(weights, row_counts=self.data_manager.agent_counts)
            initialize_master = True if b == 0 else False
            result = row_gen.solve(local_obs_weights=local_weights, verbose=False, initialize_subproblems=False, initialize_master=initialize_master)
            boot_time = time.perf_counter() - t_boot
            
            if self.comm_manager._is_root():
                theta_boots.append(row_gen.master_variables[0].X.copy())
                self._update_bootstrap_info(b, time=boot_time, iterations=result.num_iterations,
                                           objective=result.final_objective, converged=result.converged)
            self._log_bootstrap_iteration(b, row_gen.theta_iter)
        
        total_time = time.perf_counter() - t0
        if not self.comm_manager._is_root():
            return None
        
        stats_result = self.compute_bootstrap_stats(theta_boots)
        self._log_bootstrap_summary(num_bootstrap, total_time, stats_result)
        
        return stats_result

    def _update_bootstrap_info(self, bootstrap_iter, **kwargs):
        if self.comm_manager._is_root():
            if bootstrap_iter not in self.bootstrap_history:
                self.bootstrap_history[bootstrap_iter] = {}
            self.bootstrap_history[bootstrap_iter].update(kwargs)

    def _log_bootstrap_iteration(self, bootstrap_iter, theta):
        if not self.comm_manager._is_root() or not self.verbose:
            return
        if bootstrap_iter not in self.bootstrap_history:
            return
        info = self.bootstrap_history[bootstrap_iter]
        
        if self.config.row_generation.parameters_to_log is not None:
            param_indices = self.config.row_generation.parameters_to_log
        else:
            param_indices = list(range(min(5, self.dim.n_features)))
        
        if bootstrap_iter % 80 == 0:
            param_header = ', '.join(f'θ[{i}]' for i in param_indices)
            logger.info(f"{'Boot':>5} | {'Time (s)':>9} | {'RG Iter':>7} | {'Objective':>12} | Parameters ({param_header})")
            logger.info("-"*100)
        
        param_vals = ', '.join(f'{theta[i]:.5f}' for i in param_indices)
        logger.info(f"{bootstrap_iter:>5} | {info['time']:>9.3f} | {info['iterations']:>7} | {info['objective']:>12.5f} | ({param_vals})")

    def _log_bootstrap_summary(self, n_bootstrap, total_time, result):
        if not self.comm_manager._is_root() or not self.verbose:
            return
        
        self.row_generation_manager._log_instance_summary()
        
        if self.config.row_generation.parameters_to_log is not None:
            param_indices = self.config.row_generation.parameters_to_log
        else:
            param_indices = list(range(min(5, self.dim.n_features)))
        
    
        logger.info("-"*100)
        logger.info(f"Bayesian Bootstrap Summary: {n_bootstrap} samples in {total_time:.1f}s")
        logger.info(f"{'Param':>8} | {'Mean':>12} | {'SE':>12} | {'t-stat':>10}")
        logger.info("-"*100)
        for i in param_indices:
            logger.info(f"θ[{i:>3}] | {result.mean[i]:>12.5f} | {result.se[i]:>12.5f} | {result.t_stats[i]:>10.2f}")
        logger.info("-"*100)

 

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

