import numpy as np
from .result import BayesianBootstrapResult
from bundlechoice.utils import get_logger
logger = get_logger(__name__)
import time

class ResamplingMixin:
    def compute_bayesian_bootstrap(self, num_bootstrap=100, seed=None, verbose=False):
        theta_boots = []
        rng = np.random.default_rng(seed)
        row_gen = self.row_generation_manager
        t0 =  time.perf_counter()
        for b in range(num_bootstrap):
            t1 = time.perf_counter()
            if self.comm_manager._is_root():
                weights = rng.exponential(1.0, self.dim.n_obs)
                weights /= weights.sum()
                weights = np.tile(weights, self.dim.n_simulations)
            else:
                weights = None
            local_weights = self.comm_manager.Scatterv_by_row(weights, row_counts=self.data_manager.agent_counts)
            init_master = True if b == 0 else False
            row_gen.solve(local_obs_weights=local_weights, verbose=True, init_subproblems = False, init_master = init_master)
            if self.comm_manager._is_root():
                theta_boots.append(row_gen.master_variables[0].X.copy())
            iter_time = time.perf_counter() - t1
            if self.comm_manager._is_root():
                print(len(row_gen.master_model.getConstrs()))
        if not self.comm_manager._is_root():
            return None
        print(time.perf_counter() - t0)
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

