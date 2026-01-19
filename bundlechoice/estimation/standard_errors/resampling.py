import numpy as np
from .result import BayesianBootstrapResult
from bundlechoice.utils import get_logger
logger = get_logger(__name__)
import time

class ResamplingMixin:
    def compute_bayesian_bootstrap(self, num_bootstrap=100, seed=None, verbose=False):
        theta_boots = []
        rng = np.random.default_rng(seed)
        
        for b in range(num_bootstrap):
            t0 = time.perf_counter()
            if self.comm_manager._is_root():
                weights = rng.exponential(1.0, self.dim.n_obs)
                weights /= weights.sum()
                weights = np.tile(weights, self.dim.n_simulations)
            else:
                weights = None
            local_weights = self.comm_manager.Scatterv_by_row(weights, row_counts=self.data_manager.agent_counts)
            
            if b == 0:
                self.row_generation_manager.solve(local_obs_weights=local_weights, verbose=verbose)
            
            self.row_generation_manager.update_objective_for_weights(local_weights)
            if self.comm_manager._is_root():
                self.row_generation_manager.master_model.optimize()
                theta_boots.append(self.row_generation_manager.master_variables[0].X.copy())
            iter_time = time.perf_counter() - t0
            print(iter_time)
        
        if not self.comm_manager._is_root():
            return None
        # print(np.array(theta_boots))

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

