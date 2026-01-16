import numpy as np
from bundlechoice.utils import get_logger, suppress_output
from .result import EstimationResult

logger = get_logger(__name__)

class BaseEstimationManager:

    def __init__(self, comm_manager, config, data_manager, oracles_manager, subproblem_manager):
        self.comm_manager = comm_manager
        self.config = config
        self.data_manager = data_manager
        self.oracles_manager = oracles_manager
        self.subproblem_manager = subproblem_manager

        self.theta_val = None
        self.timing_stats = None

    @property
    def theta_obj_coef(self):
        return self.oracles_manager._features_at_obs_bundles

    def compute_obj_and_grad_at_root(self, theta):
    
        bundles = self.subproblem_manager.solve(theta)
        features = self.oracles_manager.features_oracle(bundles)
        utility = self.oracles_manager.utility_oracle(bundles, theta)
        
        features_sum = self.comm_manager.sum_row_and_Reduce(features)
        utility_sum = self.comm_manager.sum_row_and_Reduce(utility)
        if self.comm_manager._is_root():
            obj = utility_sum - (self.obs_features @ theta)
            grad = (features_sum - self.obs_features) / self.config.num_obs
            return obj, grad
        else:
            return None, None

    def compute_obj(self, theta):
        bundles = self.subproblem_manager.solve(theta)
        utility = self.oracles_manager.utility_oracle(bundles, theta)
        utility_sum = self.comm_manager.sum_row_and_Reduce(utility)
        if self.comm_manager._is_root():
            return utility_sum - (self.obs_features @ theta)
        else:
            return None
    
    def compute_grad(self, theta):
        bundles = self.subproblem_manager.solve(theta)
        features = self.oracles_manager.features_oracle(bundles)
        features_sum = self.comm_manager.sum_row_and_Reduce(features)
        if self.comm_manager._is_root():
            return (features_sum - self.obs_features) / self.config.num_obs
        else:
            return None


    def _create_result(self, theta, converged, num_iterations, final_objective=None, warnings=None, metadata=None):
        is_root = self.comm_manager._is_root()
        return EstimationResult(
            theta_hat=theta.copy(), converged=converged, num_iterations=num_iterations,
            final_objective=final_objective if is_root else None,
            timing=self.timing_stats if is_root else None,
            warnings=warnings or [] if is_root else [],
            metadata=metadata or {} if is_root else {})


    def _log_timing_summary(self, stats, obj_val=None, theta=None, header='SUMMARY'):
        if not self.comm_manager._is_root():
            return
        total, n_iters = stats.get('total_time', 0), stats.get('num_iterations', 0)
        pricing, master = stats.get('pricing_time', 0), stats.get('master_time', 0)
        other = total - pricing - master
        lines = [f'{"="*60}', header, f'{"="*60}']
        if obj_val is not None:
            lines.append(f'Objective: {obj_val:.6f}')
        if theta is not None:
            if len(theta) <= 10:
                lines.append(f'Theta: {np.array2string(theta, precision=4, suppress_small=True)}')
            else:
                lines.append(f'Theta: [{theta[:3]}...{theta[-3:]}] (dim={len(theta)}, range=[{theta.min():.4f}, {theta.max():.4f}])')
        lines.append(f'Iterations: {n_iters}, Time: {total:.2f}s')
        if total > 0:
            lines.append(f'  Pricing: {pricing:.2f}s ({100*pricing/total:.1f}%), Master: {master:.2f}s ({100*master/total:.1f}%), Other: {other:.2f}s')
        logger.info('\n'.join(lines))

    def log_parameter(self):
        cfg = self.config.row_generation
        if self.theta_val is None:
            return
        ids = cfg.parameters_to_log
        logger.info('Parameters: %s', np.round(self.theta_val[ids] if ids else self.theta_val, 3))
