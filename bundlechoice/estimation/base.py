import numpy as np
from bundlechoice.utils import get_logger
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
        return self._compute_theta_obj_coef()

    def _compute_theta_obj_coef(self, local_obs_weights = None):
        if local_obs_weights is None:
            local_obs_weights = self.data_manager.local_data["id_data"].get("obs_weights", None)
            local_obs_weights = local_obs_weights if local_obs_weights is not None else np.ones(self.data_manager.num_local_agent)
        local_obs_features = self.oracles_manager.features_oracle(self.data_manager.local_obs_bundles)
        return self.comm_manager.sum_row_andReduce(-local_obs_weights[:, None] * local_obs_features)
    
    def _compute_u_obj_weights(self, local_obs_weights = None):
        if local_obs_weights is None:
            local_obs_weights = self.data_manager.local_data["id_data"].get("obs_weights", None)
            local_obs_weights = local_obs_weights if local_obs_weights is not None else np.ones(self.data_manager.num_local_agent)
        

    def compute_obj_and_grad_at_root(self, theta, local_obs_weights = None):
    
        bundles = self.subproblem_manager.solve_subproblems(theta)
        features = self.oracles_manager.features_oracle(bundles)
        utility = self.oracles_manager.utility_oracle(bundles, theta)
        
        features_sum = self.comm_manager.sum_row_andReduce(local_obs_weights[:, None] * features)
        utility_sum = self.comm_manager.sum_row_andReduce(local_obs_weights * utility)
        _theta_obj_coef = self._compute_theta_obj_coef(local_obs_weights)

        if self.comm_manager._is_root():
            obj = utility_sum - (_theta_obj_coef @ theta)
            grad = (features_sum - _theta_obj_coef)
            return obj, grad
        else:
            return None, None

    def compute_obj(self, theta, local_obs_weights = None):
        bundles = self.subproblem_manager.solve_subproblems(theta)
        utility = self.oracles_manager.utility_oracle(bundles, theta)
        utility_sum = self.comm_manager.sum_row_andReduce(local_obs_weights * utility)
        _theta_obj_coef = self._compute_theta_obj_coef(local_obs_weights)
        if self.comm_manager._is_root():
            return utility_sum - (_theta_obj_coef @ theta)
        else:
            return None
    
    def compute_grad(self, theta, local_obs_weights = None):
        bundles = self.subproblem_manager.solve_subproblems(theta)
        features = self.oracles_manager.features_oracle(bundles)
        _theta_obj_coef = self._compute_theta_obj_coef(local_obs_weights)
        features_sum = self.comm_manager.sum_row_andReduce(local_obs_weights[:, None] * features)
        if self.comm_manager._is_root():
            return (features_sum - _theta_obj_coef)
        else:
            return None


    def _create_result(self, num_iterations, master_model, theta_sol, cfg):
        if self.comm_manager._is_root():
            converged = num_iterations < cfg.max_iters
            final_objective = master_model.ObjVal if hasattr(master_model, 'ObjVal') else None
            timing_stats = self.timing_stats
            warnings = [] if final_objective is not None else ['All iterations were constraint violations']
            return EstimationResult(
                theta_hat=theta_sol, converged=converged, num_iterations=num_iterations,
                final_objective=final_objective,
                timing=timing_stats,
                warnings=warnings)


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
