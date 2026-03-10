import numpy as np
from combest.utils import get_logger

logger = get_logger(__name__)


class PointEstimationManager:

    def __init__(self, comm_manager, config, data_manager, features_manager, subproblem_manager):
        self.comm_manager = comm_manager
        self.config = config
        self.data_manager = data_manager
        self.features_manager = features_manager
        self.subproblem_manager = subproblem_manager

        from combest.estimation.point_estimation import NSlackSolver, OneSlackSolver, EllipsoidSolver
        self.n_slack = NSlackSolver(self)
        self.one_slack = OneSlackSolver(self)
        self.ellipsoid = EllipsoidSolver(self)

    # ------------------------------------------------------------------
    # Objective coefficients
    # ------------------------------------------------------------------

    def compute_theta_LP_coef(self, local_obs_weights=None):
        if local_obs_weights is None:
            local_obs_weights = self.data_manager.local_obs_quantity
        local_obs_covariates = self.features_manager.covariates_oracle(self.data_manager.local_obs_bundles)
        return self.comm_manager.sum_row_andReduce(-local_obs_weights[:, None] * local_obs_covariates)

    def compute_u_LP_coef(self, local_obs_weights=None):
        if local_obs_weights is None:
            local_obs_weights = self.data_manager.local_obs_quantity
        all_weights = self.comm_manager.Gatherv_by_row(local_obs_weights, row_counts=self.comm_manager.agent_counts)
        return all_weights if self.comm_manager.is_root() else None

    # ------------------------------------------------------------------
    # Objective / gradient evaluation
    # ------------------------------------------------------------------

    def compute_nonlinear_obj_and_grad_at_root(self, theta, local_obs_weights=None):
        if local_obs_weights is None:
            local_obs_weights = self.data_manager.local_obs_quantity
        w = local_obs_weights
        bundles = self.subproblem_manager.solve(theta)

        cov_V, err_V = self.features_manager.covariates_and_errors_oracle(bundles)
        cov_Q, err_Q = self.features_manager.covariates_and_errors_oracle(
            self.data_manager.local_obs_bundles
        )

        grad = self.comm_manager.sum_row_andReduce(w[:, None] * (cov_V - cov_Q))
        const = self.comm_manager.sum_row_andReduce(w * (err_V - err_Q))

        if self.comm_manager.is_root():
            return (grad @ theta + const).item(), grad
        return None, None

    def compute_polyhedral_obj_and_grad_at_root(self, theta, local_obs_weights=None):
        bundles = self.subproblem_manager.solve(theta)
        covariates = self.features_manager.covariates_oracle(bundles)
        utility = self.features_manager.utility_oracle(bundles, theta)

        covariates_sum = self.comm_manager.sum_row_andReduce(local_obs_weights[:, None] * covariates)
        utility_sum = self.comm_manager.sum_row_andReduce(local_obs_weights * utility)
        theta_obj_coef = self.compute_theta_LP_coef(local_obs_weights)

        if self.comm_manager.is_root():
            obj = utility_sum + (theta_obj_coef @ theta)
            grad = (covariates_sum + theta_obj_coef)
            return obj, grad
        else:
            return None, None


    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_instance_summary(self):
        if not self.comm_manager.is_root():
            return None

        dim = self.config.dimensions
        metadata = {
            'n_obs': dim.n_obs,
            'n_items': dim.n_items,
            'n_covariates': dim.n_covariates,
            'n_simulations': dim.n_simulations,
            'comm_size': self.comm_manager.comm_size,
            'subproblem': self.config.subproblem.name,
        }

        header = (f"{'n_obs':>6} | {'n_items':>8} | {'n_covariates':>11} | {'n_simulations':>14} |"
                  + f" {'comm_size':>10} | {'subproblem':>20}")
        values = (f"{metadata['n_obs']:>6} | {metadata['n_items']:>8} | {metadata['n_covariates']:>11} |"
                  + f" {metadata['n_simulations']:>14} | {metadata['comm_size']:>10} | {metadata['subproblem'] or 'N/A':>20}")
        logger.info(" PROBLEM METADATA")
        logger.info("-" * 90)
        logger.info(header)
        logger.info(values)
        logger.info("-" * 90)

        return metadata
