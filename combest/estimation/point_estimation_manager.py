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

        from combest.estimation.point_estimation import NSlackSolver, OneSlackSolver, EllipsoidSolver, BundleSolver
        self.n_slack = NSlackSolver(self)
        self.one_slack = OneSlackSolver(self)
        self.ellipsoid = EllipsoidSolver(self)
        self.bundle = BundleSolver(self)

    # ------------------------------------------------------------------
    # Objective coefficients
    # ------------------------------------------------------------------



    # ------------------------------------------------------------------
    # Objective / gradient evaluation
    # ------------------------------------------------------------------
    def compute_cuts(self, theta):
        bundles = self.subproblem_manager.solve(theta)
        grads, const = self.features_manager.covariates_and_errors_oracle(bundles)
        val = grads @ theta + const
        return grads, const, val, bundles

    def compute_nonlinear_cuts_at_root(self, theta, local_obs_weights=None):
        if local_obs_weights is None:
            local_obs_weights = self.data_manager.local_obs_quantity
        w = local_obs_weights
        bundles = self.subproblem_manager.solve(theta)

        cov_V, err_V = self.features_manager.covariates_and_errors_oracle(bundles)
        cov_Q, err_Q = self.features_manager.covariates_and_errors_oracle(
            self.data_manager.local_obs_bundles
        )

        grads = self.comm_manager.Gatherv_by_row(w[:, None] * (cov_V - cov_Q), row_counts=self.comm_manager.agent_counts)
        const = self.comm_manager.Gatherv_by_row(w * (err_V - err_Q), row_counts=self.comm_manager.agent_counts)

        if self.comm_manager.is_root():
            return grads @ theta + const, grads
        return None, None


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
            return (grad @ theta + const).item() / self.config.dimensions.n_agents, grad/ self.config.dimensions.n_agents
        return None, None


    def compute_polyhedral_obj_and_grad_at_root(self, theta, local_obs_weights=None):
        if local_obs_weights is None:
            local_obs_weights = self.data_manager.local_obs_quantity
        grads, _, val, _ = self.compute_cuts(theta)
        grad = self.comm_manager.sum_row_andReduce(local_obs_weights[:, None] * grads)
        val = self.comm_manager.sum_row_andReduce(local_obs_weights * val)
        local_obs_covariates = self.features_manager.covariates_oracle(self.data_manager.local_obs_bundles)
        theta_obj_coef = self.comm_manager.sum_row_andReduce(-local_obs_weights[:, None] * local_obs_covariates)
        if self.comm_manager.is_root():
            obj = val + (theta_obj_coef @ theta)
            grad = (grad + theta_obj_coef)
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
