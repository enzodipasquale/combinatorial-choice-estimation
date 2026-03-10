import time
import numpy as np
from combest.utils import get_logger
from .row_generation import RowGenerationSolver

logger = get_logger(__name__)


class OneSlackSolver(RowGenerationSolver):

    def __init__(self, pt_estimation_manager):
        super().__init__(pt_estimation_manager)
        self.slack_counter = {}

    # ------------------------------------------------------------------
    # Template implementations
    # ------------------------------------------------------------------

    def _initialize_master(self):
        theta_obj_coef = self.pt_estimation_manager.compute_theta_LP_coef(self.local_obs_weights)
        if self.comm_manager.is_root():
            self.master_model = self._setup_gurobi_model(self.cfg.master_gurobi_params)
            lb, ub = self.cfg.theta_bounds_arrays(self.dim.n_covariates, self.dim.covariate_names)
            theta = self.master_model.addMVar(self.dim.n_covariates,
                                              obj=theta_obj_coef, lb=lb, ub=ub,
                                              name='parameter')
            u_bar = self.master_model.addVar(lb=0, obj=1, name='utility')
            self.master_variables = (theta, u_bar)
            self.master_model.optimize()
            self.slack_counter = {}

    def _distribute_solution(self):
        if self.comm_manager.is_root():
            self.theta_iter = self.master_variables[0].X
        else:
            if self.theta_iter is None:
                self.theta_iter = np.zeros(self.dim.n_covariates, dtype=np.float64)
        self.theta_iter = self.comm_manager.Bcast(self.theta_iter)

    def _master_iteration(self, pricing_results):
        covariates = self.features_manager.covariates_oracle(pricing_results)
        errors = self.features_manager.error_oracle(pricing_results)
        covariates_sum = self.comm_manager.sum_row_andReduce(
            self.local_obs_weights[:, None] * covariates)
        errors_sum = self.comm_manager.sum_row_andReduce(
            self.local_obs_weights * errors)

        t1 = time.perf_counter()
        stop, reduced_cost, n_violations = False, None, 0

        if self.comm_manager.is_root():
            theta, u_bar = self.master_variables
            u_pricing = covariates_sum @ theta.X + errors_sum
            reduced_cost = u_pricing - u_bar.X

            if reduced_cost <= self.cfg.tolerance:
                stop = True
            else:
                self.master_model.addConstr(u_bar >= covariates_sum @ theta + errors_sum)
                self._enforce_slack_counter()
                self.master_model.optimize()
                n_violations = 1

        t2 = time.perf_counter()
        stop = self.comm_manager.bcast(stop)
        return stop, reduced_cost, n_violations, (t1, t2)

    # ------------------------------------------------------------------
    # Constraint pruning
    # ------------------------------------------------------------------

    def _enforce_slack_counter(self):
        if self.cfg.max_slack_counter == float('inf'):
            return 0
        to_remove = []
        for constr in self.master_model.getConstrs():
            if constr.CBasis == 0:
                self.slack_counter.pop(constr, None)
            else:
                self.slack_counter[constr] = self.slack_counter.get(constr, 0) + 1
                if self.slack_counter[constr] >= self.cfg.max_slack_counter:
                    to_remove.append(constr)
        for constr in to_remove:
            self.master_model.remove(constr)
            self.slack_counter.pop(constr, None)
        return len(to_remove)
