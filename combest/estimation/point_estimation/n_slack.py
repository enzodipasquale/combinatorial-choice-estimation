import time
import numpy as np
import gurobipy as gp
from combest.utils import get_logger
from .row_generation import RowGenerationSolver

logger = get_logger(__name__)


class NSlackSolver(RowGenerationSolver):

    def __init__(self, pt_estimation_manager):
        super().__init__(pt_estimation_manager)
        self.u_iter_local = None
        self.all_concatenated_constraints = None

    # ------------------------------------------------------------------
    # Template implementations
    # ------------------------------------------------------------------

    def _initialize_master(self):
        theta_obj_coef = self.pt_estimation_manager.compute_theta_obj_coef(self.local_obs_weights)
        u_obj_coef = self.pt_estimation_manager.compute_u_obj_weights(self.local_obs_weights)
        if self.comm_manager.is_root():
            self.master_model = self._setup_gurobi_model(self.cfg.master_gurobi_params)
            lb, ub = self.cfg.theta_bounds_arrays(self.dim.n_covariates)
            theta = self.master_model.addMVar(self.dim.n_covariates,
                                              obj=theta_obj_coef, lb=lb, ub=ub,
                                              name='parameter')
            u = self.master_model.addMVar(self.dim.n_agents, lb=0, obj=u_obj_coef, name='utility')
            self.master_variables = (theta, u)
            self.master_model.optimize()
            self.all_concatenated_constraints = None
            if self.master_model.Status != gp.GRB.OPTIMAL:
                raise RuntimeError(f'Master problem cannot be solved at initialization, status={self.master_model.Status}')

    def _distribute_solution(self):
        if self.comm_manager.is_root():
            theta, u = self.master_variables
            theta_iter = theta.X
            u_iter = u.X
        else:
            theta_iter = np.zeros(self.dim.n_covariates, dtype=np.float64)
            u_iter = np.zeros(self.dim.n_agents, dtype=np.float64)
        self.theta_iter = self.comm_manager.Bcast(theta_iter)
        self.u_iter_local = self.comm_manager.Scatterv_by_row(u_iter,
                                                              row_counts=self.comm_manager.agent_counts,
                                                              dtype=np.float64,
                                                              shape=(self.dim.n_agents,))

    def _master_iteration(self, pricing_results):
        constraints_coeff, (stop, reduced_cost, n_violations) = self.compute_constraints_coeff(pricing_results)
        t1 = time.perf_counter()
        if (not stop) and self.comm_manager.is_root():
            self.add_master_constraints(*constraints_coeff)
        if self.comm_manager.is_root():
            self.master_model.optimize()
        t2 = time.perf_counter()
        return stop, reduced_cost, n_violations, (t1, t2)

    def _result_u_hat(self):
        return self.master_variables[1].X

    # ------------------------------------------------------------------
    # Solve override (bootstrap warm-start)
    # ------------------------------------------------------------------

    def solve(self, resampling_weights=None, initialize_master=True,
              initialize_solver=True, iteration_callback=None,
              initialization_callback=None, verbose=False):
        self.verbose = verbose
        if self.verbose:
            self.pt_estimation_manager._log_instance_summary()
        if initialize_solver:
            self.subproblem_manager.initialize_solver()
        self.local_obs_weights = resampling_weights if resampling_weights is not None \
                                 else self.data_manager.local_obs_quantity
        if initialize_master:
            self._initialize_master()
        else:
            self.update_objective_for_weights()
            if self.comm_manager.is_root():
                self.master_model.optimize()
        return self._run_loop(iteration_callback, initialization_callback)

    # ------------------------------------------------------------------
    # Violations & constraints
    # ------------------------------------------------------------------

    def _compute_local_violations(self, pricing_results, theta, u_local, weights):
        covariates_local = self.features_manager.covariates_oracle(pricing_results)
        errors_local = self.features_manager.error_oracle(pricing_results)
        u_theta = covariates_local @ theta + errors_local
        weighted_reduced_costs = weights * (u_theta - u_local)

        local_max_rc = weighted_reduced_costs.max() if weighted_reduced_costs.size > 0 else -np.inf
        local_violations = np.where(weighted_reduced_costs > self.cfg.tolerance)[0]
        local_violations_id = self.comm_manager.agent_ids[local_violations]
        covariates_and_errors_local = np.column_stack([
                                                        covariates_local[local_violations],
                                                        errors_local[local_violations]
                                                    ])
        bundles_local = pricing_results[local_violations]
        return local_max_rc, len(local_violations), local_violations_id, bundles_local, covariates_and_errors_local

    def compute_constraints_coeff(self, pricing_results):
        local_max_rc, n_local_viol, local_violations_id, bundles_local, covariates_and_errors_local = \
            self._compute_local_violations(pricing_results, self.theta_iter, self.u_iter_local, self.local_obs_weights)

        local_meta = np.array([local_max_rc, n_local_viol], dtype=np.float64)
        all_meta = self.comm_manager.Allgather(local_meta).reshape(-1, 2)
        global_max_rc = all_meta[:, 0].max()
        violation_counts = all_meta[:, 1].astype(np.int64)
        stop = global_max_rc <= self.cfg.tolerance
        reduced_cost = global_max_rc if self.comm_manager.is_root() else None

        bundles = self.comm_manager.Gatherv_by_row(bundles_local, row_counts=violation_counts)
        covariates_and_errors = self.comm_manager.Gatherv_by_row(covariates_and_errors_local, row_counts=violation_counts)
        violations_id = self.comm_manager.Gatherv_by_row(local_violations_id, row_counts=violation_counts)

        if self.comm_manager.is_root():
            covariates = covariates_and_errors[:, :-1]
            errors = covariates_and_errors[:, -1]
            n_violations = len(violations_id)
        else:
            covariates, errors, n_violations = None, None, None

        return (violations_id, bundles, covariates, errors), (stop, reduced_cost, n_violations)

    def add_master_constraints(self, indices, bundles, covariates, errors):
        if not self.comm_manager.is_root() or self.master_model is None:
            return
        theta, u = self.master_variables
        constr = self.master_model.addConstr(u[indices] >= covariates @ theta + errors)
        if self.all_concatenated_constraints is None:
            self.all_concatenated_constraints = constr
        else:
            self.all_concatenated_constraints = gp.concatenate([self.all_concatenated_constraints, constr])
        return constr

    # ------------------------------------------------------------------
    # Bootstrap support
    # ------------------------------------------------------------------

    def update_objective_for_weights(self):
        theta_obj_coef = self.pt_estimation_manager.compute_theta_obj_coef(self.local_obs_weights)
        u_obj_weights = self.pt_estimation_manager.compute_u_obj_weights(self.local_obs_weights)
        if not self.comm_manager.is_root() or self.master_model is None:
            return
        theta, u = self.master_variables
        theta.Obj = theta_obj_coef
        u.Obj = u_obj_weights
        self.master_model.update()

    def copy_master_model(self):
        if not self.comm_manager.is_root() or self.master_model is None:
            return None, None
        model = self.master_model.copy()
        all_vars = model.getVars()
        theta = gp.MVar.fromlist(all_vars[:self.dim.n_covariates])
        u = gp.MVar.fromlist(all_vars[self.dim.n_covariates:self.dim.n_covariates + self.dim.n_agents])
        return model, (theta, u)

    def install_master_model(self, model, variables):
        if not self.comm_manager.is_root():
            return
        self.master_model = model
        self.master_variables = variables
        self.all_concatenated_constraints = None

    # ------------------------------------------------------------------
    # Constraint management
    # ------------------------------------------------------------------

    def strip_slack_constraints(self, percentile=100, hard_threshold=float('inf')):
        if not self.comm_manager.is_root() or self.master_model is None or self.all_concatenated_constraints is None:
            return 0
        slacks = self.all_concatenated_constraints.Slack
        threshold = np.percentile(slacks, 100.0 - percentile)
        below_threshold = slacks < threshold
        n_below = below_threshold.sum()
        if n_below < len(slacks) - hard_threshold:
            return self.strip_constraints_hard_threshold(hard_threshold)
        to_keep_id = np.where(~below_threshold)[0]
        to_remove_id = np.where(below_threshold)[0]
        to_remove = self.all_concatenated_constraints[to_remove_id]
        self.all_concatenated_constraints = self.all_concatenated_constraints[to_keep_id]
        self.master_model.remove(to_remove)
        return len(to_remove_id)

    def strip_constraints_hard_threshold(self, n_constraints=float('inf')):
        if not self.comm_manager.is_root() or self.master_model is None or self.all_concatenated_constraints is None:
            return 0
        if self.master_model.NumConstrs < n_constraints:
            return 0
        slacks = self.all_concatenated_constraints.Slack
        sorted_slacks = np.argsort(slacks)
        to_remove_id, to_keep_id = sorted_slacks[:-n_constraints], sorted_slacks[-n_constraints:]
        to_remove = self.all_concatenated_constraints[to_remove_id]
        self.all_concatenated_constraints = self.all_concatenated_constraints[to_keep_id]
        self.master_model.remove(to_remove)
        return len(to_remove_id)
