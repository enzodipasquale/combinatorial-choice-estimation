import time
import numpy as np
import gurobipy as gp
from combest.utils import get_logger
from combest.estimation.bundle_store import BundleStore
from .row_generation import RowGenerationSolver

logger = get_logger(__name__)


class NSlackSolver(RowGenerationSolver):

    def __init__(self, pt_estimation_manager):
        super().__init__(pt_estimation_manager)
        self.u_iter_local = None
        self.all_concatenated_constraints = None
        self.cut_agent_ids = np.empty(0, dtype=np.int32)
        self.bundle_store = None    # lazily created in _initialize_master (needs dim.n_items)

    # ------------------------------------------------------------------
    # Template implementations
    # ------------------------------------------------------------------

    def _initialize_master(self):
        theta_coef, u_coef = self.compute_LP_coef(self.local_obs_weights)
        if self.comm_manager.is_root():
            self.master_model = self._setup_gurobi_model(self.cfg.master_gurobi_params)
            lb, ub = self.cfg.theta_bounds_arrays(self.dim.n_covariates, self.dim.covariate_names)
            theta = self.master_model.addMVar(self.dim.n_covariates,
                                              obj=theta_coef, lb=lb, ub=ub,
                                              name='parameter')
            u = self.master_model.addMVar(self.dim.n_agents, lb=0, obj=u_coef, name='utility')
            self.master_variables = (theta, u)
            self.master_model.optimize()
            self.all_concatenated_constraints = None
            self.cut_agent_ids = np.empty(0, dtype=np.int32)
            self.bundle_store = BundleStore(self.dim.n_items)
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
            if self.master_model.Status != gp.GRB.OPTIMAL:
                raise RuntimeError(f'Master problem not optimal after iteration, status={self.master_model.Status}')
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


    def compute_constraints_coeff(self, cuts):
        covariates, error, u_theta , bundles  = cuts
        reduced_costs = self.local_obs_weights * (u_theta - self.u_iter_local)

        local_max_rc = reduced_costs.max() if reduced_costs.size > 0 else -np.inf
        local_violations = np.where(reduced_costs > self.cfg.tolerance)[0]
        local_violations_id = self.comm_manager.agent_ids[local_violations]
        covariates_and_error = np.column_stack([
                                                        covariates[local_violations],
                                                        error[local_violations]
                                                    ])
        bundles_local = bundles[local_violations]
        n_local_viol =  len(local_violations)
        
        local_meta = np.array([local_max_rc, n_local_viol], dtype=np.float64)
        all_meta = self.comm_manager.Allgather(local_meta).reshape(-1, 2)
        global_max_rc = all_meta[:, 0].max()
        violation_counts = all_meta[:, 1].astype(np.int64)
        stop = global_max_rc <= self.cfg.tolerance
        reduced_cost = global_max_rc if self.comm_manager.is_root() else None

        bundles = self.comm_manager.Gatherv_by_row(bundles_local, row_counts=violation_counts)
        covariates_and_error = self.comm_manager.Gatherv_by_row(covariates_and_error, row_counts=violation_counts)
        violations_id = self.comm_manager.Gatherv_by_row(local_violations_id, row_counts=violation_counts)

        if self.comm_manager.is_root():
            covariates = covariates_and_error[:, :-1]
            error = covariates_and_error[:, -1]
            n_violations = len(violations_id)
        else:
            covariates, error, bundles, n_violations = None, None, None, None

        return (violations_id, covariates, error, bundles), (stop, reduced_cost, n_violations)

    def add_master_constraints(self, indices, covariates, error, bundles):
        if not self.comm_manager.is_root() or self.master_model is None:
            return
        theta, u = self.master_variables
        constr = self.master_model.addConstr(u[indices] >= covariates @ theta + error)
        if self.all_concatenated_constraints is None:
            self.all_concatenated_constraints = constr
        else:
            self.all_concatenated_constraints = gp.concatenate([self.all_concatenated_constraints, constr])
        self.cut_agent_ids = np.concatenate([self.cut_agent_ids, np.asarray(indices, dtype=np.int32)])
        self.bundle_store.add_cuts(bundles)
        return constr

    # ------------------------------------------------------------------
    # Bootstrap support
    # ------------------------------------------------------------------

    def update_objective_for_weights(self):
        theta_obj_coef = self.compute_theta_LP_coef(self.local_obs_weights)
        u_obj_weights = self.compute_u_LP_coef(self.local_obs_weights)
        if not self.comm_manager.is_root() or self.master_model is None:
            return
        theta, u = self.master_variables
        theta.Obj = theta_obj_coef
        u.Obj = u_obj_weights
        self.master_model.update()

    def copy_master_model(self):
        if not self.comm_manager.is_root() or self.master_model is None:
            return None, None, None, None
        model = self.master_model.copy()
        all_vars = model.getVars()
        theta = gp.MVar.fromlist(all_vars[:self.dim.n_covariates])
        u = gp.MVar.fromlist(all_vars[self.dim.n_covariates:self.dim.n_covariates + self.dim.n_agents])
        cut_agent_ids = self.cut_agent_ids.copy()
        bundle_store = BundleStore.from_state(
            self.dim.n_items, {k: v.copy() for k, v in self.bundle_store.state().items()})
        return model, (theta, u), cut_agent_ids, bundle_store

    def install_master_model(self, model, variables, cut_agent_ids=None, bundle_store=None):
        if not self.comm_manager.is_root():
            return
        self.master_model = model
        self.master_variables = variables
        self.all_concatenated_constraints = None
        self.cut_agent_ids = np.empty(0, dtype=np.int32) if cut_agent_ids is None else cut_agent_ids
        self.bundle_store = BundleStore(self.dim.n_items) if bundle_store is None else bundle_store

    # ------------------------------------------------------------------
    # Constraint management
    # ------------------------------------------------------------------

    def _apply_keep_mask(self, keep_mask):
        remove_idx = np.where(~keep_mask)[0]
        keep_idx = np.where(keep_mask)[0]
        self.master_model.remove(self.all_concatenated_constraints[remove_idx])
        self.all_concatenated_constraints = self.all_concatenated_constraints[keep_idx]
        self.cut_agent_ids = self.cut_agent_ids[keep_mask]
        self.bundle_store.prune(keep_mask)
        return len(remove_idx)

    def strip_slack_constraints(self, percentile=100, hard_threshold=float('inf')):
        if not self.comm_manager.is_root() or self.master_model is None or self.all_concatenated_constraints is None:
            return 0
        slacks = self.all_concatenated_constraints.Slack
        below = slacks < np.percentile(slacks, 100.0 - percentile)
        if below.sum() < len(slacks) - hard_threshold:
            return self.strip_constraints_hard_threshold(hard_threshold)
        return self._apply_keep_mask(~below)

    def strip_constraints_hard_threshold(self, n_constraints=float('inf')):
        if not self.comm_manager.is_root() or self.master_model is None or self.all_concatenated_constraints is None:
            return 0
        if self.master_model.NumConstrs < n_constraints:
            return 0
        slacks = self.all_concatenated_constraints.Slack
        keep_idx = np.argsort(slacks)[-int(n_constraints):]
        keep_mask = np.zeros(len(slacks), dtype=bool)
        keep_mask[keep_idx] = True
        return self._apply_keep_mask(keep_mask)

    # ------------------------------------------------------------------
    # Dual solution (agent, sim, bundle, pi) for nonzero duals
    # ------------------------------------------------------------------

    def dual_solution(self, atol=1e-10):
        if not self.comm_manager.is_root() or self.master_model is None:
            return None
        constrs = self.master_model.getConstrs()
        if not constrs:
            return None
        pi = np.asarray(self.master_model.getAttr('Pi', constrs))
        if len(pi) != len(self.cut_agent_ids) or len(pi) != self.bundle_store.cut_to_bundle.size:
            return None    # tracking state out of sync with model (e.g. legacy checkpoint)
        nz = np.where(np.abs(pi) > atol)[0]
        agent_ids = self.cut_agent_ids[nz]
        return {
            'agent_ids': agent_ids,
            'sim_ids': (agent_ids // self.dim.n_obs).astype(np.int32),
            'obs_ids': (agent_ids % self.dim.n_obs).astype(np.int32),
            'bundles': self.bundle_store.get(nz),
            'pi': pi[nz],
        }
