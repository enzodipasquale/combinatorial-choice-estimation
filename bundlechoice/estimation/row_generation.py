import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from bundlechoice.utils import get_logger, make_timing_stats
from .base import BaseEstimationManager
logger = get_logger(__name__)

class RowGenerationManager(BaseEstimationManager):

    def __init__(self, comm_manager, config, data_manager, oracles_manager, subproblem_manager):
        super().__init__(comm_manager, config, data_manager, oracles_manager, subproblem_manager)
        self.master_model = None
        self.master_variables = None
        self.timing_stats = None
        self.theta_val = None
        self.theta_hat = None
        self.slack_counter = None
        self.constraint_info = {}

    def _initialize_master_problem(self, initial_constraints=None):
        cfg, dim = self.config.row_generation, self.config.dimensions
        obs_features = self.get_obs_features()
        if self.comm_manager._is_root() and self._agent_weights is not None:
            weights_tiled = np.tile(self._agent_weights, dim.num_simulations)
            obs_features = (weights_tiled[:, None] * self._features_at_obs_bundles).sum(0)
        if self.comm_manager._is_root():
            self.constraint_info.clear()
            self.master_model = self._setup_gurobi_model(cfg.gurobi_settings)
            theta = self.master_model.addMVar(dim.num_features, obj=-obs_features, ub=cfg.theta_ubs, name='parameter')
            if cfg.theta_lbs is not None:
                for k in range(dim.num_features):
                    if k < len(cfg.theta_lbs) and cfg.theta_lbs[k] is not None:
                        theta[k].lb = float(cfg.theta_lbs[k])
            else:
                theta.lb = 0.0
            if self._theta_init_for_start is not None:
                theta.Start = self._theta_init_for_start
            n_agents = dim.num_simulations * dim.num_obs
            u_obj = np.tile(self._agent_weights, dim.num_simulations) if self._agent_weights is not None else 1
            u = self.master_model.addMVar(n_agents, obj=u_obj, name='utility')
            self.master_variables = (theta, u)
            if initial_constraints is not None and len(initial_constraints.get('indices', [])) > 0:
                self.add_constraints(initial_constraints['indices'], initial_constraints['bundles'])
            if cfg.master_init_callback is not None:
                cfg.master_init_callback(self.master_model, theta, u)
            self.master_model.optimize()
            logger.info('Master Initialized')
            self.theta_val = theta.X if self.master_model.Status == GRB.OPTIMAL else np.zeros(dim.num_features, dtype=np.float64)
            if self.master_model.Status != GRB.OPTIMAL:
                logger.warning('Master problem not optimal at initialization, status=%s', self.master_model.Status)
            self.log_parameter()
        else:
            self.theta_val = np.empty(dim.num_features, dtype=np.float64)
        self.theta_val = self.comm_manager._Bcast(self.theta_val, root=0)

    def _master_iteration(self, local_pricing_results):
        from mpi4py import MPI
        cfg, dim = self.config.row_generation, self.config.dimensions
        features_local = self.oracles_manager.compute_rank_features(local_pricing_results)
        errors_local = self.oracles_manager.compute_rank_errors(local_pricing_results)
        global_indices_local = self.data_manager.local_data['global_indices']
        all_counts = self.data_manager.local_data['agent_counts']
        n_agents = dim.num_simulations * dim.num_obs
        if self.comm_manager._is_root():
            theta, u = self.master_variables
            u_master_all, theta_current = u.X, theta.X
        else:
            u_master_all = np.empty(n_agents, dtype=np.float64)
            theta_current = np.empty(dim.num_features, dtype=np.float64)
        u_master_local = self.comm_manager._Scatterv_by_row(u_master_all, counts=all_counts, dtype=np.float64)
        theta_current = self.comm_manager._Bcast(theta_current, root=0)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            u_sim_local = features_local @ theta_current + errors_local
        local_violations = np.where(~np.isclose(u_master_local, u_sim_local, rtol=1e-05, atol=1e-05) & (u_master_local > u_sim_local))[0]
        if len(local_violations) > 0:
            logger.warning('Rank %d: Possible failure of demand oracle at local ids: %s', self.comm_manager.rank, local_violations[:10])
        reduced_costs_local = u_sim_local - u_master_local
        local_max_rc = reduced_costs_local.max() if len(reduced_costs_local) > 0 else -np.inf
        max_reduced_cost = self.comm_manager.comm.sum_row_and_Reduce(local_max_rc, op=MPI.MAX)
        local_rows_to_add = np.where(u_sim_local > u_master_local * (1 + cfg.tol_row_generation) + cfg.tolerance_optimality)[0]
        viol_global_ids = global_indices_local[local_rows_to_add]
        viol_bundles = local_pricing_results[local_rows_to_add]
        viol_features = features_local[local_rows_to_add]
        viol_errors = errors_local[local_rows_to_add]
        all_viol_ids = self.comm_manager._Gatherv_by_row(viol_global_ids, root=0)
        all_viol_bundles = self.comm_manager._Gatherv_by_row(viol_bundles, root=0)
        all_viol_features = self.comm_manager._Gatherv_by_row(viol_features, root=0)
        all_viol_errors = self.comm_manager._Gatherv_by_row(viol_errors, root=0)
        stop = False
        if self.comm_manager._is_root():
            self.log_parameter()
            logger.info(f'ObjVal: {self.master_model.ObjVal}, Reduced cost: {max_reduced_cost}')
            suboptimal_mode = getattr(self.subproblem_manager, '_suboptimal_mode', False)
            if max_reduced_cost < cfg.tolerance_optimality:
                stop = not suboptimal_mode
                if suboptimal_mode:
                    logger.info('Reduced cost below tolerance, but suboptimal cuts mode active - continuing')
            num_new = len(all_viol_ids) if all_viol_ids is not None else 0
            logger.info('New constraints: %d', num_new)
            if num_new > 0:
                theta, u = self.master_variables
                for i in range(num_new):
                    idx = int(all_viol_ids[i])
                    constr = self.master_model.addConstr(u[idx] >= all_viol_errors[i] + all_viol_features[i] @ theta)
                    self.constraint_info[constr] = (idx, all_viol_bundles[i].copy())
            self._enforce_slack_counter()
            logger.info('Number of constraints: %d', self.master_model.NumConstrs)
            self.master_model.optimize()
            theta_val = self.master_variables[0].X
            cfg.tol_row_generation *= cfg.row_generation_decay
        else:
            theta_val = np.empty(dim.num_features, dtype=np.float64)
        self.theta_val, stop = self.comm_manager.Bcast(theta_val, root=0)
        return stop

    def solve(self, callback=None, theta_init=None, agent_weights=None, initial_constraints=None):
        cfg, dim = self.config.row_generation, self.config.dimensions
        self._agent_weights = np.asarray(agent_weights, dtype=np.float64) if agent_weights is not None and self.comm_manager._is_root() else None
        self._theta_init_for_start = theta_init.theta_hat if hasattr(theta_init, 'theta_hat') else theta_init if theta_init is not None else None
        if self.comm_manager._is_root():
            logger.info(f'ROW GENERATION: {dim.num_obs} agents x {dim.num_items} items, {dim.num_features} features, max_iters={cfg.max_iters}')
        tic = time.perf_counter()
        self.subproblem_manager.initialize_subproblems()
        if initial_constraints is None and theta_init is not None:
            theta_arr = theta_init.theta_hat if hasattr(theta_init, 'theta_hat') else theta_init
            if self.comm_manager._is_root():
                self.theta_val = np.asarray(theta_arr, dtype=np.float64).copy()
            else:
                self.theta_val = np.empty(dim.num_features, dtype=np.float64)
            self.theta_val = self.comm_manager._Bcast(self.theta_val, root=0)
            local_pricing_results = self.subproblem_manager.solve_local(self.theta_val)
            bundles_sim = self.comm_manager._Gatherv_by_row(local_pricing_results, root=0)
            if self.comm_manager._is_root() and bundles_sim is not None and len(bundles_sim) > 0:
                initial_constraints = {'indices': np.arange(dim.num_simulations * dim.num_obs, dtype=np.int64), 
                                       'bundles': bundles_sim.astype(np.float64)}
        self._initialize_master_problem(initial_constraints=initial_constraints)
        self.slack_counter = {}
        iteration, pricing_times, master_times = 0, [], []
        while iteration < cfg.max_iters:
            logger.info(f'ITERATION {iteration + 1}')
            if cfg.subproblem_callback is not None:
                cfg.subproblem_callback(iteration, self.subproblem_manager, self.master_model if self.comm_manager._is_root() else None)
            t0 = time.perf_counter()
            local_pricing_results = self.subproblem_manager.solve_local(self.theta_val)
            pricing_times.append(time.perf_counter() - t0)
            t1 = time.perf_counter()
            stop = self._master_iteration(local_pricing_results)
            master_times.append(time.perf_counter() - t1)
            if callback and self.comm_manager._is_root():
                callback({'iteration': iteration + 1, 'theta': self.theta_val.copy(), 
                          'objective': getattr(self.master_model, 'ObjVal', None), 
                          'pricing_time': pricing_times[-1], 'master_time': master_times[-1]})
            if stop and iteration >= cfg.min_iters:
                break
            iteration += 1
        elapsed = time.perf_counter() - tic
        num_iters, converged = iteration + 1, iteration < cfg.max_iters
        bounds_info = self._check_bounds_hit()
        warnings_list = self._log_bounds_warnings(bounds_info)
        if self.comm_manager._is_root():
            logger.info(f'Row generation {"ended" if converged else "reached max iterations"} after {num_iters} iterations in {elapsed:.2f}s')
            obj_val = getattr(self.master_model, 'ObjVal', None)
            self.timing_stats = make_timing_stats(elapsed, num_iters, pricing_times, master_times)
            self._log_timing_summary(self.timing_stats, obj_val, self.theta_val, header='ROW GENERATION SUMMARY')
        else:
            obj_val, self.timing_stats = None, None
        self.theta_hat = self.theta_val.copy()
        result = self._create_result(self.theta_hat, converged, num_iters, obj_val)
        result.warnings.extend(warnings_list)
        result.metadata['bounds_hit'] = bounds_info
        return result

    def _on_constraint_removed(self, constr):
        self.constraint_info.pop(constr, None)

    def add_constraints(self, indices, bundles):
        if not self.comm_manager._is_root() or self.master_model is None or len(indices) == 0:
            return
        theta, u = self.master_variables
        input_data = self.data_manager.input_data
        n_agents = self.config.dimensions.num_obs
        agent_ids, sim_ids = indices % n_agents, indices // n_agents
        errors_tensor = input_data.get('errors')
        has_sim = errors_tensor is not None and errors_tensor.ndim == 3
        sim_data = lambda s: {**input_data, 'errors': errors_tensor[s]} if has_sim else input_data
        features = np.stack([self.oracles_manager.features_oracle(int(agent_ids[i]), bundles[i], input_data) for i in range(len(indices))])
        errors = np.array([self.oracles_manager.error_oracle(int(agent_ids[i]), bundles[i], sim_data(int(sim_ids[i]))) for i in range(len(indices))])
        for i, idx in enumerate(indices):
            constr = self.master_model.addConstr(u[idx] >= errors[i] + features[i] @ theta)
            self.constraint_info[constr] = (idx, bundles[i].copy())
        logger.info('Added %d constraints', len(indices))

    def get_constraints(self):
        if not self.comm_manager._is_root() or self.master_model is None:
            return None
        if not self.constraint_info:
            return self._empty_constraints_dict()
        indices, bundles = zip(*[(idx, bundle) for idx, bundle in self.constraint_info.values()])
        return {'indices': np.array(indices, dtype=np.int64), 'bundles': np.array(bundles, dtype=np.float64)}

    def get_binding_constraints(self, tolerance=1e-06):
        if not self.comm_manager._is_root() or self.master_model is None:
            return None
        indices, bundles = [], []
        for constr in self.master_model.getConstrs():
            if constr in self.constraint_info and abs(constr.Slack) <= tolerance:
                idx, bundle = self.constraint_info[constr]
                indices.append(idx)
                bundles.append(bundle)
        return self._empty_constraints_dict() if not indices else {'indices': np.array(indices, dtype=np.int64), 'bundles': np.array(bundles, dtype=np.float64)}

    def strip_slack_constraints(self, tolerance=1e-06):
        if not self.comm_manager._is_root() or self.master_model is None:
            return 0
        to_remove = [c for c in self.master_model.getConstrs() if c in self.constraint_info and abs(c.Slack) > tolerance]
        for constr in to_remove:
            self.master_model.remove(constr)
            self.constraint_info.pop(constr, None)
            self.slack_counter.pop(constr, None)
        if to_remove:
            self.master_model.update()
            logger.info('Stripped %d slack constraints', len(to_remove))
        return len(to_remove)

    def update_objective_for_weights(self, agent_weights):
        if not self.comm_manager._is_root() or self.master_model is None:
            return
        theta, u = self.master_variables
        weights_tiled = np.tile(agent_weights, self.config.dimensions.num_simulations)
        theta.Obj = -(weights_tiled[:, None] * self._features_at_obs_bundles).sum(0)
        u.Obj = weights_tiled
        self.master_model.update()

    def solve_reuse_model(self, agent_weights, strip_slack=False, reset_lp=False):
        cfg, dim = self.config.row_generation, self.config.dimensions
        self._agent_weights = np.asarray(agent_weights, dtype=np.float64) if self.comm_manager._is_root() else None
        tic = time.perf_counter()
        error_msg = None
        if self.comm_manager._is_root():
            try:
                if self.master_model is None:
                    raise RuntimeError('No existing model to reuse. Call solve() first.')
                if strip_slack:
                    self.strip_slack_constraints()
                self.update_objective_for_weights(self._agent_weights)
                if reset_lp:
                    self.master_model.reset(0)
                self.master_model.optimize()
                self.theta_val = self.master_variables[0].X if self.master_model.Status == GRB.OPTIMAL else np.zeros(dim.num_features, dtype=np.float64)
            except Exception as e:
                error_msg = str(e)
                self.theta_val = np.zeros(dim.num_features, dtype=np.float64)
        else:
            self.theta_val = np.empty(dim.num_features, dtype=np.float64)
        error_msg = self.comm_manager.comm.bcast(error_msg, root=0)
        if error_msg is not None:
            raise RuntimeError(f'Root process failed: {error_msg}')
        self.theta_val = self.comm_manager._Bcast(self.theta_val, root=0)
        iteration = 0
        while iteration < cfg.max_iters:
            local_pricing_results = self.subproblem_manager.solve_local(self.theta_val)
            stop = self._master_iteration(local_pricing_results)
            if stop and iteration >= cfg.min_iters:
                break
            iteration += 1
        elapsed = time.perf_counter() - tic
        num_iters, converged = iteration + 1, iteration < cfg.max_iters
        bounds_info = self._check_bounds_hit()
        warnings_list = self._log_bounds_warnings(bounds_info)
        if self.comm_manager._is_root():
            obj_val = self.master_model.ObjVal if self.master_model.Status == GRB.OPTIMAL else float('inf')
            self.timing_stats = {'total_time': elapsed, 'num_iterations': num_iters}
        else:
            obj_val, self.timing_stats = None, None
        result = self._create_result(self.theta_val, converged, num_iters, obj_val)
        result.warnings.extend(warnings_list)
        result.metadata['bounds_hit'] = bounds_info
        return result



    def _check_bounds_hit(self, tol=1e-06):
        from gurobipy import GRB
        empty = {'hit_lower': [], 'hit_upper': [], 'any_hit': False}
        if not self.comm_manager._is_root() or self.master_model is None:
            return empty
        theta = self.master_variables[0]
        hit_lower = [k for k in range(self.config.dimensions.num_features) 
                    if theta[k].LB > -GRB.INFINITY and abs(theta[k].X - theta[k].LB) < tol]
        hit_upper = [k for k in range(self.config.dimensions.num_features) 
                    if theta[k].UB < GRB.INFINITY and abs(theta[k].X - theta[k].UB) < tol]
        return {'hit_lower': hit_lower, 'hit_upper': hit_upper, 'any_hit': bool(hit_lower or hit_upper)}


    def _enforce_slack_counter(self):
        cfg = self.config.row_generation
        if cfg.max_slack_counter >= float('inf') or self.master_model is None:
            return 0
        if self.slack_counter is None:
            self.slack_counter = {}
        to_remove = []
        for constr in self.master_model.getConstrs():
            if constr.Slack < -1e-06:
                self.slack_counter[constr] = self.slack_counter.get(constr, 0) + 1
                if self.slack_counter[constr] >= cfg.max_slack_counter:
                    to_remove.append(constr)
            if constr.Pi > 1e-06:
                self.slack_counter.pop(constr, None)
        for constr in to_remove:
            self.master_model.remove(constr)
            self.slack_counter.pop(constr, None)
            self._on_constraint_removed(constr)
        if to_remove:
            logger.info('Removed %d slack constraints', len(to_remove))
        return len(to_remove)


    def _log_bounds_warnings(self, bounds_info):
        warnings_list = []
        if self.comm_manager._is_root() and bounds_info["any_hit"]:
            for bound_type in ["lower", "upper"]:
                if bounds_info[f"hit_{bound_type}"]:
                    msg = f"Theta hit {bound_type.upper()} bound at indices: {bounds_info[f"hit_{bound_type}"]}"
                    logger.warning(msg)
                    warnings_list.append(msg)
        return warnings_list

    def _setup_gurobi_model(self, gurobi_settings=None):
        import gurobipy as gp
        params = {"Method": 0, "LPWarmStart": 2, "OutputFlag": 0, **(gurobi_settings or {})}
        with suppress_output():
            model = gp.Model()
            for k, v in params.items():
                if v is not None:
                    model.setParam(k, v)
        return model