import time
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Any, Dict, List, Callable
import gurobipy as gp
from gurobipy import GRB
from bundlechoice.utils import get_logger, suppress_output, make_timing_stats
from .base import BaseEstimationManager
from .result import EstimationResult
logger = get_logger(__name__)

class RowGenerationManager(BaseEstimationManager):

    def __init__(self, comm_manager, dimensions_cfg, row_generation_cfg, data_manager, oracles_manager, subproblem_manager):
        super().__init__(comm_manager=comm_manager, dimensions_cfg=dimensions_cfg, data_manager=data_manager, oracles_manager=oracles_manager, subproblem_manager=subproblem_manager)
        self.row_generation_cfg = row_generation_cfg
        self.master_model = None
        self.master_variables = None
        self.timing_stats = None
        self.theta_val = None
        self.theta_hat = None
        self.slack_counter = None
        self.constraint_info = {}

    def _setup_gurobi_model_params(self):
        return self._setup_gurobi_model(self.row_generation_cfg.gurobi_settings)

    def _initialize_master_problem(self, initial_constraints=None):
        obs_features = self.get_obs_features()
        if self.comm_manager._is_root() and hasattr(self, '_agent_weights') and (self._agent_weights is not None):
            weights_tiled = np.tile(self._agent_weights, self.dimensions_cfg.num_simulations)
            obs_features = (weights_tiled[:, None] * self.agents_obs_features).sum(0)
        if self.comm_manager._is_root():
            self.constraint_info.clear()
            self.master_model = self._setup_gurobi_model_params()
            theta = self.master_model.addMVar(self.dimensions_cfg.num_features, obj=-obs_features, ub=self.row_generation_cfg.theta_ubs, name='parameter')
            if self.row_generation_cfg.theta_lbs is not None:
                for k in range(self.dimensions_cfg.num_features):
                    if k < len(self.row_generation_cfg.theta_lbs) and self.row_generation_cfg.theta_lbs[k] is not None:
                        theta[k].lb = float(self.row_generation_cfg.theta_lbs[k])
            else:
                theta.lb = 0.0
            if hasattr(self, '_theta_init_for_start') and self._theta_init_for_start is not None:
                theta.Start = self._theta_init_for_start
            if hasattr(self, '_agent_weights') and self._agent_weights is not None:
                u_obj = np.tile(self._agent_weights, self.dimensions_cfg.num_simulations)
                u = self.master_model.addMVar(self.dimensions_cfg.num_simulations * self.dimensions_cfg.num_obs, obj=u_obj, name='utility')
            else:
                u = self.master_model.addMVar(self.dimensions_cfg.num_simulations * self.dimensions_cfg.num_obs, obj=1, name='utility')
            self.master_variables = (theta, u)
            if initial_constraints is not None and len(initial_constraints.get('indices', [])) > 0:
                self.add_constraints(initial_constraints['indices'], initial_constraints['bundles'])
            if self.row_generation_cfg.master_init_callback is not None:
                self.row_generation_cfg.master_init_callback(self.master_model, theta, u)
            self.master_model.optimize()
            logger.info('Master Initialized')
            if self.master_model.Status == GRB.OPTIMAL:
                self.theta_val = theta.X
            else:
                logger.warning('Master problem not optimal at initialization, status=%s', self.master_model.Status)
                self.theta_val = np.zeros(self.dimensions_cfg.num_features, dtype=np.float64)
            self.log_parameter()
        else:
            self.theta_val = np.empty(self.dimensions_cfg.num_features, dtype=np.float64)
        self.theta_val = self.comm_manager._broadcast_array(self.theta_val, root=0)

    def _master_iteration(self, local_pricing_results):
        from mpi4py import MPI
        features_local = self.oracles_manager.compute_rank_features(local_pricing_results)
        errors_local = self.oracles_manager.compute_rank_errors(local_pricing_results)
        global_indices_local = self.data_manager.local_data['global_indices']
        all_counts = self.data_manager.local_data['agent_counts']
        if self.comm_manager._is_root():
            theta, u = self.master_variables
            u_master_all = u.X
            theta_current = theta.X
        else:
            u_master_all = np.empty(self.dimensions_cfg.num_simulations * self.dimensions_cfg.num_obs, dtype=np.float64)
            theta_current = np.empty(self.dimensions_cfg.num_features, dtype=np.float64)
        u_master_local = self.comm_manager._scatter_array_by_row(u_master_all, counts=all_counts, dtype=np.float64)
        theta_current = self.comm_manager._broadcast_array(theta_current, root=0)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            u_sim_local = features_local @ theta_current + errors_local
        local_violations = np.where(~np.isclose(u_master_local, u_sim_local, rtol=1e-05, atol=1e-05) & (u_master_local > u_sim_local))[0]
        if len(local_violations) > 0:
            logger.warning('Rank %d: Possible failure of demand oracle at local ids: %s', self.comm_manager.rank, local_violations[:10])
        reduced_costs_local = u_sim_local - u_master_local
        local_max_rc = reduced_costs_local.max() if len(reduced_costs_local) > 0 else -np.inf
        max_reduced_cost = self.comm_manager.comm.allreduce(local_max_rc, op=MPI.MAX)
        tol_opt = self.row_generation_cfg.tolerance_optimality
        tol_rg = self.row_generation_cfg.tol_row_generation
        local_rows_to_add = np.where(u_sim_local > u_master_local * (1 + tol_rg) + tol_opt)[0]
        viol_global_ids = global_indices_local[local_rows_to_add]
        viol_bundles = local_pricing_results[local_rows_to_add]
        viol_features = features_local[local_rows_to_add]
        viol_errors = errors_local[local_rows_to_add]
        all_viol_ids = self.comm_manager._gather_array_by_row(viol_global_ids, root=0)
        all_viol_bundles = self.comm_manager._gather_array_by_row(viol_bundles, root=0)
        all_viol_features = self.comm_manager._gather_array_by_row(viol_features, root=0)
        all_viol_errors = self.comm_manager._gather_array_by_row(viol_errors, root=0)
        stop = False
        if self.comm_manager._is_root():
            self.log_parameter()
            logger.info(f'ObjVal: {self.master_model.ObjVal}')
            logger.info('Reduced cost: %s', max_reduced_cost)
            suboptimal_mode = getattr(self.subproblem_manager, '_suboptimal_mode', False)
            if max_reduced_cost < tol_opt:
                if not suboptimal_mode:
                    stop = True
                else:
                    logger.info('Reduced cost below tolerance, but suboptimal cuts mode active - continuing')
            num_new = len(all_viol_ids) if all_viol_ids is not None else 0
            logger.info('New constraints: %d', num_new)
            if num_new > 0:
                for i in range(num_new):
                    idx = int(all_viol_ids[i])
                    constr = self.master_model.addConstr(u[idx] >= all_viol_errors[i] + all_viol_features[i] @ theta)
                    self.constraint_info[constr] = (idx, all_viol_bundles[i].copy())
            self._enforce_slack_counter()
            logger.info('Number of constraints: %d', self.master_model.NumConstrs)
            self.master_model.optimize()
            theta_val = theta.X
            self.row_generation_cfg.tol_row_generation *= self.row_generation_cfg.row_generation_decay
        else:
            theta_val = None
            stop = False
        if self.comm_manager._is_root():
            theta_to_broadcast = theta_val
        else:
            theta_to_broadcast = np.empty(self.dimensions_cfg.num_features, dtype=np.float64)
        self.theta_val, stop = self.comm_manager.Bcast(theta_to_broadcast, root=0)
        return stop

    def solve(self, callback=None, theta_init=None, agent_weights=None, initial_constraints=None):
        if self.comm_manager._is_root():
            lines = ['=' * 70, 'ROW GENERATION', '=' * 70, '']
            lines.append(f'  Problem: {self.dimensions_cfg.num_obs} agents × {self.dimensions_cfg.num_items} items, {self.dimensions_cfg.num_features} features')
            if self.dimensions_cfg.num_simulations > 1:
                lines.append(f'  Simulations: {self.dimensions_cfg.num_simulations}')
            lines.append(f"  Max iterations: {(self.row_generation_cfg.max_iters if self.row_generation_cfg.max_iters != float('inf') else '∞')}")
            lines.append(f'  Min iterations: {self.row_generation_cfg.min_iters}')
            lines.append(f'  Optimality tolerance: {self.row_generation_cfg.tolerance_optimality}')
            if self.row_generation_cfg.max_slack_counter < float('inf'):
                lines.append(f'  Max slack counter: {self.row_generation_cfg.max_slack_counter}')
            if self.row_generation_cfg.tol_row_generation > 0:
                lines.append(f'  Row generation tolerance: {self.row_generation_cfg.tol_row_generation}')
            if self.row_generation_cfg.row_generation_decay > 0:
                lines.append(f'  Tolerance decay: {self.row_generation_cfg.row_generation_decay}')
            lines.append('')
            lines.append('  Starting row generation algorithm...')
            if agent_weights is not None:
                lines.append('  Using agent weights (Bayesian bootstrap)')
            lines.append('')
            logger.info('\n'.join(lines))
        if agent_weights is not None and self.comm_manager._is_root():
            self._agent_weights = np.asarray(agent_weights, dtype=np.float64)
        else:
            self._agent_weights = None
        tic = time.perf_counter()
        self.subproblem_manager.initialize_local()
        if initial_constraints is not None:
            if self.comm_manager._is_root():
                n_init = len(initial_constraints.get('indices', []))
                logger.info('Using %d provided initial constraints (warm start)', n_init)
        elif theta_init is not None:
            if self.comm_manager._is_root():
                logger.info('Initializing with provided theta (warm start)')
                if hasattr(theta_init, 'theta_hat'):
                    theta_init_array = theta_init.theta_hat
                else:
                    theta_init_array = theta_init
                self.theta_val = np.asarray(theta_init_array, dtype=np.float64).copy()
            else:
                self.theta_val = np.empty(self.dimensions_cfg.num_features, dtype=np.float64)
            self.theta_val = self.comm_manager._broadcast_array(self.theta_val, root=0)
            local_pricing_results = self.subproblem_manager.solve_local(self.theta_val)
            bundles_sim = self.comm_manager._gather_array_by_row(local_pricing_results, root=0)
            if self.comm_manager._is_root() and bundles_sim is not None and (len(bundles_sim) > 0):
                indices = np.arange(self.dimensions_cfg.num_simulations * self.dimensions_cfg.num_obs, dtype=np.int64)
                initial_constraints = {'indices': indices, 'bundles': bundles_sim.astype(np.float64)}
                logger.info('Pre-computed %d initial constraints from theta_init', len(indices))
        if theta_init is not None:
            if hasattr(theta_init, 'theta_hat'):
                self._theta_init_for_start = theta_init.theta_hat
            else:
                self._theta_init_for_start = theta_init
        else:
            self._theta_init_for_start = None
        self._initialize_master_problem(initial_constraints=initial_constraints)
        self.slack_counter = {}
        iteration = 0
        pricing_times = []
        master_times = []
        while iteration < self.row_generation_cfg.max_iters:
            logger.info(f'ITERATION {iteration + 1}')
            if self.row_generation_cfg.subproblem_callback is not None:
                master_model = self.master_model if self.comm_manager._is_root() else None
                self.row_generation_cfg.subproblem_callback(iteration, self.subproblem_manager, master_model)
            t0 = time.perf_counter()
            local_pricing_results = self.subproblem_manager.solve_local(self.theta_val)
            pricing_time = time.perf_counter() - t0
            pricing_times.append(pricing_time)
            t1 = time.perf_counter()
            stop = self._master_iteration(local_pricing_results)
            master_time = time.perf_counter() - t1
            master_times.append(master_time)
            if callback and self.comm_manager._is_root():
                callback({'iteration': iteration + 1, 'theta': self.theta_val.copy() if self.theta_val is not None else None, 'objective': self.master_model.ObjVal if hasattr(self.master_model, 'ObjVal') else None, 'pricing_time': pricing_time, 'master_time': master_time})
            if stop and iteration >= self.row_generation_cfg.min_iters:
                break
            iteration += 1
        elapsed = time.perf_counter() - tic
        num_iters = iteration + 1
        converged = iteration < self.row_generation_cfg.max_iters
        bounds_info = self._check_bounds_hit()
        warnings_list = self._log_bounds_warnings(bounds_info)
        if self.comm_manager._is_root():
            msg = 'ended' if converged else 'reached max iterations'
            logger.info(f'Row generation {msg} after {num_iters} iterations in {elapsed:.2f} seconds.')
            obj_val = self.master_model.ObjVal if hasattr(self.master_model, 'ObjVal') else None
            self.timing_stats = make_timing_stats(elapsed, num_iters, pricing_times, master_times)
            self._log_timing_summary(self.timing_stats, obj_val, self.theta_val, header='ROW GENERATION SUMMARY')
        else:
            obj_val = None
            self.timing_stats = None
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
        n_agents = self.dimensions_cfg.num_obs
        agent_ids, sim_ids = (indices % n_agents, indices // n_agents)
        errors_tensor = input_data.get('errors')
        has_sim = errors_tensor is not None and errors_tensor.ndim == 3

        def sim_data(s):
            return {**input_data, 'errors': errors_tensor[s]} if has_sim else input_data
        features = np.stack([self.oracles_manager.features_oracle(int(agent_ids[i]), bundles[i], input_data) for i in range(len(indices))])
        errors = np.array([self.oracles_manager.error_oracle(int(agent_ids[i]), bundles[i], sim_data(int(sim_ids[i]))) for i in range(len(indices))])
        for i, idx in enumerate(indices):
            constr = self.master_model.addConstr(u[idx] >= errors[i] + features[i] @ theta)
            self.constraint_info[constr] = (idx, bundles[i].copy())
        logger.info('Added %d constraints to master problem', len(indices))

    def get_constraints(self):
        if not self.comm_manager._is_root() or self.master_model is None:
            return None
        indices = []
        bundles = []
        for constr, (idx, bundle) in self.constraint_info.items():
            indices.append(idx)
            bundles.append(bundle)
        logger.debug('get_constraints: extracted %d constraints from constraint_info', len(indices))
        if len(indices) == 0:
            return self._empty_constraints_dict()
        return {'indices': np.array(indices, dtype=np.int64), 'bundles': np.array(bundles, dtype=np.float64)}

    def get_binding_constraints(self, tolerance=1e-06):
        if not self.comm_manager._is_root() or self.master_model is None:
            return None
        indices = []
        bundles = []
        for constr in self.master_model.getConstrs():
            if constr in self.constraint_info:
                if abs(constr.Slack) <= tolerance:
                    idx, bundle = self.constraint_info[constr]
                    indices.append(idx)
                    bundles.append(bundle)
        if len(indices) == 0:
            return self._empty_constraints_dict()
        return {'indices': np.array(indices, dtype=np.int64), 'bundles': np.array(bundles, dtype=np.float64)}

    def strip_slack_constraints(self, tolerance=1e-06):
        if not self.comm_manager._is_root() or self.master_model is None:
            return 0
        to_remove = []
        for constr in self.master_model.getConstrs():
            if constr in self.constraint_info:
                if abs(constr.Slack) > tolerance:
                    to_remove.append(constr)
        for constr in to_remove:
            self.master_model.remove(constr)
            self.constraint_info.pop(constr, None)
            self.slack_counter.pop(constr, None)
        if len(to_remove) > 0:
            self.master_model.update()
            logger.info('Stripped %d slack constraints, %d binding remain', len(to_remove), self.master_model.NumConstrs)
        return len(to_remove)

    def update_objective_for_weights(self, agent_weights):
        if not self.comm_manager._is_root() or self.master_model is None:
            return
        theta, u = self.master_variables
        weights_tiled = np.tile(agent_weights, self.dimensions_cfg.num_simulations)
        obs_features = (weights_tiled[:, None] * self.agents_obs_features).sum(0)
        theta.Obj = -obs_features
        u.Obj = np.tile(agent_weights, self.dimensions_cfg.num_simulations)
        self.master_model.update()

    def solve_reuse_model(self, agent_weights, strip_slack=False, reset_lp=False):
        if self.comm_manager._is_root():
            self._agent_weights = np.asarray(agent_weights, dtype=np.float64)
        else:
            self._agent_weights = None
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
                theta, u = self.master_variables
                if self.master_model.Status == GRB.OPTIMAL:
                    self.theta_val = theta.X
                else:
                    self.theta_val = np.zeros(self.dimensions_cfg.num_features, dtype=np.float64)
            except Exception as e:
                error_msg = str(e)
                self.theta_val = np.zeros(self.dimensions_cfg.num_features, dtype=np.float64)
        else:
            self.theta_val = np.empty(self.dimensions_cfg.num_features, dtype=np.float64)
        error_msg = self.comm_manager.comm.bcast(error_msg, root=0)
        if error_msg is not None:
            raise RuntimeError(f'Root process failed: {error_msg}')
        self.theta_val = self.comm_manager._broadcast_array(self.theta_val, root=0)
        iteration = 0
        while iteration < self.row_generation_cfg.max_iters:
            local_pricing_results = self.subproblem_manager.solve_local(self.theta_val)
            stop = self._master_iteration(local_pricing_results)
            if stop and iteration >= self.row_generation_cfg.min_iters:
                break
            iteration += 1
        toc = time.perf_counter()
        num_iters = iteration + 1
        converged = iteration < self.row_generation_cfg.max_iters
        bounds_info = self._check_bounds_hit()
        warnings_list = self._log_bounds_warnings(bounds_info)
        if self.comm_manager._is_root():
            obj_val = self.master_model.ObjVal if self.master_model.Status == GRB.OPTIMAL else float('inf')
            self.timing_stats = {'total_time': toc - tic, 'num_iterations': num_iters}
        else:
            obj_val = None
            self.timing_stats = None
        result = self._create_result(self.theta_val, converged, num_iters, obj_val)
        result.warnings.extend(warnings_list)
        result.metadata['bounds_hit'] = bounds_info
        return result