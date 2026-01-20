import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from mpi4py import MPI
from bundlechoice.utils import get_logger, suppress_output
from .base import BaseEstimationManager
from .result import EstimationResult
logger = get_logger(__name__)

class RowGenerationManager(BaseEstimationManager):

    def __init__(self, comm_manager, config, data_manager, oracles_manager, subproblem_manager):
        super().__init__(comm_manager, config, data_manager, oracles_manager, subproblem_manager)
        self.master_model = None
        self.master_variables = None
        self.timing_stats = None
        self.theta_iter = None
        self.u_iter_local = None

        self.slack_counter = {}
        self.constraint_info = {}
        self.cfg = self.config.row_generation
        self.dim = self.config.dimensions
        

    
    def _initialize_master_problem(self, initial_constraints=None, theta_warmstart=None, master_init_callback=None):
        _theta_obj_coef = self._compute_theta_obj_coef(self.local_obs_weights)
        u_obj_coef = self._compute_u_obj_weights(self.local_obs_weights)
        if self.comm_manager._is_root():
            self.constraint_info = {}
            self.master_model = self._setup_gurobi_model(self.cfg.gurobi_settings)
            theta = self.master_model.addMVar(self.dim.n_features, 
                                                obj= _theta_obj_coef, 
                                                lb= self.cfg.theta_lbs,
                                                ub= self.cfg.theta_ubs, 
                                                name= 'parameter')
            u = self.master_model.addMVar(self.dim.num_agents, obj=u_obj_coef, name='utility')
            if theta_warmstart is not None:
                theta.Start = theta_warmstart
            self.master_variables = (theta, u)
            if initial_constraints is not None:
                self.add_constraints(initial_constraints['indices'], initial_constraints['bundles'])
            if master_init_callback is not None:
                master_init_callback(self.master_model, theta, u)
            self.master_model.optimize()
            if self.master_model.Status != GRB.OPTIMAL:
                raise RuntimeError('Master problem cannot be solved at initialization, status=%s', 
                                    self.master_model.Status)
        self._Bcast_theta_and_Scatterv_u_vals()


    def _Bcast_theta_and_Scatterv_u_vals(self):
        if self.comm_manager._is_root():
            theta, u = self.master_variables
            theta_iter = theta.X
            u_iter = u.X
        else:
            theta_iter = np.empty(self.dim.n_features, dtype=np.float64)
            u_iter = np.empty(self.dim.num_agents, dtype=np.float64)
        self.theta_iter = self.comm_manager.Bcast(theta_iter)
        self.u_iter_local = self.comm_manager.Scatterv_by_row(u_iter, 
                                                              row_counts=self.data_manager.agent_counts,
                                                              dtype=np.float64,
                                                              shape=(self.dim.num_agents,))

    def _master_iteration(self, pricing_results, iteration, callback=None):
        features_local = self.oracles_manager.features_oracle(pricing_results)
        errors_local = self.oracles_manager.error_oracle(pricing_results)
        u_local = features_local @ self.theta_iter + errors_local
        local_reduced_costs = u_local - self.u_iter_local
        reduced_cost = self.comm_manager.Reduce(local_reduced_costs.max(0), op=MPI.MAX)
        stop = (reduced_cost[0] <= self.cfg.tol_row_generation) if self.comm_manager._is_root() else None
        stop = self.comm_manager.bcast(stop)
        if stop:
            suboptimal_mode = getattr(self.subproblem_manager, '_suboptimal_mode', False)
            if suboptimal_mode:
                logger.info('Reduced cost below tolerance, but suboptimal cuts mode active - continuing')
                return False
            return True

        local_violations = np.where(local_reduced_costs > self.cfg.tol_row_generation)[0]
        local_violations_id = self.data_manager.obs_ids[local_violations]
        violation_counts = self.comm_manager.Allgather(np.array([len(local_violations)], dtype=np.int64)).flatten()
        
        bundles = self.comm_manager.Gatherv_by_row(pricing_results[local_violations], row_counts=violation_counts)
        features = self.comm_manager.Gatherv_by_row(features_local[local_violations], row_counts=violation_counts)
        errors = self.comm_manager.Gatherv_by_row(errors_local[local_violations], row_counts=violation_counts)
        violations_id = self.comm_manager.Gatherv_by_row(local_violations_id, row_counts=violation_counts)
        
        if self.comm_manager._is_root():
            self.add_constraints(violations_id, bundles, features, errors)
            self._enforce_slack_counter()
            self.master_model.optimize()
            self.cfg.tol_row_generation *= self.cfg.row_generation_decay
        self._Bcast_theta_and_Scatterv_u_vals()
        if callback is not None:
            callback(iteration, self.subproblem_manager, self.master_model)
        return stop

    def _row_generation_iteration(self, iteration, master_iteration_callback):
        t0 = time.perf_counter()
        pricing_results = self.subproblem_manager.solve_subproblems(self.theta_iter)
        self.pricing_times.append(time.perf_counter() - t0)
        t1 = time.perf_counter()
        stop = self._master_iteration(pricing_results, iteration, master_iteration_callback)
        self.master_times.append(time.perf_counter() - t1)
        return stop

    def solve(self, local_obs_weights=None,
                    theta_warmstart=None,  
                    initial_constraints=None, 
                    initialize_master = True,
                    init_subproblems = True,
                    master_iteration_callback=None,
                    master_init_callback=None,
                    verbose = None):

        self.verbose = verbose if verbose is not None else True
        self._local_obs_weights = local_obs_weights
        self.subproblem_manager.initialize_subproblems() if init_subproblems else None  

        if initialize_master:
            self._initialize_master_problem(initial_constraints, theta_warmstart, master_init_callback) 
        else:
            has_master = self.comm_manager.bcast(self.master_variables is not None if self.comm_manager._is_root() else None)
            if has_master:
                if local_obs_weights is not None:
                    self.update_objective_for_weights(local_obs_weights)
                if self.comm_manager._is_root():
                    self.master_model.optimize()
                self._Bcast_theta_and_Scatterv_u_vals()
            else:
                raise RuntimeError('initialize_master was set to False and no master_variables values where found.')
        # if self.comm_manager._is_root():
        #     theta_coeff = self.master_variables[0].Obj
        #     u_coeff = self.master_variables[1].Obj
        #     logger.info(f"before rg loop: {theta_coeff}")
        #     logger.info(f"before rg loop: {u_coeff}")

        result = self.row_generation_loop(master_iteration_callback)
        return result

    def row_generation_loop(self, master_iteration_callback):
        iteration, self.pricing_times, self.master_times, t0 = 0, [], [], time.perf_counter()
        while iteration < self.cfg.max_iters:
            stop = self._row_generation_iteration(iteration, master_iteration_callback)
            if stop and iteration >= self.cfg.min_iters:
                break
            iteration += 1
        elapsed = time.perf_counter() - t0
        self._log_summary(iteration + 1, elapsed)
        result = self._create_result(iteration + 1)
        return result




    def add_constraints(self, indices, bundles, features, errors):
        if not self.comm_manager._is_root() or self.master_model is None:
            return
        theta, u = self.master_variables
        constr = self.master_model.addConstr(u[indices] >= features @ theta + errors)
        self.constraint_info[constr] = (indices, bundles.copy())
        if self.verbose:        
            logger.info('Added %d constraints', len(indices))
        return constr
    
    def _setup_gurobi_model(self, gurobi_settings=None):
        params = {"Method": 0, "LPWarmStart": 2, "OutputFlag": 0, **(gurobi_settings or {})}
        with suppress_output():
            model = gp.Model()
            for k, v in params.items():
                if v is not None:
                    model.setParam(k, v)
        return model

    def update_objective_for_weights(self, local_obs_weights):

        _theta_obj_coef = self._compute_theta_obj_coef(local_obs_weights)
        _u_obj_weights = self._compute_u_obj_weights(local_obs_weights)
        if not self.comm_manager._is_root() or self.master_model is None:
            return
        theta, u = self.master_variables
        theta.Obj = _theta_obj_coef
        u.Obj = _u_obj_weights
        self.master_model.update()
        self.master_model.reset(0)


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
        if to_remove and self.verbose:
            logger.info('Removed %d slack constraints', len(to_remove))
        return len(to_remove)



    def strip_nonbasic_constraints(self):
        if not self.comm_manager._is_root() or self.master_model is None:
            return 0
        to_remove = [c for c in self.master_model.getConstrs() 
                    if c in self.constraint_info and c.CBasis == -1]
        for constr in to_remove:
            self.master_model.remove(constr)
            self.constraint_info.pop(constr, None)
            self.slack_counter.pop(constr, None)
        if to_remove:
            self.master_model.update()
            if self.verbose:
                logger.info('Stripped %d slack constraints', len(to_remove))
        return len(to_remove)



    def _log_summary(self, n_iters, total):
        if not self.comm_manager._is_root() or not self.verbose:
            return
        idx = self.cfg.parameters_to_log or range(len(self.theta_iter))
        logger.info("-"*55)
        vals = ', '.join(f'θ[{i}]={self.theta_iter[i]:.5f}' for i in idx)
        logger.info('Estimated Parameters:')
        logger.info(f'  {vals}')
        logger.info(f'Objective Value = {self.master_model.ObjVal}')

        p, m = np.array(self.pricing_times), np.array(self.master_times)
        logger.info('Timing Summary:')
        logger.info(f'  Terminated after {n_iters} iterations in {total:.1f}s')
        logger.info(f'  {"total":>17}  {"avg":>8}  {"range":>8}')
        logger.info(f'  pricing  {p.sum():>7.2f}s  {p.mean():>7.3f}s  [{p.min():.3f}, {p.max():.3f}]')
        logger.info(f'  master   {m.sum():>7.2f}s  {m.mean():>7.3f}s  [{m.min():.3f}, {m.max():.3f}]')
        logger.info("-"*55)
        

    # def get_binding_constraints(self, tolerance=1e-06):
    #     if not self.comm_manager._is_root() or self.master_model is None:
    #         return None
    #     indices, bundles = [], []
    #     for constr in self.master_model.getConstrs():
    #         if constr in self.constraint_info and abs(constr.Slack) <= tolerance:
    #             idx, bundle = self.constraint_info[constr]
    #             indices.append(idx)
    #             bundles.append(bundle)
    #     return {'indices': np.array(indices, dtype=np.int64), 'bundles': np.array(bundles, dtype=np.bool_)}


    def _check_bounds_hit(self, tol=None):
        empty = {'hit_lower': [], 'hit_upper': [], 'any_hit': False}
        if not self.comm_manager._is_root() or self.master_model is None:
            return empty
        theta = self.master_variables[0]

        if tol is None:
            tol = max(1e-8, self.master_model.Params.FeasibilityTol)

        hit_lower = [k for k in range(self.config.dimensions.n_features)
                    if theta[k].LB > -GRB.INFINITY and (theta[k].X - theta[k].LB) <= tol]

        hit_upper = [k for k in range(self.config.dimensions.n_features)
                    if theta[k].UB <  GRB.INFINITY and (theta[k].UB - theta[k].X) <= tol]

        return {'hit_lower': hit_lower, 'hit_upper': hit_upper, 'any_hit': bool(hit_lower or hit_upper)}


    def _create_result(self, num_iterations = None):
        if self.comm_manager._is_root():
            converged = num_iterations < self.cfg.max_iters if num_iterations is not None else None
            return EstimationResult(
                theta_hat=self.theta_iter, converged=converged, num_iterations=num_iterations,
                final_objective=self.master_model.ObjVal,
                timing= (self.pricing_times, self.master_times),
                warnings=None)
