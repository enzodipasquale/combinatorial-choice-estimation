import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from mpi4py import MPI
from bundlechoice.utils import get_logger, suppress_output
from .base import BaseEstimationManager
logger = get_logger(__name__)

class RowGenerationManager(BaseEstimationManager):

    def __init__(self, comm_manager, config, data_manager, oracles_manager, subproblem_manager):
        super().__init__(comm_manager, config, data_manager, oracles_manager, subproblem_manager)
        self.master_model = None
        self.master_variables = None
        self.timing_stats = None
        self.theta_iter = None
        self.theta_sol = None
        self.u_iter = None
        self.u_iter_local = None

        self.slack_counter = {}
        self.constraint_info = {}
        self.cfg = self.config.row_generation
        self.dim = self.config.dimensions

    def _initialize_master_problem(self, initial_constraints=None, theta_warmstart=None):
        theta_obj_coef = self._compute_theta_obj_coef(self.obs_weights)
        u_obj_coef = self._compute_u_obj_weights(self.obs_weights)
        if self.comm_manager._is_root():
            self.constraint_info = {}
            self.master_model = self._setup_gurobi_model(self.cfg.gurobi_settings)
            theta = self.master_model.addMVar(self.dim.n_features, 
                                                obj= theta_obj_coef, 
                                                lb= self.cfg.theta_lbs,
                                                ub= self.cfg.theta_ubs, 
                                                name= 'parameter')
            u = self.master_model.addMVar(self.dim.num_agents, obj=u_obj_coef, name='utility')
            master_variables = (theta, u)
            if theta_warmstart is not None:
                theta.Start = theta_warmstart
            self.master_variables = (theta, u)
            if initial_constraints is not None:
                self.add_constraints(initial_constraints['indices'], initial_constraints['bundles'])
            if self.cfg.master_init_callback is not None:
                self.cfg.master_init_callback(self.master_model, master_variables)
            self.master_model.optimize()
            if self.master_model.Status != GRB.OPTIMAL:
                raise RuntimeError('Master problem not optimal at initialization, status=%s', 
                                    self.master_model.Status)
            self.theta_iter = theta.X
            u_iter = u.X
        else:
            self.theta_iter = np.empty(self.dim.n_features, dtype=np.float64)
            u_iter = None
        self.comm_manager.Bcast(self.theta_iter)
        self.u_iter_local = self.comm_manager.Scatterv_by_row(u_iter, 
                                                              row_counts=self.data_manager.agent_counts,
                                                              dtype=np.float64,
                                                              shape=(self.dim.num_agents,))

    def _master_iteration(self, pricing_results):
        features_local = self.oracles_manager.features_oracle(pricing_results)
        errors_local = self.oracles_manager.error_oracle(pricing_results)
        u_local = features_local @ self.theta_iter + errors_local
        local_reduced_costs = u_local - self.u_iter_local
        reduced_cost = self.comm_manager.Reduce(local_reduced_costs.max(0), op=MPI.MAX)
        if self.comm_manager._is_root():
            print("obj val:", self.master_model.ObjVal)
            print("num constraints:", self.master_model.NumConstrs)
            print("reduced cost:", reduced_cost)
        stop = (reduced_cost < self.cfg.tol_row_generation).all() if self.comm_manager._is_root() else None
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
            theta_iter = self.master_variables[0].X
            u_iter = self.master_variables[1].X
            self.cfg.tol_row_generation *= self.cfg.row_generation_decay
        else:
            theta_iter = np.empty(self.dim.n_features, dtype=np.float64)
            u_iter = np.empty(self.dim.num_agents, dtype=np.float64)
        self.theta_iter = self.comm_manager.Bcast(theta_iter)
        self.u_iter_local = self.comm_manager.Scatterv_by_row(u_iter, 
                                                              row_counts=self.data_manager.agent_counts,
                                                              dtype=np.float64,
                                                              shape=(self.dim.num_agents,))
        return stop

    def solve(self, callback=None, 
                    obs_weights=None,
                    theta_warmstart=None,  
                    initial_constraints=None, 
                    init_master = True,
                    init_subproblems = True):

        self.obs_weights = obs_weights
        self.subproblem_manager.initialize_subproblems() if init_subproblems else None  
        self._initialize_master_problem(initial_constraints, theta_warmstart) if init_master else None
        
        iteration, pricing_times, master_times, t0 = 0, [], [], time.perf_counter()
        while iteration < 10:#self.cfg.max_iters:
            
            if self.cfg.subproblem_callback is not None:
                self.cfg.subproblem_callback(iteration, self.subproblem_manager, self.master_model)
            t1 = time.perf_counter()
            if self.comm_manager._is_root():
                print("iteration", iteration)
            
            pricing_results = self.subproblem_manager.solve_subproblems(self.theta_iter)
            pricing_times.append(time.perf_counter() - t1)
            t2 = time.perf_counter()
            stop = self._master_iteration(pricing_results)
            master_times.append(time.perf_counter() - t2)
            if stop and iteration >= self.cfg.min_iters:
                break
            iteration += 1
        elapsed = time.perf_counter() - t0
        result = self._create_result(iteration +1, self.master_model, self.theta_iter, self.cfg)
        return result


    #########


    def add_constraints(self, indices, bundles, features, errors):
        if not self.comm_manager._is_root() or self.master_model is None:
            return
        theta, u = self.master_variables
        constr = self.master_model.addConstr(u[indices] >= features @ theta + errors)
        self.constraint_info[constr] = (indices, bundles.copy())
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

    def update_objective_for_weights(self, obs_weights):
        if not self.comm_manager._is_root() or self.master_model is None:
            return
        theta, u = self.master_variables
        weights_tiled = np.tile(obs_weights, self.config.dimensions.n_simulations)
        theta.Obj = - weights_tiled * self.theta_obj_coef
        u.Obj = weights_tiled
        self.master_model.update()


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
        if to_remove:
            logger.info('Removed %d slack constraints', len(to_remove))
        return len(to_remove)



    # def strip_slack_constraints(self, tolerance=1e-06):
    #     if not self.comm_manager._is_root() or self.master_model is None:
    #         return 0
    #     to_remove = [c for c in self.master_model.getConstrs() 
    #                 if c in self.constraint_info and abs(c.Slack) > tolerance]
    #     for constr in to_remove:
    #         self.master_model.remove(constr)
    #         self.constraint_info.pop(constr, None)
    #         self.slack_counter.pop(constr, None)
    #     if to_remove:
    #         self.master_model.update()
    #         logger.info('Stripped %d slack constraints', len(to_remove))
    #     return len(to_remove)


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


    # def _check_bounds_hit(self, tol=1e-06):
    #     empty = {'hit_lower': [], 'hit_upper': [], 'any_hit': False}
    #     if not self.comm_manager._is_root() or self.master_model is None:
    #         return empty
    #     theta = self.master_variables[0]
    #     hit_lower = [k for k in range(self.config.dimensions.n_features) 
    #                 if abs(theta[k].X - theta[k].LB) < tol]
    #     hit_upper = [k for k in range(self.config.dimensions.n_features) 
    #                 if abs(theta[k].X - theta[k].UB) < tol]
    #     return {'hit_lower': hit_lower, 'hit_upper': hit_upper, 'any_hit': bool(hit_lower or hit_upper)}


    # def _log_bounds_warnings(self, bounds_info):
    #     warnings_list = []
    #     if self.comm_manager._is_root() and bounds_info["any_hit"]:
    #         for bound_type in ["lower", "upper"]:
    #             if bounds_info[f"hit_{bound_type}"]:
    #                 msg = f"Theta hit {bound_type.upper()} bound at indices: {bounds_info[f"hit_{bound_type}"]}"
    #                 logger.warning(msg)
    #                 warnings_list.append(msg)
    #     return warnings_list
