import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from mpi4py import MPI
from bundlechoice.utils import get_logger, suppress_output, format_number
from .base import BaseEstimationManager
from .result import EstimationResult
logger = get_logger(__name__)

class RowGenerationManager(BaseEstimationManager):

    def __init__(self, comm_manager, config, data_manager, oracles_manager, subproblem_manager):
        super().__init__(comm_manager, config, data_manager, oracles_manager, subproblem_manager)
        self.master_model = None
        self.master_variables = None

        self.theta_iter = None
        self.u_iter_local = None

        self.cfg = self.config.row_generation
        self.dim = self.config.dimensions

    @property
    def has_master_vars(self):
        return self.comm_manager.bcast(self.master_model is not None)

    
    def _initialize_master_problem(self):
        _theta_obj_coef = self._compute_theta_obj_coef(self.local_obs_weights)
        u_obj_coef = self._compute_u_obj_weights(self.local_obs_weights)
        if self.comm_manager.is_root():
            self.master_model = self._setup_gurobi_model(self.cfg.master_GRB_settings)
            theta = self.master_model.addMVar(self.dim.n_features, 
                                                obj= _theta_obj_coef, 
                                                lb= self.cfg.theta_lbs,
                                                ub= self.cfg.theta_ubs, 
                                                name= 'parameter')
            u = self.master_model.addMVar(self.dim.n_agents, obj=u_obj_coef, name='utility')
            self.master_variables = (theta, u)
            self.master_model.optimize()

            if self.master_model.Status != GRB.OPTIMAL:
                raise RuntimeError('Master problem cannot be solved at initialization, status=%s', 
                                    self.master_model.Status)
        


    def _Bcast_theta_and_Scatterv_u_vals(self):
        if self.comm_manager.is_root():
            theta, u = self.master_variables
            theta_iter = theta.X
            u_iter = u.X
        else:
            theta_iter = np.empty(self.dim.n_features, dtype=np.float64)
            u_iter = np.empty(self.dim.n_agents, dtype=np.float64)
        self.theta_iter = self.comm_manager.Bcast(theta_iter)
        self.u_iter_local = self.comm_manager.Scatterv_by_row(u_iter, 
                                                              row_counts=self.data_manager.agent_counts,
                                                              dtype=np.float64,
                                                              shape=(self.dim.n_agents,))


    def _update_iteration_info(self, iteration, **kwargs):
        if self.comm_manager.is_root():
            if iteration not in self.iteration_history:
                self.iteration_history[iteration] = {}
            update_dict = {}
            update_dict.update(kwargs)
            update_dict.update({'objective': self.master_model.ObjVal, 
                                'n_constraints': self.master_model.NumConstrs})
            
            self.iteration_history[iteration].update(update_dict)

   


   
    def solve(self, local_obs_weights=None,
                    initialize_master = True,
                    initialize_subproblems = True,
                    iteration_callback=None,
                    initialization_callback=None,
                    verbose = False):
        
        
        self.verbose = verbose if verbose is not None else True
        if self.verbose:
            self._log_instance_summary()

        self.subproblem_manager.initialize_subproblems() if initialize_subproblems else None  
        if initialize_master:
            self._initialize_master_problem() 
        elif self.has_master_vars:
            if local_obs_weights is not None:
                self.update_objective_for_weights(local_obs_weights)
            if self.comm_manager.is_root():
                self.master_model.optimize()
        else:
            raise RuntimeError('initialize_master was set to False and no master_variables values where found.')
        if initialization_callback is not None:
            initialization_callback(self)

        self._Bcast_theta_and_Scatterv_u_vals()
        if self.verbose:
            logger.info(" " )
            logger.info(" ROW GENERATION")
        result = self.row_generation_loop(iteration_callback)
        return result

    def row_generation_loop(self, callback):
        iteration, self.iteration_history, t0 = 0, {}, time.perf_counter()
        while iteration < self.cfg.max_iters:
            if callback is not None:
                callback(iteration, self)
            stop = self._row_generation_iteration(iteration)
            if stop and iteration >= self.cfg.min_iters:
                break
            iteration += 1
        elapsed = time.perf_counter() - t0
        self._log_summary(iteration + 1, elapsed)
        result = self._create_result(iteration + 1)
        return result

    def _row_generation_iteration(self, iteration):
        t0 = time.perf_counter()
        pricing_results = self.subproblem_manager.solve_subproblems(self.theta_iter)
        pricing_time_local = time.perf_counter() - t0
        
        pricing_time = self.comm_manager.Reduce(np.array([pricing_time_local], dtype=np.float64),op=MPI.MAX)
        pricing_time = pricing_time[0] if self.comm_manager.is_root() else None
        
        t1 = time.perf_counter() if self.comm_manager.is_root() else None
        stop, reduced_cost, n_violations = self._master_iteration(pricing_results, iteration)
        master_time = time.perf_counter() - t1 if self.comm_manager.is_root() else None
        self._Bcast_theta_and_Scatterv_u_vals()
        self._update_iteration_info(iteration, pricing_time=pricing_time, 
                                                master_time=master_time,
                                                reduced_cost = reduced_cost,
                                                n_violations = n_violations)
        self._log_iteration(iteration)     
        return stop

    def _master_iteration(self, pricing_results, iteration):
            features_local = self.oracles_manager.features_oracle(pricing_results)
            errors_local = self.oracles_manager.error_oracle(pricing_results)
            u_local = features_local @ self.theta_iter + errors_local
            local_reduced_costs = u_local - self.u_iter_local
            reduced_cost = self.comm_manager.Reduce(local_reduced_costs.max(0), op=MPI.MAX)
            reduced_cost = reduced_cost[0] if self.comm_manager.is_root() else None
            stop = (reduced_cost <= self.cfg.tolerance) if self.comm_manager.is_root() else None
            stop = self.comm_manager.bcast(stop)
        
            local_violations = np.where(local_reduced_costs > self.cfg.tolerance)[0]
            local_violations_id = self.data_manager.agent_ids[local_violations]
            violation_counts = self.comm_manager.Allgather(np.array([len(local_violations)], dtype=np.int64)).flatten()
            
            bundles = self.comm_manager.Gatherv_by_row(pricing_results[local_violations], row_counts=violation_counts)
            features = self.comm_manager.Gatherv_by_row(features_local[local_violations], row_counts=violation_counts)
            errors = self.comm_manager.Gatherv_by_row(errors_local[local_violations], row_counts=violation_counts)
            violations_id = self.comm_manager.Gatherv_by_row(local_violations_id, row_counts=violation_counts)
            n_violations =  len(violations_id) if self.comm_manager.is_root() else None              
            if stop:
                if self.comm_manager.is_root():
                    self.master_model.optimize()
                return True, reduced_cost, n_violations
            if self.comm_manager.is_root():
                self.add_master_constraints(violations_id, bundles, features, errors)
                self.master_model.optimize()
            return False, reduced_cost, n_violations

    def add_master_constraints(self, indices, bundles, features, errors):
        if not self.comm_manager.is_root() or self.master_model is None:
            return
        theta, u = self.master_variables
        constr = self.master_model.addConstr(u[indices] >= features @ theta + errors)
        return constr
    
    def _setup_gurobi_model(self, master_GRB_settings=None):
        params = {"Method": 0, "LPWarmStart": 2, "OutputFlag": 0, **(master_GRB_settings or {})}
        with suppress_output():
            model = gp.Model()
            for k, v in params.items():
                if v is not None:
                    model.setParam(k, v)
        return model

    def update_objective_for_weights(self, local_obs_weights):
        _theta_obj_coef = self._compute_theta_obj_coef(local_obs_weights)
        _u_obj_weights = self._compute_u_obj_weights(local_obs_weights)
        if not self.comm_manager.is_root() or self.master_model is None:
            return
        theta, u = self.master_variables
        theta.Obj = _theta_obj_coef
        u.Obj = _u_obj_weights
        self.master_model.update()
        self.master_model.reset(0)

    def strip_nonbasic_constraints(self):
        if not self.comm_manager.is_root() or self.master_model is None:
            return 0
        to_remove = [c for c in self.master_model.getConstrs() if c.CBasis == 0]
        for constr in to_remove:
            self.master_model.remove(constr)
        if to_remove:
            self.master_model.update()
        return len(to_remove)

    def strip_slack_constraints(self, percentile=100, hard_threshold = float('inf')):
        if not self.comm_manager.is_root() or self.master_model is None:
            return 0
        constraints = list(self.master_model.getConstrs())
        slacks = np.array([c.Slack for c in constraints])
        threshold = np.percentile(slacks, 100.0 - percentile)
        if (slacks < threshold).sum() < len(constraints) - hard_threshold:
            return self.strip_constraints_hard_threshold(hard_threshold)
        to_remove = [c for c in constraints if c.Slack < threshold]
        for constr in to_remove:
            self.master_model.remove(constr)
        if to_remove:
            self.master_model.update()
        return len(to_remove)

    def strip_constraints_hard_threshold(self, n_constraints = float('inf')):
        if not self.comm_manager.is_root() or self.master_model is None:
            return 0
        constraints = list(self.master_model.getConstrs())
        if len(constraints) < n_constraints:
            return 0
        slacks = np.array([c.Slack for c in constraints])
        to_remove = [constraints[c_id] for c_id in np.argsort(slacks)[:-n_constraints]]
        for constr in to_remove:
            self.master_model.remove(constr)
        if to_remove:
            self.master_model.update()
        return len(to_remove)
        
        

    def _log_iteration(self, iteration):
        if not self.comm_manager.is_root() or not self.verbose:
            return
        info = self.iteration_history[iteration]   
        
        if self.cfg.parameters_to_log is not None:
            param_indices = self.cfg.parameters_to_log
        else:
            param_indices = list(range(min(5, len(self.theta_iter))))
        
        if iteration % 80 == 0:
            param_width = len(param_indices) * 11 - 1
            header1 = (f"{'Iter':>4} | {'Reduced':^12} | {'Pricing':^11} | "
                      f"{'Master':^10} | {'#Viol':^5} | {'Objective':^13} | "
                      f"{'Constr':>6} | {f'Parameters':^{param_width}}")
            param_label_row = ' '.join(f'{f"θ[{i}]":>10}' for i in param_indices)
            header2 = (f"{'':>4} | {'Cost':^12} | {'(s)':^11} | "
                      f"{'(s)':^10} | {'':>5} | {'Value':^13} | "
                      f"{'':>6} | {param_label_row}")
            logger.info(header1)
            logger.info(header2)
            logger.info("-" * len(header1))
        
        param_vals = ' '.join(format_number(self.theta_iter[i], width=10, precision=5) for i in param_indices)
        row = (f"{iteration:>4} | {format_number(info['reduced_cost'], width=12, precision=6)} | "
               f"{format_number(info['pricing_time'], width=10, precision=3)}s | "
               f"{format_number(info['master_time'], width=9, precision=3)}s | "
               f"{info['n_violations']:>5} | "
               f"{format_number(info['objective'], width=13, precision=5)} | "
               f"{info['n_constraints']:>6} | {param_vals}")
        logger.info(row)

    def _log_summary(self, n_iters, total_time):
        if not self.comm_manager.is_root() or not self.verbose:
            return
        
        idx = self.cfg.parameters_to_log or range(len(self.theta_iter))
        p = np.array([self.iteration_history[i]['pricing_time'] for i in sorted(self.iteration_history.keys())])
        m = np.array([self.iteration_history[i]['master_time'] for i in sorted(self.iteration_history.keys())])
        
        # Get final iteration info
        final_iter = max(self.iteration_history.keys())
        final_info = self.iteration_history[final_iter]
        n_constraints = final_info.get('n_constraints', self.master_model.NumConstrs)
        n_violations = final_info.get('n_violations', 0)
        reduced_cost = final_info.get('reduced_cost', 0.0)
        logger.info(" " )
        logger.info(" ROW GENERATION SUMMARY" )
        logger.info("-" * 80)
    
        param_labels = ' | '.join(f'{f"θ[{i}]":>12}' for i in idx)
        logger.info(f"{'Parameters':>17} | {param_labels}")
        param_vals = ' | '.join(format_number(self.theta_iter[i], width=12, precision=5) for i in idx)
        logger.info(f"{'':>17} | {param_vals}")
        
        logger.info("-" * 80)
        
        # Summary statistics header and values
        logger.info(f"{'ObjVal':>12} | {'#Consts':>8} | {'#Viols':>6} | {'Reduced Cost':>12} | {'Time (s)':>9} | {'#Iters':>7}")
        logger.info(
            f"{format_number(self.master_model.ObjVal, width=12, precision=5)} | "
            f"{n_constraints:>8} | "
            f"{n_violations:>6} | "
            f"{format_number(reduced_cost, width=12, precision=6)} | "
            f"{format_number(total_time, width=9, precision=3)} | "
            f"{n_iters:>7}"
        )
        
        logger.info("-" * 80)
        
        # Timing summary table
        logger.info(f"{'Time':>17} | {'Total (s)':>10} | {'Avg (s)':>10} | {'Range (s)':>11}")
      
        logger.info(
            f"{'pricing':>17} | "
            f"{format_number(p.sum(), width=10, precision=3)} | "
            f"{format_number(p.mean(), width=10, precision=3)} | "
            f"[{format_number(p.min(), precision=3)}, {format_number(p.max(), precision=3)}]"
        )
        logger.info(
            f"{'master':>17} | "
            f"{format_number(m.sum(), width=10, precision=3)} | "
            f"{format_number(m.mean(), width=10, precision=3)} | "
            f"[{format_number(m.min(), precision=3)}, {format_number(m.max(), precision=3)}]"
        )
        logger.info(" " )
        
    def _check_bounds_hit(self, tol=None):
        empty = {'hit_lower': [], 'hit_upper': [], 'any_hit': False}
        if not self.comm_manager.is_root() or self.master_model is None:
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
        if self.comm_manager.is_root():
            converged = num_iterations < self.cfg.max_iters if num_iterations is not None else None
            pricing_times = [self.iteration_history[i]['pricing_time'] for i in sorted(self.iteration_history.keys())]
            master_times = [self.iteration_history[i]['master_time'] for i in sorted(self.iteration_history.keys())]
            final_iter = max(self.iteration_history.keys())
            final_reduced_cost = self.iteration_history[final_iter].get('reduced_cost', 0.0)
            return EstimationResult(
                theta_hat=self.theta_iter, converged=converged, num_iterations=num_iterations,
                final_objective=self.master_model.ObjVal,
                n_constraints=self.master_model.NumConstrs,
                final_reduced_cost=final_reduced_cost,
                timing= (pricing_times, master_times),
                warnings=None)
