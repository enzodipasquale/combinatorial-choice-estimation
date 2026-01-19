import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from bundlechoice.utils import get_logger, suppress_output
from .base import BaseEstimationManager
logger = get_logger(__name__)

class RowGeneration1SlackManager(BaseEstimationManager):

    def __init__(self, comm_manager, config, data_manager, oracles_manager, subproblem_manager):
        super().__init__(comm_manager, config, data_manager, oracles_manager, subproblem_manager)
        self.master_model = None
        self.master_variables = None
        self.theta_val = None
        self.theta_hat = None
        self.slack_counter = None
        self.timing_stats = None
        self.cfg = self.config.row_generation
        self.dim = self.config.dimensions

    def _initialize_master_problem(self):
        obs_features = self._compute_theta_obj_coef()
        if self.comm_manager._is_root():
            self.master_model = self._setup_gurobi_model(self.cfg.gurobi_settings)
            theta = self.master_model.addMVar(self.dim.n_features, obj=-obs_features, 
                                               ub=self.cfg.theta_ubs, name='parameter')
            theta.lb = self.cfg.theta_lbs if self.cfg.theta_lbs is not None else 0.0
            u_bar = self.master_model.addVar(obj=1, name='utility')
            self.master_model.optimize()
            logger.info('Master Initialized (1slack formulation)')
            self.master_variables = (theta, u_bar)
            self.theta_val = theta.X
        else:
            self.theta_val = np.empty(self.dim.n_features, dtype=np.float64)
        self.comm_manager.Bcast(self.theta_val)

    def _master_iteration(self, optimal_bundles):
        features = self.oracles_manager.features_oracle(optimal_bundles)
        errors = self.oracles_manager.error_oracle(optimal_bundles)
        features_all = self.comm_manager.Gatherv_by_row(features, row_counts=self.data_manager.agent_counts)
        errors_all = self.comm_manager.Gatherv_by_row(errors, row_counts=self.data_manager.agent_counts)
        
        stop = False
        if self.comm_manager._is_root():
            theta, u_bar = self.master_variables
            u_sim = (features_all @ theta.X).sum() + errors_all.sum()
            u_master = u_bar.X
            logger.info(f'ObjVal: {self.master_model.ObjVal}')
            reduced_cost = u_sim - u_master
            logger.info('Reduced cost: %s', reduced_cost)
            
            if reduced_cost < self.cfg.tolerance_optimality:
                stop = True
            elif u_sim > u_master * (1 + self.cfg.tol_row_generation) + self.cfg.tolerance_optimality:
                agents_utilities = features_all @ theta + errors_all.sum()
                self.master_model.addConstr(u_bar >= agents_utilities.sum())
                self._enforce_slack_counter()
                logger.info('Number of constraints: %d', self.master_model.NumConstrs)
                self.master_model.optimize()
                self.cfg.tol_row_generation *= self.cfg.row_generation_decay
            theta_val = theta.X
        else:
            theta_val = np.empty(self.dim.n_features, dtype=np.float64)
            
        self.comm_manager.Bcast(theta_val)
        self.theta_val = theta_val
        stop = self.comm_manager.bcast(stop)
        return stop

    def solve(self, callback=None):
        logger.info('=== ROW GENERATION (1SLACK) ===')
        t0 = time.perf_counter()
        self.subproblem_manager.initialize_subproblems()
        self._initialize_master_problem()
        self.slack_counter = {}
        
        iteration = 0
        while iteration < self.cfg.max_iters:
            logger.info(f'ITERATION {iteration + 1}')
            optimal_bundles = self.subproblem_manager.solve_subproblems(self.theta_val)
            stop = self._master_iteration(optimal_bundles)
            if stop and iteration >= self.cfg.min_iters:
                break
            iteration += 1
            
        elapsed = time.perf_counter() - t0
        num_iters = iteration + 1
        converged = iteration < self.cfg.max_iters
        
        if self.comm_manager._is_root():
            msg = 'ended' if converged else 'reached max iterations'
            logger.info(f'Row generation (1slack) {msg} after {num_iters} iterations in {elapsed:.2f} seconds.')
            
        self.theta_hat = self.theta_val.copy()
        return self._create_result(num_iters, self.master_model, self.theta_hat, self.cfg)

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
