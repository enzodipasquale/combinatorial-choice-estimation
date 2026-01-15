import time
import numpy as np
from typing import Optional, Any, Dict
from numpy.typing import NDArray
import gurobipy as gp
from gurobipy import GRB
from bundlechoice.utils import get_logger, suppress_output, make_timing_stats
from .base import BaseEstimationManager
from .result import EstimationResult
logger = get_logger(__name__)

class RowGeneration1SlackManager(BaseEstimationManager):

    def __init__(self, comm_manager, dimensions_cfg, row_generation_cfg, data_manager, oracles_manager, subproblem_manager):
        super().__init__(comm_manager=comm_manager, dimensions_cfg=dimensions_cfg, data_manager=data_manager, oracles_manager=oracles_manager, subproblem_manager=subproblem_manager)
        self.row_generation_cfg = row_generation_cfg
        self.master_model = None
        self.master_variables = None
        self.theta_val = None
        self.theta_hat = None
        self.slack_counter = None
        self.timing_stats = None

    def _setup_gurobi_model_params(self):
        return self._setup_gurobi_model(self.row_generation_cfg.gurobi_settings)

    def _initialize_master_problem(self):
        obs_features = self.get_obs_features()
        if self.comm_manager._is_root():
            self.master_model = self._setup_gurobi_model_params()
            theta = self.master_model.addMVar(self.dimensions_cfg.num_features, obj=-obs_features, ub=self.row_generation_cfg.theta_ubs, name='parameter')
            if self.row_generation_cfg.theta_lbs is not None:
                theta.lb = self.row_generation_cfg.theta_lbs
            else:
                theta.lb = 0.0
            u_bar = self.master_model.addVar(obj=1, name='utility')
            self.master_model.optimize()
            logger.info('Master Initialized (1slack formulation)')
            self.master_variables = (theta, u_bar)
            self.theta_val = theta.X
            self.log_parameter()
        else:
            self.theta_val = np.empty(self.dimensions_cfg.num_features, dtype=np.float64)
        self.theta_val = self.comm_manager._broadcast_array(self.theta_val, root=0)

    def _master_iteration(self, optimal_bundles):
        x_sim = self.oracles_manager.compute_gathered_features(optimal_bundles)
        errors_sim = self.oracles_manager.compute_gathered_errors(optimal_bundles)
        stop = False
        if self.comm_manager._is_root():
            theta, u_bar = self.master_variables
            u_sim = (x_sim @ theta.X).sum() + errors_sim.sum()
            u_master = u_bar.X
            self.log_parameter()
            logger.info(f'ObjVal: {self.master_model.ObjVal}')
            reduced_cost = u_sim - u_master
            logger.info('Reduced cost: %s', reduced_cost)
            if reduced_cost < self.row_generation_cfg.tolerance_optimality:
                stop = True
            elif u_sim > u_master * (1 + self.row_generation_cfg.tol_row_generation) + self.row_generation_cfg.tolerance_optimality:
                agents_utilities = (x_sim @ theta).sum() + errors_sim.sum()
                self.master_model.addConstr(u_bar >= agents_utilities)
                self._enforce_slack_counter()
                logger.info('Number of constraints: %d', self.master_model.NumConstrs)
                self.master_model.optimize()
                self.row_generation_cfg.tol_row_generation *= self.row_generation_cfg.row_generation_decay
            theta_val = theta.X
        else:
            theta_val = np.empty(self.dimensions_cfg.num_features, dtype=np.float64)
        self.theta_val, stop = self.comm_manager.Bcast(theta_val, root=0)
        return stop

    def solve(self):
        if self.comm_manager._is_root():
            lines = ['=' * 70, 'ROW GENERATION (1SLACK)', '=' * 70, '']
            lines.append(f'  Problem: {self.dimensions_cfg.num_obs} agents × {self.dimensions_cfg.num_items} items, {self.dimensions_cfg.num_features} features')
            if self.dimensions_cfg.num_simulations > 1:
                lines.append(f'  Simulations: {self.dimensions_cfg.num_simulations}')
            lines.append(f"  Max iterations: {(self.row_generation_cfg.max_iters if self.row_generation_cfg.max_iters != float('inf') else '∞')}")
            lines.append(f'  Min iterations: {self.row_generation_cfg.min_iters}')
            lines.append(f'  Optimality tolerance: {self.row_generation_cfg.tolerance_optimality}')
            if self.row_generation_cfg.max_slack_counter < float('inf'):
                lines.append(f'  Max slack counter: {self.row_generation_cfg.max_slack_counter}')
            lines.append('')
            lines.append('  Starting row generation algorithm (1slack formulation)...')
            lines.append('')
            logger.info('\n'.join(lines))
        tic = time.perf_counter()
        self.subproblem_manager.initialize_local()
        self._initialize_master_problem()
        self.slack_counter = {}
        iteration = 0
        total_pricing = 0.0
        while iteration < self.row_generation_cfg.max_iters:
            logger.info(f'ITERATION {iteration + 1}')
            t0 = time.perf_counter()
            optimal_bundles = self.subproblem_manager.solve_local(self.theta_val)
            total_pricing += time.perf_counter() - t0
            stop = self._master_iteration(optimal_bundles)
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
            logger.info(f'Row generation (1slack) {msg} after {num_iters} iterations in {elapsed:.2f} seconds.')
            obj_val = self.master_model.ObjVal if hasattr(self.master_model, 'ObjVal') else None
            self.timing_stats = make_timing_stats(elapsed, num_iters, total_pricing)
            self._log_timing_summary(self.timing_stats, obj_val, self.theta_val, header='ROW GENERATION (1-SLACK) SUMMARY')
        else:
            obj_val = None
            self.timing_stats = None
        self.theta_hat = self.theta_val.copy()
        result = self._create_result(self.theta_hat, converged, num_iters, obj_val)
        result.warnings.extend(warnings_list)
        result.metadata['bounds_hit'] = bounds_info
        return result