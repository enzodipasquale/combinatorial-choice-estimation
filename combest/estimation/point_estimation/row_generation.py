import os
import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from combest.utils import get_logger, suppress_output, format_number
from combest.estimation.result import RowGenerationEstimationResult

logger = get_logger(__name__)


class RowGenerationSolver:

    def __init__(self, pt_estimation_manager):
        self.pt_estimation_manager = pt_estimation_manager
        self.comm_manager = pt_estimation_manager.comm_manager
        self.config = pt_estimation_manager.config
        self.data_manager = pt_estimation_manager.data_manager
        self.features_manager = pt_estimation_manager.features_manager
        self.subproblem_manager = pt_estimation_manager.subproblem_manager
        self.cfg = self.config.row_generation
        self.dim = self.config.dimensions
        self.master_model = None
        self.master_variables = None
        self.theta_iter = None
        self.local_obs_weights = None

    # ------------------------------------------------------------------
    # Template methods
    # ------------------------------------------------------------------

    def _initialize_master(self):
        raise NotImplementedError

    def _distribute_solution(self):
        raise NotImplementedError

    def _master_iteration(self, pricing_results):
        """Returns (stop, reduced_cost, n_violations, (t1, t2))"""
        raise NotImplementedError

    def _result_u_hat(self):
        return None

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def compute_LP_coef(self, local_obs_weights=None):
        if local_obs_weights is None:
            local_obs_weights = self.data_manager.local_obs_quantity
        local_obs_covariates = self.features_manager.covariates_oracle(self.data_manager.local_obs_bundles)
        theta_coeff = self.comm_manager.sum_row_andReduce(-local_obs_weights[:, None] * local_obs_covariates)
        u_coeff = self.comm_manager.Gatherv_by_row(local_obs_weights, row_counts=self.comm_manager.agent_counts)
        return theta_coeff, u_coeff if self.comm_manager.is_root() else None


    def compute_theta_LP_coef(self, local_obs_weights=None):
        if local_obs_weights is None:
            local_obs_weights = self.data_manager.local_obs_quantity
        local_obs_covariates = self.features_manager.covariates_oracle(self.data_manager.local_obs_bundles)
        return self.comm_manager.sum_row_andReduce(-local_obs_weights[:, None] * local_obs_covariates)

    def compute_u_LP_coef(self, local_obs_weights=None):
        if local_obs_weights is None:
            local_obs_weights = self.data_manager.local_obs_quantity
        all_weights = self.comm_manager.Gatherv_by_row(local_obs_weights, row_counts=self.comm_manager.agent_counts)
        return all_weights if self.comm_manager.is_root() else None

    def solve(self, resampling_weights=None, initialize_solver=True,
              iteration_callback=None, initialization_callback=None, verbose=False):
        self.verbose = verbose
        if self.verbose:
            self.pt_estimation_manager._log_instance_summary()
        if initialize_solver:
            self.subproblem_manager.initialize_solver()
        self.local_obs_weights = resampling_weights if resampling_weights is not None \
                                 else self.data_manager.local_obs_quantity
        self._initialize_master()
        return self._run_loop(iteration_callback, initialization_callback)

    def _run_loop(self, iteration_callback=None, initialization_callback=None):
        if initialization_callback is not None:
            initialization_callback(self)
        self._distribute_solution()
        if self.verbose and self.comm_manager.is_root():
            logger.info(" ")
            logger.info(" ROW GENERATION")
        result = self.row_generation_loop(iteration_callback)
        if self.comm_manager.is_root() and self.cfg.save_master_model_dir:
            dir = self.cfg.save_master_model_dir
            os.makedirs(dir, exist_ok=True)
            self.master_model.write(os.path.join(dir, "master.lp"))
            self.master_model.write(os.path.join(dir, "master.sol"))
        return result

    # ------------------------------------------------------------------
    # Row generation loop
    # ------------------------------------------------------------------

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
        # Final solve at theta_hat so predicted_bundles are at the solution
        self.subproblem_manager.solve(self.theta_iter)
        result = self._create_result(iteration + 1, total_time=elapsed)
        if result is not None and self.verbose:
            result.log_summary(self.dim.display_indices, self.dim.covariate_labels,
                               self.dim.covariate_label_width)
        return result

    def _row_generation_iteration(self, iteration):
        t0 = time.perf_counter()
        cuts = self.pt_estimation_manager.compute_cuts(self.theta_iter)
        stop, reduced_cost, n_violations, (t1, t2) = self._master_iteration(cuts)
        pricing_time = t1 - t0
        master_time = t2 - t1 if self.comm_manager.is_root() else None
        self._distribute_solution()
        self._update_iteration_info(iteration, pricing_time=pricing_time,
                                    master_time=master_time,
                                    reduced_cost=reduced_cost,
                                    n_violations=n_violations)
        self._log_iteration(iteration)
        return stop

    # ------------------------------------------------------------------
    # Iteration bookkeeping
    # ------------------------------------------------------------------

    def _update_iteration_info(self, iteration, **kwargs):
        if self.comm_manager.is_root():
            if iteration not in self.iteration_history:
                self.iteration_history[iteration] = {}
            kwargs.update({'objective': self.master_model.ObjVal,
                           'n_constraints': self.master_model.NumConstrs})
            self.iteration_history[iteration].update(kwargs)

    def _log_iteration(self, iteration):
        if not self.comm_manager.is_root() or not self.verbose:
            return
        info = self.iteration_history[iteration]
        param_indices = self.dim.display_indices
        w = max(self.dim.covariate_label_width, 10)

        if iteration % 80 == 0:
            param_width = len(param_indices) * (w + 1) - 1
            header1 = (f"{'Iter':>4} | {'Reduced':^12} | {'#Viol':^5} | {'Pricing':^11} | "
                      f"{'Master':^10} | {'Objective':^13} | "
                      f"{'Constr':>6} | {f'Parameters':^{param_width}}")
            param_label_row = ' '.join(f'{self.dim.display_label(i):>{w}}' for i in param_indices)
            header2 = (f"{'':>4} | {'Cost':^12} | {'':^5} | {'(s)':^11} | "
                      f"{'(s)':^10} | {'Value':^13} | "
                      f"{'':>6} | {param_label_row}")
            logger.info(header1)
            logger.info(header2)
            logger.info("-" * len(header1))

        param_vals = ' '.join(format_number(self.theta_iter[i], width=w, precision=5) for i in param_indices)
        row = (f"{iteration:>4} | {format_number(info['reduced_cost'], width=12, precision=6)} | "
               f"{info['n_violations']:>5} | "
               f"{format_number(info['pricing_time'], width=10, precision=3)}s | "
               f"{format_number(info['master_time'], width=9, precision=3)}s | "
               f"{format_number(info['objective'], width=13, precision=5)} | "
               f"{info['n_constraints']:>6} | {param_vals}")
        logger.info(row)

    # ------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------

    def _create_result(self, num_iterations=None, total_time=None):
        # gather predicted_bundles from all ranks (runs on all ranks)
        predicted_bundles = self._gather_predicted_bundles()
        if not self.comm_manager.is_root():
            return None
        converged = num_iterations <= self.cfg.max_iters if num_iterations is not None else None
        pricing_times = [self.iteration_history[i]['pricing_time'] for i in sorted(self.iteration_history)]
        master_times = [self.iteration_history[i]['master_time'] for i in sorted(self.iteration_history)]
        final_info = self.iteration_history[max(self.iteration_history)]
        return RowGenerationEstimationResult(
            theta_hat=self.theta_iter,
            converged=converged,
            num_iterations=num_iterations,
            final_objective=self.master_model.ObjVal,
            n_constraints=self.master_model.NumConstrs,
            final_reduced_cost=final_info.get('reduced_cost', 0.0),
            total_time=total_time,
            final_n_violations=final_info.get('n_violations', 0),
            u_hat=self._result_u_hat(),
            predicted_bundles=predicted_bundles,
            timing=(pricing_times, master_times),
            warnings=[])

    def _gather_predicted_bundles(self):
        local = getattr(self.subproblem_manager, 'predicted_bundles', None)
        if local is None:
            return None
        return self.comm_manager.Gatherv_by_row(
            local, row_counts=self.comm_manager.agent_counts)

    def _check_bounds_hit(self, tol=None):
        empty = {'hit_lower': [], 'hit_upper': [], 'any_hit': False}
        if not self.comm_manager.is_root() or self.master_model is None:
            return empty
        theta = self.master_variables[0]
        if tol is None:
            tol = max(1e-8, self.master_model.Params.FeasibilityTol)
        hit_lower = [k for k in range(self.dim.n_covariates)
                    if theta[k].LB > -GRB.INFINITY and (theta[k].X - theta[k].LB) <= tol]
        hit_upper = [k for k in range(self.dim.n_covariates)
                    if theta[k].UB <  GRB.INFINITY and (theta[k].UB - theta[k].X) <= tol]
        return {'hit_lower': hit_lower, 'hit_upper': hit_upper, 'any_hit': bool(hit_lower or hit_upper)}

    # ------------------------------------------------------------------
    # Gurobi helpers
    # ------------------------------------------------------------------

    def _setup_gurobi_model(self, master_gurobi_params=None):
        params = {"Method": 0, "LPWarmStart": 2, "OutputFlag": 0, **(master_gurobi_params or {})}
        with suppress_output():
            model = gp.Model()
            for k, v in params.items():
                if v is not None:
                    model.setParam(k, v)
        return model
