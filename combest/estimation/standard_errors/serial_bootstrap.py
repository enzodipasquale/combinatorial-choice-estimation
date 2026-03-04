import numpy as np
import gurobipy as gp
import time
from combest.utils import get_logger, format_number

logger = get_logger(__name__)


class SerialBootstrapMixin:

    def compute_bootstrap(self, num_bootstrap=100,
                          seed=None, verbose=False,
                          pt_estimate_callbacks=(None, None),
                          bootstrap_callback=None,
                          method='bayesian'):
        from combest.estimation.point_estimation.n_slack import NSlackSolver
        if not isinstance(self.row_generation_manager, NSlackSolver):
            raise TypeError("Bootstrap requires the n_slack formulation.")
        initialization_callback, iteration_callback = pt_estimate_callbacks
        theta_boots = []
        self.bootstrap_history = {}
        self.verbose = verbose
        self.row_gen = self.row_generation_manager
        t0 = time.perf_counter()

        # === 1. Point estimation (weights default to obs_quantity) ===
        self.point_result = self.row_gen.solve(
            initialize_master=True,
            initialize_solver=True,
            iteration_callback=iteration_callback,
            initialization_callback=initialization_callback,
            verbose=verbose
        )

        # === 2. Snapshot the converged model (root only) ===
        base_model, base_vars = self.row_gen.copy_master_model()

        # === 3. Generate and scatter bootstrap weights ===
        gen = self.generate_weights_bayesian_bootstrap if method == 'bayesian' \
              else self.generate_weights_standard_bootstrap
        weights = gen(seed, num_bootstrap)

        local_weights = self.comm_manager.Scatterv_by_row(
            weights, row_counts=self.comm_manager.agent_counts,
            dtype=np.float64, shape=(self.dim.n_agents, num_bootstrap))

        if self.verbose and self.comm_manager.is_root():
            self.pt_estimation_manager._log_instance_summary()
            logger.info(" ")
            logger.info(" BAYESIAN BOOTSTRAP")

        # === 4. Bootstrap loop ===
        for b in range(num_bootstrap):
            t_boot = time.perf_counter()

            # Copy base model -> fresh start each boot
            if self.comm_manager.is_root():
                model_b = base_model.copy()
                all_vars = model_b.getVars()
                theta_b = gp.MVar.fromlist(all_vars[:self.dim.n_covariates])
                u_b = gp.MVar.fromlist(all_vars[self.dim.n_covariates:
                                                self.dim.n_covariates + self.dim.n_agents])
                self.row_gen.install_master_model(model_b, (theta_b, u_b))

            if bootstrap_callback is not None:
                bootstrap_callback(b, self)

            self.result = self.row_gen.solve(
                resampling_weights=local_weights[:, b],
                verbose=False,
                initialize_solver=False,
                initialize_master=False,
                iteration_callback=iteration_callback,
                initialization_callback=initialization_callback)

            self.boot_time = time.perf_counter() - t_boot

            if self.comm_manager.is_root():
                theta_boots.append(self.row_gen.master_variables[0].X.copy())
                self._update_bootstrap_info(b)
            self._log_bootstrap_iteration(b)

        total_time = time.perf_counter() - t0
        if not self.comm_manager.is_root():
            return None

        stats_result = self.compute_bootstrap_stats(theta_boots)
        self._log_bootstrap_summary(num_bootstrap, total_time, stats_result)
        return stats_result

    # ------------------------------------------------------------------
    # Serial-specific logging
    # ------------------------------------------------------------------

    def _update_bootstrap_info(self, bootstrap_iter):
        if not self.comm_manager.is_root():
            return
        if bootstrap_iter not in self.bootstrap_history:
            self.bootstrap_history[bootstrap_iter] = {}

        pricing_times, master_times = self.result.timing if isinstance(self.result.timing, tuple) else ([], [])
        total_pricing_time = sum(pricing_times) if pricing_times else 0.0
        total_master_time = sum(master_times) if master_times else 0.0

        self.bootstrap_history[bootstrap_iter].update({
            'time': self.boot_time,
            'iterations': self.result.num_iterations,
            'objective': self.result.final_objective,
            'converged': self.result.converged,
            'n_constraints': self.result.n_constraints,
            'pricing_time': total_pricing_time,
            'master_time': total_master_time,
            'reduced_cost': self.result.final_reduced_cost,
            'n_violations': self.result.final_n_violations
        })

    def _log_bootstrap_iteration(self, bootstrap_iter):
        if not self.comm_manager.is_root() or not self.verbose:
            return
        info = self.bootstrap_history[bootstrap_iter]
        param_indices = self.dim.get_display_indices(self.config.standard_errors.parameters_to_log)
        w = max(self.dim.covariate_label_width, 10)

        if bootstrap_iter % 100 == 0:
            param_width = len(param_indices) * (w + 1) - 1
            header1 = (f"{'Boot':>5} | {'Time':^9} | {'Pricing':^9} | {'Master':^9} | {'RG':^5} | "
                      f"{'#Constr':>7} | {'Reduced':^12} | {'#Viol':^5} | {'Objective':^12} | {f'Parameters':^{param_width}}")
            param_label_row = ' '.join(f'{self.dim.covariate_labels[i]:>{w}}' for i in param_indices)
            header2 = (f"{'':>5} | {'(s)':^9} | {'(s)':^9} | {'(s)':^9} | {'Iters':^5} | "
                      f"{'':>7} | {'Cost':^12} | {'':^5} | {'Value':^12} | {param_label_row}")
            logger.info("-" * len(header1))
            logger.info(header1)
            logger.info(header2)
            logger.info("-" * len(header1))
        theta = self.result.theta_hat
        param_vals = ' '.join(format_number(theta[i], width=w, precision=5) for i in param_indices)
        time_str = f"{info['time']:>9.3f}"
        pricing_time_str = f"{info.get('pricing_time', 0.0):>9.3f}"
        master_time_str = f"{info.get('master_time', 0.0):>9.3f}"
        reduced_cost_str = format_number(info.get('reduced_cost', 0.0), width=12, precision=6)
        row = (f"{bootstrap_iter:>5} | {time_str} | {pricing_time_str} | {master_time_str} | "
               f"{info['iterations']:>5} | "
               f"{info.get('n_constraints', 0):>7} | "
               f"{reduced_cost_str} | "
               f"{info.get('n_violations', 0):>5} | "
               f"{format_number(info['objective'], width=12, precision=5)} | {param_vals}")
        logger.info(row)

    def _log_bootstrap_summary(self, n_bootstrap, total_time, result):
        if not self.comm_manager.is_root() or not self.verbose:
            return

        param_indices = self.dim.get_display_indices(self.config.standard_errors.parameters_to_log)
        w = max(self.dim.covariate_label_width, 8)

        sep_width = w + 4 + 12 + 3 + 12 + 3 + 10
        logger.info(" ")
        logger.info("-" * sep_width)
        logger.info(f" SERIAL BOOTSTRAP SUMMARY: {n_bootstrap} samples in {total_time:.1f}s")
        logger.info("-" * sep_width)
        logger.info(f"{'Param':>{w}} | {'Mean':>12} | {'SE':>12} | {'t-stat':>10}")
        logger.info("-" * sep_width)
        for i in param_indices:
            logger.info(f"{self.dim.covariate_labels[i]:>{w}} | {result.mean[i]:>12.5f} | {result.se[i]:>12.5f} | {result.t_stats[i]:>10.2f}")
        logger.info("-" * sep_width)
