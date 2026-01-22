import numpy as np
from .result import BayesianBootstrapResult
from bundlechoice.utils import get_logger, format_number
logger = get_logger(__name__)
import time

class ResamplingMixin:
    def compute_bayesian_bootstrap(self, num_bootstrap=100, 
                                            seed=None, 
                                            verbose=False, 
                                            row_gen_iteration_callback = None,
                                            row_gen_initialization_callback = None, 
                                            bootstrap_callback = None
                                            ):
        theta_boots = []
        self.bootstrap_history = {}
        self.verbose = verbose
        rng = np.random.default_rng(seed)
        row_gen = self.row_generation_manager
        t0 = time.perf_counter()

        if self.verbose:
            row_gen._log_instance_summary()
            logger.info(" " )
            logger.info(" BAYESIAN BOOTSTRAP")
        
        for b in range(num_bootstrap):
            t_boot = time.perf_counter()
            if self.comm_manager._is_root():
                weights = rng.exponential(1.0, self.dim.n_obs)
                weights /= weights.mean()
                weights = np.tile(weights, self.dim.n_simulations)
            else:
                weights = None
            local_weights = self.comm_manager.Scatterv_by_row(weights, row_counts=self.data_manager.agent_counts)
            initialize_master = True if b == 0 else False
            initialize_subproblems = True if (b == 0 and not self.subproblem_manager._subproblems_are_initialized) else False
            self.result = row_gen.solve(local_obs_weights= local_weights, 
                                        verbose= False, 
                                        initialize_subproblems= initialize_subproblems, 
                                        initialize_master= initialize_master,
                                        iteration_callback= row_gen_iteration_callback,
                                        initialization_callback= row_gen_initialization_callback)
            self.boot_time = time.perf_counter() - t_boot
            
            if self.comm_manager._is_root():
                theta_boots.append(row_gen.master_variables[0].X.copy())
                self._update_bootstrap_info(b)
            self._log_bootstrap_iteration(b, row_gen.theta_iter)
            if bootstrap_callback is not None:
                bootstrap_callback(self, row_gen)
        
        total_time = time.perf_counter() - t0
        if not self.comm_manager._is_root():
            return None
        
        stats_result = self.compute_bootstrap_stats(theta_boots)
        self._log_bootstrap_summary(num_bootstrap, total_time, stats_result)
        
        return stats_result

    def _update_bootstrap_info(self, bootstrap_iter):
        if not self.comm_manager._is_root():
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
            'reduced_cost': self.result.final_reduced_cost
        })

    def _log_bootstrap_iteration(self, bootstrap_iter, theta):
        if not self.comm_manager._is_root() or not self.verbose:
            return
        if bootstrap_iter not in self.bootstrap_history:
            return
        if bootstrap_iter == 0:
            logger.info("-"*55)
        info = self.bootstrap_history[bootstrap_iter]    
        if self.config.row_generation.parameters_to_log is not None:
            param_indices = self.config.row_generation.parameters_to_log
        else:
            param_indices = list(range(min(5, self.dim.n_features)))
        
        if bootstrap_iter % 80 == 0:
            param_width = len(param_indices) * 11 - 1
            header1 = (f"{'Boot':>5} | {'Time':^9} | {'Pricing':^9} | {'Master':^9} | {'RG':^5} | "
                      f"{'#Constr':>7} | {'Reduced':^12} | {'Objective':^12} | {f'Parameters':^{param_width}}")
            param_label_row = ' '.join(f'{f"θ[{i}]":>10}' for i in param_indices)
            header2 = (f"{'':>5} | {'(s)':^9} | {'(s)':^9} | {'(s)':^9} | {'Iters':^5} | "
                      f"{'':>7} | {'Cost':^12} | {'Value':^12} | {param_label_row}")
            logger.info(header1)
            logger.info(header2)
            logger.info("-" * len(header1))
        
        param_vals = ' '.join(format_number(theta[i], width=10, precision=5) for i in param_indices)
        time_str = f"{info['time']:>9.3f}"
        pricing_time_str = f"{info.get('pricing_time', 0.0):>9.3f}"
        master_time_str = f"{info.get('master_time', 0.0):>9.3f}"
        reduced_cost_str = format_number(info.get('reduced_cost', 0.0), width=12, precision=6)
        row = (f"{bootstrap_iter:>5} | {time_str} | {pricing_time_str} | {master_time_str} | "
               f"{info['iterations']:>5} | "
               f"{info.get('n_constraints', 0):>7} | "
               f"{reduced_cost_str} | "
               f"{format_number(info['objective'], width=12, precision=5)} | {param_vals}")
        logger.info(row)

    def _log_bootstrap_summary(self, n_bootstrap, total_time, result):
        if not self.comm_manager._is_root() or not self.verbose:
            return
        
        if self.config.row_generation.parameters_to_log is not None:
            param_indices = self.config.row_generation.parameters_to_log
        else:
            param_indices = list(range(min(5, self.dim.n_features)))
        
        logger.info(" " )
        logger.info("-"*55)
        logger.info(f" BAYESIAN BOOTSTRAP SUMMARY: {n_bootstrap} samples in {total_time:.1f}s")
        logger.info("-"*55)
        logger.info(f"{'Param':>6} | {'Mean':>12} | {'SE':>12} | {'t-stat':>10}")
        logger.info("-"*55)
        for i in param_indices:
            logger.info(f"θ[{i:>3}] | {result.mean[i]:>12.5f} | {result.se[i]:>12.5f} | {result.t_stats[i]:>10.2f}")
        logger.info("-"*55)

 

    def compute_bootstrap_stats(self, theta_boots, theta_hat=None, confidence=0.95):
        if not self.comm_manager._is_root():
            return None
        
        theta_boots = np.asarray(theta_boots)
        n_samples, _ = theta_boots.shape
        
        mean = theta_boots.mean(axis=0)
        se = theta_boots.std(axis=0, ddof=1)
        point_est = theta_hat if theta_hat is not None else mean
        t_stats = np.where(se > 1e-16, point_est / se, np.nan)
        alpha = 1 - confidence
        ci_lower = np.percentile(theta_boots, 100 * alpha / 2, axis=0)
        ci_upper = np.percentile(theta_boots, 100 * (1 - alpha / 2), axis=0)
        return BayesianBootstrapResult(
            mean=mean,
            se=se,
            t_stats=t_stats,
            n_samples=n_samples,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence=confidence,
        )

