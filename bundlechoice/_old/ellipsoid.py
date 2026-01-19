import time
import math
import numpy as np
from .base import BaseEstimationManager
from bundlechoice.utils import get_logger
logger = get_logger(__name__)

class EllipsoidManager(BaseEstimationManager):

    def __init__(self, comm_manager, config, data_manager, oracles_manager, subproblem_manager):
        super().__init__(comm_manager, config, data_manager, oracles_manager, subproblem_manager)
        self.theta_iter = None
        self.B_iter = None
        self.iteration_count = 0
        self.timing_stats = None
        self.cfg = self.config.ellipsoid
        self.dim = self.config.dimensions
        n = self.dim.n_features
        self.n = n
        self.alpha = (n ** 2 / (n ** 2 - 1)) ** (1 / 4)
        self.gamma = self.alpha * ((n - 1) / (n + 1)) ** (1 / 2)
        self.gamma_1 = (n ** 2 / (n ** 2 - 1)) ** (1 / 2)
        self.gamma_2 = self.gamma_1 * (2 / (n + 1))

    def solve(self, callback=None, theta_init=None):
        logger.info('=== ELLIPSOID METHOD ===')
        t0 = time.perf_counter()
        self.subproblem_manager.initialize_subproblems()
        self._initialize_ellipsoid(theta_init)
        
        if self.cfg.num_iters is not None:
            num_iters = min(self.cfg.num_iters, self.cfg.max_iterations)
        else:
            computed = int(self.n * (self.n - 1) * math.log(1.0 / self.cfg.solver_precision))
            num_iters = min(computed, self.cfg.max_iterations)
            
        keep_last_n = min(1000, num_iters)
        vals, centers = [], []
        
        for iteration in range(1, num_iters + 1):
            logger.info(f'ELLIPSOID ITERATION {iteration}')
            logger.info(f'THETA: {np.round(self.theta_iter, 4)}')
            
            violated = np.where(self.theta_iter < 0.0)[0]
            if violated.size > 0:
                direction = np.zeros_like(self.theta_iter)
                direction[violated[0]] = -1.0
                obj_value = np.inf
            else:
                obj_value, gradient = self.compute_obj_and_grad_at_root(self.theta_iter)
                direction = gradient if self.comm_manager._is_root() else None
                if self.comm_manager._is_root():
                    vals.append(obj_value)
                    centers.append(self.theta_iter.copy())
                    if len(centers) > keep_last_n:
                        vals = vals[-keep_last_n:]
                        centers = centers[-keep_last_n:]
                        
            self._update_ellipsoid(direction)
            if not self.comm_manager._is_root():
                self.theta_iter = np.empty(self.dim.n_features, dtype=np.float64)
            self.comm_manager.Bcast(self.theta_iter)
            
            if callback and self.comm_manager._is_root() and obj_value != np.inf:
                callback({'iteration': iteration, 'theta': self.theta_iter.copy(), 'objective': obj_value,
                          'best_objective': min(vals) if vals else None})
                          
        elapsed = time.perf_counter() - t0
        logger.info('Ellipsoid method completed %d iterations in %.2f seconds.', num_iters, elapsed)
        
        if self.comm_manager._is_root():
            if len(vals) > 0:
                best_idx = np.argmin(vals)
                best_theta = np.array(centers)[best_idx]
                best_obj = vals[best_idx]
                logger.info(f'Best objective: {best_obj:.4f} at iteration {best_idx + 1}')
            else:
                best_theta = self.theta_iter
                best_obj = None
        else:
            best_theta = np.empty(self.dim.n_features, dtype=np.float64)
            best_obj = None
            
        self.comm_manager.Bcast(best_theta)
        return self._create_result(num_iters, None, best_theta, self.cfg)

    def _initialize_ellipsoid(self, theta_init=None):
        if theta_init is not None:
            self.theta_iter = theta_init.copy()
        else:
            self.theta_iter = 0.1 * np.ones(self.n)
        self.B_iter = self.cfg.initial_radius * np.eye(self.n)

    def _update_ellipsoid(self, d):
        if not self.comm_manager._is_root():
            return
        dTBd = d.T @ self.B_iter @ d
        if dTBd <= 0 or not np.isfinite(dTBd):
            logger.warning('Ellipsoid update: dTBd <= 0 or non-finite, skipping')
            return
        b = self.B_iter @ d / np.sqrt(dTBd)
        if not np.all(np.isfinite(b)):
            logger.warning('Ellipsoid update: b is non-finite, skipping')
            return
        self.theta_iter = self.theta_iter - 1 / (self.n + 1) * b
        B_new = self.gamma_1 * self.B_iter - self.gamma_2 * np.outer(b, b)
        if np.all(np.isfinite(B_new)):
            self.B_iter = B_new
        else:
            logger.warning('Ellipsoid update: B_new is non-finite, skipping B update')
