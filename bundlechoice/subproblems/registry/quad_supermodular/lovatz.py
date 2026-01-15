import numpy as np
from typing import Any, Optional
from .quadratic_supermodular_base import QuadraticSupermodular

class QuadraticSOptLovasz(QuadraticSupermodular):

    def solve(self, theta, pb=None):
        linear, quadratic = self.build_quadratic_matrix(theta)
        P_i_j_j = quadratic.copy()
        diag_indices = np.arange(self.dimensions_cfg.num_items)
        P_i_j_j[:, diag_indices, diag_indices] += linear
        agent_data = self.data_manager.local_data.get('agent_data') or {}
        constraint_mask = agent_data.get('constraint_mask') if self.has_constraint_mask else None
        z_t = np.full((self.data_manager.num_local_agents, self.dimensions_cfg.num_items), 0.5, dtype=np.float64)
        if constraint_mask is not None:
            z_t[~constraint_mask] = 0.0
        z_best = z_t.copy()
        val_best = np.full(self.data_manager.num_local_agents, -np.inf, dtype=np.float64)
        num_iters = int(self.config.settings.get('num_iters_SGM', max(100000, 1000 * self.dimensions_cfg.num_items)))
        alpha_base = float(self.config.settings.get('alpha', 0.1 / np.sqrt(self.dimensions_cfg.num_items)))
        method = self.config.settings.get('method', 'constant_step_length')
        grad_i_j = np.zeros((self.data_manager.num_local_agents, self.dimensions_cfg.num_items), dtype=np.float64)
        tril_mask = np.tril(np.ones((self.dimensions_cfg.num_items, self.dimensions_cfg.num_items), dtype=bool), k=0)
        for iter in range(num_iters):
            grad_i_j, val_i = self._grad_lovasz_extension_batch(z_t, P_i_j_j, tril_mask)
            if constraint_mask is not None:
                grad_i_j[~constraint_mask] = 0.0
            grad_norm_i = np.linalg.norm(grad_i_j, axis=1, keepdims=True)
            grad_norm_i = np.maximum(grad_norm_i, 1e-10)
            if method == 'constant_step_length':
                step_i = alpha_base / grad_norm_i
            elif method == 'constant_step_size':
                step_i = np.full((self.data_manager.num_local_agents, 1), alpha_base)
            elif method == 'constant_over_sqrt_k':
                step_i = alpha_base / (grad_norm_i * np.sqrt(iter + 1))
            elif method == 'mirror_descent':
                z_t = z_t * np.exp(alpha_base * grad_i_j / grad_norm_i)
                if constraint_mask is not None:
                    z_t[~constraint_mask] = 0.0
                z_t = np.clip(z_t, 0.0, 1.0)
                improved = val_i > val_best
                z_best[improved] = z_t[improved]
                val_best[improved] = val_i[improved]
                continue
            else:
                step_i = np.zeros((self.data_manager.num_local_agents, 1))
            z_new = z_t + step_i * grad_i_j
            if constraint_mask is not None:
                z_new[~constraint_mask] = 0.0
            z_t = np.clip(z_new, 0.0, 1.0)
            improved = val_i > val_best
            z_best[improved] = z_t[improved]
            val_best[improved] = val_i[improved]
        grad_final, val_final = self._grad_lovasz_extension_batch(z_t, P_i_j_j, tril_mask)
        final_improved = val_final > val_best
        z_best[final_improved] = z_t[final_improved]
        val_best[final_improved] = val_final[final_improved]
        z_rounded = z_best.copy()
        near_boundary = np.abs(z_rounded - 0.5) < 1e-08
        z_rounded[near_boundary] = np.where(z_rounded[near_boundary] >= 0.5, 0.5 + 1e-07, 0.5 - 1e-07)
        optimal_bundles = z_rounded > 0.5
        if self.config.settings.get('verbose', False):
            violations_rounding = ((z_best > 0.1) & (z_best < 0.9)).sum(axis=1)
            aggregate_demand = optimal_bundles.sum(axis=1)
            if violations_rounding.max() > 0:
                print(f'violations mean: {violations_rounding.mean()}, max: {violations_rounding.max()}, demand mean: {aggregate_demand.mean()}, min: {aggregate_demand.min()}, max: {aggregate_demand.max()}')
        return optimal_bundles

    def _grad_lovasz_extension_batch(self, z_i_j, P_i_j_j, tril_mask):
        n_agents, n_items = z_i_j.shape
        sigma_i_j = np.argsort(z_i_j, axis=1)[:, ::-1]
        P_sigma = np.take_along_axis(np.take_along_axis(P_i_j_j, sigma_i_j[:, :, None], axis=1), sigma_i_j[:, None, :], axis=2)
        P_sigma_sym = P_sigma + P_sigma.transpose(0, 2, 1)
        diag_indices = np.arange(n_items)
        P_sigma_sym[:, diag_indices, diag_indices] = np.diagonal(P_sigma, axis1=1, axis2=2)
        grad_i_sigma = (P_sigma_sym * tril_mask[None, :, :]).sum(axis=2)
        grad_i_j = np.zeros_like(z_i_j)
        np.put_along_axis(grad_i_j, sigma_i_j, grad_i_sigma, axis=1)
        fun_value_i = (z_i_j * grad_i_j).sum(axis=1)
        return (grad_i_j, fun_value_i)

class QuadraticSOptLovaszSolver:

    def __init__(self, P_j_j, constraint_mask, subproblem_settings, errors=None):
        self.P_j_j = P_j_j
        self.constraint_mask = constraint_mask
        self.num_items = P_j_j.shape[0]
        self.subproblem_settings = subproblem_settings or {}
        self.errors = errors
        self._tril_mask = np.tril(np.ones((self.num_items, self.num_items), dtype=bool), k=0)

    def solve(self):
        z_t = np.full(self.num_items, 0.5, dtype=np.float64)
        if self.constraint_mask is not None:
            z_t[~self.constraint_mask] = 0.0
        z_best = z_t.copy()
        val_best = -np.inf
        num_iters = int(self.subproblem_settings.get('num_iters_SGM', max(100, 2 * self.num_items)))
        alpha_base = float(self.subproblem_settings.get('alpha', 1.0 / np.sqrt(self.num_items)))
        method = self.subproblem_settings.get('method', 'constant_over_sqrt_k')
        grad_j = np.zeros(self.num_items, dtype=np.float64)
        for iter in range(num_iters):
            grad_j, val = self._grad_lovatz_extension(z_t)
            if self.constraint_mask is not None:
                grad_j[~self.constraint_mask] = 0.0
            grad_norm = np.linalg.norm(grad_j)
            if grad_norm > 1e-10:
                if method == 'constant_step_length':
                    step = alpha_base / grad_norm
                elif method == 'constant_step_size':
                    step = alpha_base
                elif method == 'constant_over_sqrt_k':
                    step = alpha_base / (grad_norm * np.sqrt(iter + 1))
                elif method == 'mirror_descent':
                    z_t = z_t * np.exp(alpha_base * grad_j / grad_norm)
                    if self.constraint_mask is not None:
                        z_t[~self.constraint_mask] = 0.0
                    z_t = np.clip(z_t, 0.0, 1.0)
                    if val > val_best:
                        z_best = z_t.copy()
                        val_best = val
                    continue
                else:
                    step = 0.0
                z_new = z_t + step * grad_j
            else:
                z_new = z_t.copy()
            if self.constraint_mask is not None:
                z_new[~self.constraint_mask] = 0.0
            z_t = np.clip(z_new, 0.0, 1.0)
            if val > val_best:
                z_best = z_t.copy()
                val_best = val
        optimal_bundle = z_best > 0.5
        if self.subproblem_settings.get('verbose', False):
            violations_rounding = ((z_best > 0.1) & (z_best < 0.9)).sum()
            aggregate_demand = optimal_bundle.sum()
            print(f'violations: {violations_rounding}, demand: {aggregate_demand}, final_val: {val_best:.6f}')
        return optimal_bundle

    def _grad_lovatz_extension(self, z_i_j):
        if z_i_j.ndim == 2:
            n_agents, n_items = z_i_j.shape
            sigma_i_j = np.argsort(z_i_j, axis=1)[:, ::-1]
            P_sigma = np.take_along_axis(np.take_along_axis(self.P_j_j, sigma_i_j[:, :, None], axis=1), sigma_i_j[:, None, :], axis=2)
            P_sigma_sym = P_sigma + P_sigma.transpose(0, 2, 1)
            np.einsum('ijj->ij', P_sigma_sym)[:] = np.diagonal(P_sigma, axis1=1, axis2=2)
            mask = np.tril(np.ones((n_items, n_items), dtype=bool), k=0)
            grad_i_sigma = (P_sigma_sym * mask[None, :, :]).sum(axis=2)
            grad_i_j = np.zeros_like(z_i_j)
            np.put_along_axis(grad_i_j, sigma_i_j, grad_i_sigma, axis=1)
            fun_value_i = (z_i_j * grad_i_j).sum(axis=1)
            return (grad_i_j, fun_value_i)
        else:
            n_items = z_i_j.shape[0]
            sigma_j = np.argsort(z_i_j)[::-1]
            P_sigma = self.P_j_j[np.ix_(sigma_j, sigma_j)]
            P_sigma_sym = P_sigma + P_sigma.T
            np.fill_diagonal(P_sigma_sym, np.diag(P_sigma))
            grad_sigma = (P_sigma_sym * self._tril_mask).sum(axis=1)
            grad_j = np.zeros_like(z_i_j)
            grad_j[sigma_j] = grad_sigma
            fun_value = np.dot(z_i_j, grad_j)
            return (grad_j, fun_value)

def grad_lovasz_extension_batch(z_i_j, P_i_j_j):
    n_agents, n_items = z_i_j.shape
    sigma_i_j = np.argsort(z_i_j, axis=1)[:, ::-1]
    P_sigma = np.take_along_axis(np.take_along_axis(P_i_j_j, sigma_i_j[:, :, None], axis=1), sigma_i_j[:, None, :], axis=2)
    mask = np.tril(np.ones((n_items, n_items), dtype=bool))
    grad_i_sigma = (P_sigma * mask).sum(axis=2)
    grad_i_j = np.zeros_like(z_i_j)
    np.put_along_axis(grad_i_j, sigma_i_j, grad_i_sigma, axis=1)
    fun_value_i = (z_i_j * grad_i_j).sum(axis=1)
    return (grad_i_j, fun_value_i)