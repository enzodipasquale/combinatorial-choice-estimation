"""
Quadratic Supermodular Lovász Subproblem Solver
----------------------------------------------
Implements quadratic supermodular minimization for combinatorial choice estimation using the Lovász extension (SGM/gradient method).
Handles modular and quadratic agent/item features, supports missing data, and is designed for MPI batch solving.
"""
import numpy as np
from typing import Any, Optional
from .quadratic_supermodular_base import QuadraticSupermodular

class QuadraticSOptLovasz(QuadraticSupermodular):
    """
    Subproblem for quadratic supermodular minimization via Lovász extension (SGM/gradient method).
    Handles modular/quadratic agent/item features, missing data, and batch MPI solving.
    """
    def solve(self, theta: np.ndarray, pb: Optional[Any] = None) -> np.ndarray:
        P_i_j_j = self.build_quadratic_matrix(theta)
        optimal_bundles = np.zeros((self.num_local_agents, self.num_items), dtype=bool)
        for i in range(self.num_local_agents):
            solver = QuadraticSOptLovaszSolver(P_i_j_j[i], self.constraint_mask[i], self.config.settings, self.errors[i])
            optimal_bundle = solver.solve()
            optimal_bundles[i] = optimal_bundle
        return optimal_bundles

class QuadraticSOptLovaszSolver:
    def __init__(self, P_j_j: np.ndarray, constraint_mask: Optional[np.ndarray], subproblem_settings: dict, errors: Optional[np.ndarray] = None):
        self.P_j_j = P_j_j
        self.constraint_mask = constraint_mask
        self.num_items = P_j_j.shape[0]
        self.subproblem_settings = subproblem_settings or {}
        self.errors = errors

    def solve(self) -> np.ndarray:
        # Support both single-agent and batch (multi-agent) optimization
        z_t = np.full((self.num_items,), 0.5) if self.P_j_j.ndim == 2 else np.full(self.P_j_j.shape[:2], 0.5)
        if self.constraint_mask is not None:
            if z_t.ndim == 1:
                z_t[~self.constraint_mask] = 0
            else:
                z_t[~self.constraint_mask] = 0
        z_best = np.zeros_like(z_t)
        val_best = -np.inf if z_t.ndim == 1 else np.full(z_t.shape[0], -np.inf)
        num_iters = int(self.subproblem_settings.get("num_iters_SGM", 100))
        alpha = float(self.subproblem_settings.get("alpha", np.sqrt(self.num_items)))
        method = self.subproblem_settings.get("method", "constant_over_sqrt_k")
        for iter in range(num_iters):
            grad_j, val = self._grad_lovatz_extension(z_t)
            if self.constraint_mask is not None:
                grad_j[~self.constraint_mask] = 0
            if method == 'constant_step_length':
                grad_norm = np.linalg.norm(grad_j, axis=-1, keepdims=True)
                z_new = z_t + alpha * grad_j / np.where(grad_norm > 0, grad_norm, 1)
            elif method == 'constant_step_size':
                z_new = z_t + alpha * grad_j
            elif method == 'constant_over_sqrt_k':
                grad_norm = np.linalg.norm(grad_j, axis=-1, keepdims=True)
                z_new = z_t + alpha * grad_j / np.where(grad_norm > 0, grad_norm, 1) / np.sqrt(iter + 1)
            elif method == 'mirror_descent':
                grad_norm = np.linalg.norm(grad_j, axis=-1, keepdims=True)
                z_new = z_t * np.exp(alpha * grad_j / np.where(grad_norm > 0, grad_norm, 1))
            else:
                z_new = z_t
            if self.constraint_mask is not None:
                z_new[~self.constraint_mask] = 0
            z_t = np.clip(z_new, 0, 1)
            improved = val > val_best
            if z_t.ndim == 1:
                if improved:
                    z_best = z_t.copy()
                    val_best = val
            else:
                z_best[improved] = z_t[improved]
                val_best[improved] = val[improved]
        optimal_bundle = (z_best > 0)
        if self.subproblem_settings.get("verbose", False):
            if z_t.ndim == 1:
                violations_rounding = ((z_best > .1) & (z_best < .9)).sum()
                aggregate_demand = optimal_bundle.sum()
                print(f"violations: {violations_rounding}, demand: {aggregate_demand}")
            else:
                violations_rounding = ((z_best > .1) & (z_best < .9)).sum(axis=1)
                aggregate_demand = optimal_bundle.sum(axis=1)
                if violations_rounding.max() > 0:
                    print(f"violations mean: {violations_rounding.mean()}, max: {violations_rounding.max()}, "
                        f"demand mean: {aggregate_demand.mean()}, min: {aggregate_demand.min()}, max: {aggregate_demand.max()}")
        return optimal_bundle

    def _grad_lovatz_extension(self, z_i_j):
        # Batched version if z_i_j is 2D, else single-agent version
        if z_i_j.ndim == 2:
            n_agents, n_items = z_i_j.shape
            sigma_i_j = np.argsort(z_i_j, axis=1)[:, ::-1]
            P_sigma = np.take_along_axis(
                np.take_along_axis(self.P_j_j, sigma_i_j[:, :, None], axis=1),
                sigma_i_j[:, None, :], axis=2
            )
            mask = np.tril(np.ones((n_items, n_items), dtype=bool))
            grad_i_sigma = (P_sigma * mask).sum(axis=2)
            grad_i_j = np.zeros_like(z_i_j)
            np.put_along_axis(grad_i_j, sigma_i_j, grad_i_sigma, axis=1)
            fun_value_i = (z_i_j * grad_i_j).sum(axis=1)
            return grad_i_j, fun_value_i
        else:
            n_items = z_i_j.shape[0]
            sigma_j = np.argsort(z_i_j)[::-1]
            P_sigma = self.P_j_j[np.ix_(sigma_j, sigma_j)]
            grad_sigma = np.tril(P_sigma).sum(axis=1)
            grad_j = np.zeros_like(z_i_j)
            grad_j[sigma_j] = grad_sigma
            fun_value = np.dot(z_i_j, grad_j)
            return grad_j, fun_value


def grad_lovasz_extension_batch(z_i_j, P_i_j_j):
    """
    Vectorized Lovász extension gradient for a batch of agents.
    Args:
        z_i_j: (n_agents, n_items) array of continuous variables
        P_i_j_j: (n_agents, n_items, n_items) quadratic matrices
    Returns:
        grad_i_j: (n_agents, n_items) gradient for each agent
        fun_value_i: (n_agents,) Lovász extension value for each agent
    """
    n_agents, n_items = z_i_j.shape

    # Sort z_i_j for each agent in descending order
    sigma_i_j = np.argsort(z_i_j, axis=1)[:, ::-1]  # (n_agents, n_items)

    # For each agent, permute P_i_j_j according to sigma_i_j
    P_sigma = np.take_along_axis(
        np.take_along_axis(P_i_j_j, sigma_i_j[:, :, None], axis=1),
        sigma_i_j[:, None, :], axis=2
    )  # (n_agents, n_items, n_items)

    # Lower-triangular mask for summing
    mask = np.tril(np.ones((n_items, n_items), dtype=bool))

    # Compute grad_i_sigma: sum over lower triangle for each agent
    grad_i_sigma = (P_sigma * mask).sum(axis=2)  # (n_agents, n_items)

    # Scatter grad_i_sigma back to original order
    grad_i_j = np.zeros_like(z_i_j)
    np.put_along_axis(grad_i_j, sigma_i_j, grad_i_sigma, axis=1)

    # Lovász extension value
    fun_value_i = (z_i_j * grad_i_j).sum(axis=1)

    return grad_i_j, fun_value_i 