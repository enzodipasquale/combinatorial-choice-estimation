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
        """
        Solve for all agents using vectorized Lovász extension.
        Computes gradients and takes steps across all agents simultaneously.
        """
        linear, quadratic = self.build_quadratic_matrix(theta)
        # Combine linear (diagonal) and quadratic (off-diagonal) terms into single matrix
        # P_i_j_j[i] has linear[i] on diagonal and quadratic[i] on off-diagonal
        P_i_j_j = quadratic.copy()  # Start with quadratic terms (off-diagonal)
        # Add linear terms to diagonal using advanced indexing
        diag_indices = np.arange(self.num_items)
        P_i_j_j[:, diag_indices, diag_indices] += linear
        
        agent_data = self.local_data.get("agent_data") or {}
        constraint_mask = agent_data.get("constraint_mask") if self.has_constraint_mask else None
        
        # Initialize for all agents at once: (num_local_agents, num_items)
        z_t = np.full((self.num_local_agents, self.num_items), 0.5, dtype=np.float64)
        if constraint_mask is not None:
            z_t[~constraint_mask] = 0.0
        
        z_best = z_t.copy()
        val_best = np.full(self.num_local_agents, -np.inf, dtype=np.float64)
        
        # Adaptive parameters - use many iterations to ensure exact convergence
        # Default: use a very large number of iterations to guarantee convergence to optimal solution
        num_iters = int(self.config.settings.get("num_iters_SGM", max(100000, 1000 * self.num_items)))
        alpha_base = float(self.config.settings.get("alpha", 0.1 / np.sqrt(self.num_items)))  # Smaller step size for stability
        # Use constant_step_length for better convergence (doesn't decay)
        method = self.config.settings.get("method", "constant_step_length")
        
        # Pre-allocate gradient array
        grad_i_j = np.zeros((self.num_local_agents, self.num_items), dtype=np.float64)
        
        # Pre-compute triangular mask
        tril_mask = np.tril(np.ones((self.num_items, self.num_items), dtype=bool), k=0)
        
        for iter in range(num_iters):
            # Vectorized gradient computation for all agents
            grad_i_j, val_i = self._grad_lovasz_extension_batch(z_t, P_i_j_j, tril_mask)
            
            # Zero out gradients for constrained items
            if constraint_mask is not None:
                grad_i_j[~constraint_mask] = 0.0
            
            # Compute gradient norms for all agents
            grad_norm_i = np.linalg.norm(grad_i_j, axis=1, keepdims=True)  # (num_local_agents, 1)
            grad_norm_i = np.maximum(grad_norm_i, 1e-10)  # Avoid division by zero
            
            # Compute steps for all agents
            if method == 'constant_step_length':
                step_i = alpha_base / grad_norm_i
            elif method == 'constant_step_size':
                step_i = np.full((self.num_local_agents, 1), alpha_base)
            elif method == 'constant_over_sqrt_k':
                step_i = alpha_base / (grad_norm_i * np.sqrt(iter + 1))
            elif method == 'mirror_descent':
                # Mirror descent: z_new = z * exp(alpha * grad / ||grad||)
                z_t = z_t * np.exp(alpha_base * grad_i_j / grad_norm_i)
                if constraint_mask is not None:
                    z_t[~constraint_mask] = 0.0
                z_t = np.clip(z_t, 0.0, 1.0)
                # Track best solutions
                improved = val_i > val_best
                z_best[improved] = z_t[improved]
                val_best[improved] = val_i[improved]
                continue
            else:
                step_i = np.zeros((self.num_local_agents, 1))
            
            # Gradient ascent step for all agents (MAXIMIZE objective)
            z_new = z_t + step_i * grad_i_j
            
            # Apply constraints
            if constraint_mask is not None:
                z_new[~constraint_mask] = 0.0
            
            # Project to [0, 1]
            z_t = np.clip(z_new, 0.0, 1.0)
            
            # Track best solutions
            improved = val_i > val_best
            z_best[improved] = z_t[improved]
            val_best[improved] = val_i[improved]
        
        # Final check: compare z_best and final z_t, use whichever is better
        grad_final, val_final = self._grad_lovasz_extension_batch(z_t, P_i_j_j, tril_mask)
        final_improved = val_final > val_best
        z_best[final_improved] = z_t[final_improved]
        val_best[final_improved] = val_final[final_improved]
        
        # Round to binary solutions - use strict threshold to avoid ties
        # Push values very close to 0.5 away from the boundary
        z_rounded = z_best.copy()
        near_boundary = np.abs(z_rounded - 0.5) < 1e-8
        z_rounded[near_boundary] = np.where(z_rounded[near_boundary] >= 0.5, 0.5 + 1e-7, 0.5 - 1e-7)
        optimal_bundles = (z_rounded > 0.5)
        
        if self.config.settings.get("verbose", False):
            violations_rounding = ((z_best > 0.1) & (z_best < 0.9)).sum(axis=1)
            aggregate_demand = optimal_bundles.sum(axis=1)
            if violations_rounding.max() > 0:
                print(f"violations mean: {violations_rounding.mean()}, max: {violations_rounding.max()}, "
                      f"demand mean: {aggregate_demand.mean()}, min: {aggregate_demand.min()}, max: {aggregate_demand.max()}")
        
        return optimal_bundles
    
    def _grad_lovasz_extension_batch(self, z_i_j: np.ndarray, P_i_j_j: np.ndarray, tril_mask: np.ndarray):
        """
        Vectorized gradient computation for batch of agents.
        Args:
            z_i_j: (num_local_agents, num_items) continuous variables
            P_i_j_j: (num_local_agents, num_items, num_items) quadratic matrices
            tril_mask: (num_items, num_items) lower triangular mask
        Returns:
            grad_i_j: (num_local_agents, num_items) gradients
            fun_value_i: (num_local_agents,) function values
        """
        n_agents, n_items = z_i_j.shape
        
        # Sort z_i_j for each agent in descending order
        sigma_i_j = np.argsort(z_i_j, axis=1)[:, ::-1]  # (n_agents, n_items)
        
        # Permute P_i_j_j according to sorted order for each agent
        P_sigma = np.take_along_axis(
            np.take_along_axis(P_i_j_j, sigma_i_j[:, :, None], axis=1),
            sigma_i_j[:, None, :], axis=2
        )  # (n_agents, n_items, n_items)
        
        # Make symmetric: P + P^T (diagonal counted once)
        P_sigma_sym = P_sigma + P_sigma.transpose(0, 2, 1)
        # Fix diagonal (counted twice in transpose, so subtract once)
        diag_indices = np.arange(n_items)
        P_sigma_sym[:, diag_indices, diag_indices] = np.diagonal(P_sigma, axis1=1, axis2=2)
        
        # Gradient: sum over lower triangle (including diagonal)
        grad_i_sigma = (P_sigma_sym * tril_mask[None, :, :]).sum(axis=2)  # (n_agents, n_items)
        
        # Scatter grad_i_sigma back to original order
        grad_i_j = np.zeros_like(z_i_j)
        np.put_along_axis(grad_i_j, sigma_i_j, grad_i_sigma, axis=1)
        
        # Function value: z^T * grad (Lovász extension value)
        fun_value_i = (z_i_j * grad_i_j).sum(axis=1)  # (n_agents,)
        
        return grad_i_j, fun_value_i

class QuadraticSOptLovaszSolver:
    def __init__(self, P_j_j: np.ndarray, constraint_mask: Optional[np.ndarray], subproblem_settings: dict, errors: Optional[np.ndarray] = None):
        self.P_j_j = P_j_j
        self.constraint_mask = constraint_mask
        self.num_items = P_j_j.shape[0]
        self.subproblem_settings = subproblem_settings or {}
        self.errors = errors
        
        # Pre-compute triangular mask for efficiency
        self._tril_mask = np.tril(np.ones((self.num_items, self.num_items), dtype=bool), k=0)

    def solve(self) -> np.ndarray:
        """
        Solve using subgradient method on Lovász extension.
        Optimized for single-agent case.
        """
        # Initialize: start at 0.5 for all items
        z_t = np.full(self.num_items, 0.5, dtype=np.float64)
        if self.constraint_mask is not None:
            z_t[~self.constraint_mask] = 0.0
        
        z_best = z_t.copy()
        val_best = -np.inf
        
        # Adaptive parameters based on problem size
        num_iters = int(self.subproblem_settings.get("num_iters_SGM", max(100, 2 * self.num_items)))
        alpha_base = float(self.subproblem_settings.get("alpha", 1.0 / np.sqrt(self.num_items)))
        method = self.subproblem_settings.get("method", "constant_over_sqrt_k")
        
        # Pre-allocate arrays for efficiency
        grad_j = np.zeros(self.num_items, dtype=np.float64)
        
        for iter in range(num_iters):
            grad_j, val = self._grad_lovatz_extension(z_t)
            
            # Zero out gradient for constrained items
            if self.constraint_mask is not None:
                grad_j[~self.constraint_mask] = 0.0
            
            # Update step based on method (optimized)
            grad_norm = np.linalg.norm(grad_j)
            if grad_norm > 1e-10:  # Avoid division by zero
                if method == 'constant_step_length':
                    step = alpha_base / grad_norm
                elif method == 'constant_step_size':
                    step = alpha_base
                elif method == 'constant_over_sqrt_k':
                    step = alpha_base / (grad_norm * np.sqrt(iter + 1))
                elif method == 'mirror_descent':
                    # Mirror descent: z_new = z * exp(alpha * grad / ||grad||)
                    z_t = z_t * np.exp(alpha_base * grad_j / grad_norm)
                    if self.constraint_mask is not None:
                        z_t[~self.constraint_mask] = 0.0
                    z_t = np.clip(z_t, 0.0, 1.0)
                    # Check for improvement
                    if val > val_best:
                        z_best = z_t.copy()
                        val_best = val
                    continue
                else:
                    step = 0.0
                
                # Gradient ascent step (MAXIMIZE objective: MinCut minimizes -P, so maximizes P)
                z_new = z_t + step * grad_j
            else:
                z_new = z_t.copy()
            
            # Apply constraints
            if self.constraint_mask is not None:
                z_new[~self.constraint_mask] = 0.0
            
            # Project to [0, 1]
            z_t = np.clip(z_new, 0.0, 1.0)
            
            # Track best solution
            if val > val_best:
                z_best = z_t.copy()
                val_best = val
        
        # Round to binary solution
        optimal_bundle = (z_best > 0.5)
        
        if self.subproblem_settings.get("verbose", False):
            violations_rounding = ((z_best > 0.1) & (z_best < 0.9)).sum()
            aggregate_demand = optimal_bundle.sum()
            print(f"violations: {violations_rounding}, demand: {aggregate_demand}, final_val: {val_best:.6f}")
        
        return optimal_bundle

    def _grad_lovatz_extension(self, z_i_j):
        """
        Compute gradient of Lovász extension for quadratic supermodular function.
        
        For quadratic form f(x) = x^T P x where P is upper triangular:
        - The gradient at x is: grad = (P + P^T) * x = P*x + P^T*x
        - For Lovász extension, we evaluate at sorted points
        - At sorted point z (descending order), gradient[k] = sum_{j<=k} (P[sigma[j], sigma[k]] + P[sigma[k], sigma[j]])
        """
        if z_i_j.ndim == 2:
            # Batched version
            n_agents, n_items = z_i_j.shape
            sigma_i_j = np.argsort(z_i_j, axis=1)[:, ::-1]  # Descending order
            P_sigma = np.take_along_axis(
                np.take_along_axis(self.P_j_j, sigma_i_j[:, :, None], axis=1),
                sigma_i_j[:, None, :], axis=2
            )
            # Make symmetric: P + P^T (diagonal counted once)
            P_sigma_sym = P_sigma + P_sigma.transpose(0, 2, 1)
            np.einsum('ijj->ij', P_sigma_sym)[:] = np.diagonal(P_sigma, axis1=1, axis2=2)
            
            # Gradient: sum over lower triangle (including diagonal)
            mask = np.tril(np.ones((n_items, n_items), dtype=bool), k=0)
            grad_i_sigma = (P_sigma_sym * mask[None, :, :]).sum(axis=2)
            
            grad_i_j = np.zeros_like(z_i_j)
            np.put_along_axis(grad_i_j, sigma_i_j, grad_i_sigma, axis=1)
            fun_value_i = (z_i_j * grad_i_j).sum(axis=1)
            return grad_i_j, fun_value_i
        else:
            # Single-agent version (optimized)
            n_items = z_i_j.shape[0]
            sigma_j = np.argsort(z_i_j)[::-1]  # Descending order
            
            # Permute P according to sorted order
            P_sigma = self.P_j_j[np.ix_(sigma_j, sigma_j)]
            
            # Make symmetric: P + P^T (diagonal counted once)
            P_sigma_sym = P_sigma + P_sigma.T
            np.fill_diagonal(P_sigma_sym, np.diag(P_sigma))
            
            # Gradient at sorted point: sum over lower triangle (including diagonal)
            # This computes: grad[sigma[k]] = sum_{j=0}^{k} P_sym[sigma[j], sigma[k]]
            grad_sigma = (P_sigma_sym * self._tril_mask).sum(axis=1)
            
            # Map back to original order
            grad_j = np.zeros_like(z_i_j)
            grad_j[sigma_j] = grad_sigma
            
            # Function value: z^T * grad (Lovász extension value)
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