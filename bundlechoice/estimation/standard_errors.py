"""
Sandwich standard errors computation for bundle choice estimation.

Computes sandwich standard errors via:
1. B matrix: (1/N) sum_i g_i g_i^T (outer product of per-agent subgradients)
2. A matrix: Jacobian via finite differences
3. Variance: (1/N) A^{-1} B A^{-1}
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from numpy.typing import NDArray
from mpi4py import MPI

from bundlechoice.base import HasDimensions, HasData, HasComm
from bundlechoice.config import DimensionsConfig, StandardErrorsConfig
from bundlechoice.comm_manager import CommManager
from bundlechoice.data_manager import DataManager
from bundlechoice.feature_manager import FeatureManager
from bundlechoice.subproblems.subproblem_manager import SubproblemManager


# =============================================================================
# Numerical Utilities
# =============================================================================

def compute_adaptive_step_size(theta_k: float, base_step: float = 1e-4) -> float:
    """
    Compute adaptive step size for finite differences.
    
    Scales step with parameter magnitude while respecting precision limits.
    """
    eps = np.finfo(np.float64).eps
    min_step = np.sqrt(eps)  # ~1.5e-8
    
    # Scale by parameter magnitude
    scale = max(1.0, abs(theta_k))
    h = base_step * scale
    
    # Ensure step is not too small (numerical precision)
    h = max(h, min_step * scale)
    
    # Ensure step is not too large relative to parameter
    if abs(theta_k) > 0:
        h = min(h, 0.1 * abs(theta_k))  # At most 10% of parameter value
    
    return h


@dataclass
class StandardErrorsResult:
    """Result of standard errors computation."""
    se: NDArray[np.float64]  # Standard errors for selected parameters
    se_all: NDArray[np.float64]  # Standard errors for all parameters
    theta_beta: NDArray[np.float64]  # Parameter values for selected
    beta_indices: NDArray[np.int64]  # Which parameters were selected
    variance: NDArray[np.float64]  # Full variance matrix
    A_matrix: NDArray[np.float64]  # Jacobian matrix
    B_matrix: NDArray[np.float64]  # Outer product matrix
    t_stats: NDArray[np.float64]  # t-statistics for selected parameters


class StandardErrorsManager(HasDimensions, HasData, HasComm):
    """
    Computes sandwich standard errors for bundle choice estimation.
    
    The sandwich estimator is: Var(theta) = (1/N) A^{-1} B A^{-1}
    where:
    - B = (1/N) sum_i g_i g_i^T (outer product of per-agent subgradients)
    - A = Jacobian of average subgradient (computed via finite differences)
    """
    
    def __init__(
        self,
        comm_manager: CommManager,
        dimensions_cfg: DimensionsConfig,
        data_manager: DataManager,
        feature_manager: FeatureManager,
        subproblem_manager: SubproblemManager,
        se_cfg: StandardErrorsConfig,
    ):
        # Use consistent naming with other managers
        self.comm_manager = comm_manager
        self.dimensions_cfg = dimensions_cfg
        self.data_manager = data_manager
        self.feature_manager = feature_manager
        self.subproblem_manager = subproblem_manager
        self.se_cfg = se_cfg
        self._obs_features: Optional[NDArray[np.float64]] = None
        # Caches for mean observed features (avoid recomputation in Jacobian)
        self._mean_obs_full: Optional[NDArray[np.float64]] = None
        self._mean_obs_subset: Optional[dict] = None  # keyed by tuple(beta_indices)
    
    def clear_cache(self) -> None:
        """Clear cached values. Call if underlying data changes."""
        self._obs_features = None
        self._mean_obs_full = None
        self._mean_obs_subset = None
    
    def compute(
        self,
        theta_hat: NDArray[np.float64],
        num_simulations: Optional[int] = None,
        step_size: Optional[float] = None,
        beta_indices: Optional[NDArray[np.int64]] = None,
        seed: Optional[int] = None,
        optimize_for_subset: bool = True,
        error_sigma: Optional[float] = None,
    ) -> Optional[StandardErrorsResult]:
        """
        Compute sandwich standard errors.
        
        Args:
            theta_hat: Estimated parameter vector (root rank only)
            num_simulations: Number of simulations for SE (overrides config)
            step_size: Step size for finite differences (overrides config)
            beta_indices: Which parameters to compute SE for (default: all)
            seed: Random seed for error generation
            optimize_for_subset: If True and beta_indices provided, only compute
                matrices for the subset (faster). If False, compute full matrices.
            error_sigma: Std dev of errors (should match estimation errors, default: 1.0)
            
        Returns:
            StandardErrorsResult on root rank, None on other ranks
        """
        # Use config defaults if not specified
        num_simulations = num_simulations or self.se_cfg.num_simulations
        step_size = step_size or self.se_cfg.step_size
        seed = seed if seed is not None else self.se_cfg.seed
        error_sigma = error_sigma if error_sigma is not None else self.se_cfg.error_sigma
        
        # Broadcast theta to all ranks
        theta_hat = self.comm.bcast(theta_hat, root=0)
        
        # Default: all parameters
        if beta_indices is None:
            beta_indices = np.arange(self.num_features, dtype=np.int64)
        beta_indices = self.comm.bcast(beta_indices, root=0)
        
        # Generate errors for SE computation
        errors_all_sims = self._generate_se_errors(num_simulations, seed, error_sigma)
        
        # Cache observed features (gathered to root)
        if self._obs_features is None:
            obs_bundles = self.local_data["obs_bundles"]
            self._obs_features = self.feature_manager.compute_gathered_features(obs_bundles)
        
        # Determine if we should optimize for subset
        is_subset = len(beta_indices) < self.num_features
        use_subset_opt = optimize_for_subset and is_subset
        
        # Print header
        if self.is_root():
            print("\n" + "=" * 70)
            print("STANDARD ERRORS COMPUTATION")
            print("=" * 70)
            print(f"  Simulations: {num_simulations}")
            print(f"  Step size: {step_size}")
            print(f"  Total parameters: {self.num_features}")
            if use_subset_opt:
                print(f"  Computing subset: {len(beta_indices)} params (optimized)")
        
        if use_subset_opt:
            # Optimized: only compute for beta_indices subset
            B_sub = self._compute_B_matrix_subset(theta_hat, errors_all_sims, beta_indices)
            A_sub = self._compute_A_matrix_subset(theta_hat, errors_all_sims, step_size, beta_indices)
            
            self.comm.Barrier()
            if self.is_root():
                A_cond = np.linalg.cond(A_sub)
                B_cond = np.linalg.cond(B_sub)
                print(f"\n  A matrix: cond={A_cond:.2e}")
                print(f"  B matrix: cond={B_cond:.2e}")
                
                # Check for singular/ill-conditioned A matrix
                if not np.isfinite(A_cond) or A_cond > 1e16:
                    print("\n  ❌ ERROR: A matrix is singular or extremely ill-conditioned!")
                    print("     Cannot compute standard errors reliably.")
                    print("     Try: more simulations, larger sample, or check model specification.")
                    return None
                
                # Strict inversion - no pseudoinverse
                try:
                    A_inv = np.linalg.solve(A_sub, np.eye(len(beta_indices)))
                except np.linalg.LinAlgError:
                    print("\n  ❌ ERROR: A matrix is singular, cannot invert!")
                    return None
                
                # Sandwich formula
                V_sub = (1.0 / self.num_agents) * (A_inv @ B_sub @ A_inv.T)
                
                # Check for valid variance
                diag_V = np.diag(V_sub)
                if np.any(diag_V < 0):
                    neg_count = np.sum(diag_V < 0)
                    print(f"\n  ⚠ WARNING: {neg_count} negative variances detected!")
                
                se_beta = np.sqrt(np.maximum(np.diag(V_sub), 0))
                theta_beta = theta_hat[beta_indices]
                t_stats = np.where(se_beta > 1e-16, theta_beta / se_beta, np.nan)
                
                print("\n" + "-" * 70)
                print("Standard Errors:")
                print("-" * 70)
                for i, idx in enumerate(beta_indices):
                    print(f"  θ[{idx}] = {theta_hat[idx]:.6f}, SE = {se_beta[i]:.6f}, t = {t_stats[i]:.2f}")
                
                return StandardErrorsResult(
                    se=se_beta,
                    se_all=se_beta,  # Only subset computed
                    theta_beta=theta_beta,
                    beta_indices=beta_indices,
                    variance=V_sub,
                    A_matrix=A_sub,
                    B_matrix=B_sub,
                    t_stats=t_stats,
                )
            return None
        
        # Full computation
        B_full = self._compute_B_matrix(theta_hat, errors_all_sims)
        A_full = self._compute_A_matrix(theta_hat, errors_all_sims, step_size)
        
        self.comm.Barrier()
        if self.is_root():
            A_cond = np.linalg.cond(A_full)
            B_cond = np.linalg.cond(B_full)
            print(f"\n  A matrix: cond={A_cond:.2e}")
            print(f"  B matrix: cond={B_cond:.2e}")
            
            # Check for singular/ill-conditioned A matrix
            if not np.isfinite(A_cond) or A_cond > 1e16:
                print("\n  ❌ ERROR: A matrix is singular or extremely ill-conditioned!")
                print("     Cannot compute standard errors reliably.")
                print("     Try: more simulations, larger sample, or check model specification.")
                return None
            
            # Strict inversion - no pseudoinverse
            try:
                A_inv = np.linalg.solve(A_full, np.eye(self.num_features))
            except np.linalg.LinAlgError:
                print("\n  ❌ ERROR: A matrix is singular, cannot invert!")
                return None
            
            # Sandwich formula
            V_full = (1.0 / self.num_agents) * (A_inv @ B_full @ A_inv.T)
            
            # Check for valid variance
            diag_V = np.diag(V_full)
            if np.any(diag_V < 0):
                neg_count = np.sum(diag_V < 0)
                print(f"\n  ⚠ WARNING: {neg_count} negative variances detected!")
            
            se_all = np.sqrt(np.maximum(np.diag(V_full), 0))
            
            se_beta = se_all[beta_indices]
            theta_beta = theta_hat[beta_indices]
            t_stats = np.where(se_beta > 1e-16, theta_beta / se_beta, np.nan)
            
            print("\n" + "-" * 70)
            print("Standard Errors:")
            print("-" * 70)
            for i, idx in enumerate(beta_indices):
                print(f"  θ[{idx}] = {theta_hat[idx]:.6f}, SE = {se_beta[i]:.6f}, t = {t_stats[i]:.2f}")
            
            return StandardErrorsResult(
                se=se_beta,
                se_all=se_all,
                theta_beta=theta_beta,
                beta_indices=beta_indices,
                variance=V_full,
                A_matrix=A_full,
                B_matrix=B_full,
                t_stats=t_stats,
            )
        return None
    
    def compute_B_inverse(
        self,
        theta_hat: NDArray[np.float64],
        num_simulations: Optional[int] = None,
        beta_indices: Optional[NDArray[np.int64]] = None,
        seed: Optional[int] = None,
        error_sigma: Optional[float] = None,
    ) -> Optional[StandardErrorsResult]:
        """
        Compute standard errors using B^{-1} only (no finite differences).
        
        This is faster than the full sandwich estimator since it doesn't require
        computing the A matrix via finite differences. Uses Var(θ) = (1/N) B^{-1}.
        
        Args:
            theta_hat: Estimated parameter vector (root rank only)
            num_simulations: Number of simulations for B matrix
            beta_indices: Which parameters to compute SE for (default: all)
            seed: Random seed for error generation
            error_sigma: Std dev of errors (should match estimation errors, default: 1.0)
            
        Returns:
            StandardErrorsResult on root rank, None on other ranks
        """
        num_simulations = num_simulations or self.se_cfg.num_simulations
        seed = seed if seed is not None else self.se_cfg.seed
        error_sigma = error_sigma if error_sigma is not None else self.se_cfg.error_sigma
        
        theta_hat = self.comm.bcast(theta_hat, root=0)
        
        if beta_indices is None:
            beta_indices = np.arange(self.num_features, dtype=np.int64)
        beta_indices = self.comm.bcast(beta_indices, root=0)
        
        errors_all_sims = self._generate_se_errors(num_simulations, seed, error_sigma)
        
        if self._obs_features is None:
            obs_bundles = self.local_data["obs_bundles"]
            self._obs_features = self.feature_manager.compute_gathered_features(obs_bundles)
        
        is_subset = len(beta_indices) < self.num_features
        
        if self.is_root():
            print("\n" + "=" * 70)
            print("STANDARD ERRORS (B-INVERSE METHOD)")
            print("=" * 70)
            print(f"  Simulations: {num_simulations}")
            print(f"  Parameters: {len(beta_indices)}")
        
        if is_subset:
            B_mat = self._compute_B_matrix_subset(theta_hat, errors_all_sims, beta_indices)
        else:
            B_mat = self._compute_B_matrix(theta_hat, errors_all_sims)
        
        self.comm.Barrier()
        
        if self.is_root():
            B_cond = np.linalg.cond(B_mat)
            print(f"\n  B matrix: cond={B_cond:.2e}")
            
            if not np.isfinite(B_cond) or B_cond > 1e16:
                print("\n  ❌ ERROR: B matrix is singular!")
                return None
            
            try:
                B_inv = np.linalg.solve(B_mat, np.eye(len(beta_indices)))
            except np.linalg.LinAlgError:
                print("\n  ❌ ERROR: B matrix is singular, cannot invert!")
                return None
            
            # Variance = (1/N) B^{-1}
            V = B_inv / self.num_agents
            
            diag_V = np.diag(V)
            if np.any(diag_V < 0):
                neg_count = np.sum(diag_V < 0)
                print(f"\n  ⚠ WARNING: {neg_count} negative variances detected!")
            
            se = np.sqrt(np.maximum(np.diag(V), 0))
            theta_beta = theta_hat[beta_indices]
            t_stats = np.where(se > 1e-16, theta_beta / se, np.nan)
            
            print("\n" + "-" * 70)
            print("Standard Errors (B-inverse):")
            print("-" * 70)
            for i, idx in enumerate(beta_indices):
                print(f"  θ[{idx}] = {theta_hat[idx]:.6f}, SE = {se[i]:.6f}, t = {t_stats[i]:.2f}")
            
            # Create dummy A matrix (identity) for result structure
            A_dummy = np.eye(len(beta_indices))
            
            return StandardErrorsResult(
                se=se,
                se_all=se,
                theta_beta=theta_beta,
                beta_indices=beta_indices,
                variance=V,
                A_matrix=A_dummy,
                B_matrix=B_mat,
                t_stats=t_stats,
            )
        return None
    
    def compute_bootstrap(
        self,
        theta_hat: NDArray[np.float64],
        solve_fn,  # Function that takes (bc, data) and returns theta_hat
        num_bootstrap: int = 100,
        beta_indices: Optional[NDArray[np.int64]] = None,
        seed: Optional[int] = None,
    ) -> Optional[StandardErrorsResult]:
        """
        Compute standard errors via bootstrap resampling of agents.
        
        Resamples agents with replacement and re-estimates theta for each bootstrap sample.
        SE = std(theta_hat across bootstrap samples).
        
        Args:
            theta_hat: Original parameter estimate
            solve_fn: Function(bc) -> theta_hat that runs estimation
            num_bootstrap: Number of bootstrap samples
            beta_indices: Which parameters to report (default: all)
            seed: Random seed
        """
        if beta_indices is None:
            beta_indices = np.arange(self.num_features, dtype=np.int64)
        
        if self.is_root():
            print("\n" + "=" * 70)
            print("STANDARD ERRORS (BOOTSTRAP)")
            print("=" * 70)
            print(f"  Bootstrap samples: {num_bootstrap}")
            print(f"  Parameters: {len(beta_indices)}")
            
            if seed is not None:
                np.random.seed(seed)
            
            # Get original data
            obs_bundles = self.data_manager.full_data["obs_bundles"]
            agent_data = self.data_manager.full_data["agent_data"]
            item_data = self.data_manager.full_data.get("item_data")
            N = self.num_agents
            
            theta_boots = []
            for b in range(num_bootstrap):
                if (b + 1) % 20 == 0:
                    print(f"  Bootstrap {b+1}/{num_bootstrap}...")
                
                # Resample agents with replacement
                idx = np.random.choice(N, size=N, replace=True)
                
                boot_data = {
                    "obs_bundle": obs_bundles[idx],
                    "agent_data": {k: v[idx] for k, v in agent_data.items()},
                    "errors": np.random.randn(N, self.num_items),
                }
                if item_data is not None:
                    boot_data["item_data"] = item_data
                
                # Re-estimate on bootstrap sample
                theta_b = solve_fn(boot_data)
                if theta_b is not None:
                    theta_boots.append(theta_b)
            
            if len(theta_boots) < 10:
                print("  ❌ Too few successful bootstrap samples")
                return None
            
            theta_boots = np.array(theta_boots)
            se_all = np.std(theta_boots, axis=0, ddof=1)
            se = se_all[beta_indices]
            theta_beta = theta_hat[beta_indices]
            t_stats = np.where(se > 1e-16, theta_beta / se, np.nan)
            
            print("\n" + "-" * 70)
            print("Standard Errors (Bootstrap):")
            print("-" * 70)
            for i, idx in enumerate(beta_indices):
                print(f"  θ[{idx}] = {theta_hat[idx]:.6f}, SE = {se[i]:.6f}, t = {t_stats[i]:.2f}")
            
            return StandardErrorsResult(
                se=se, se_all=se_all, theta_beta=theta_beta,
                beta_indices=beta_indices,
                variance=np.diag(se**2),
                A_matrix=np.eye(len(beta_indices)),
                B_matrix=np.eye(len(beta_indices)),
                t_stats=t_stats,
            )
        return None
    
    def compute_subsampling(
        self,
        theta_hat: NDArray[np.float64],
        solve_fn,  # Function that takes data dict and returns theta_hat
        subsample_size: Optional[int] = None,
        num_subsamples: int = 100,
        beta_indices: Optional[NDArray[np.int64]] = None,
        seed: Optional[int] = None,
    ) -> Optional[StandardErrorsResult]:
        """
        Compute standard errors via subsampling.
        
        Draws subsamples of size b < N without replacement and re-estimates.
        SE = sqrt(b/N) * std(theta_hat across subsamples).
        
        Args:
            theta_hat: Original parameter estimate
            solve_fn: Function(data_dict) -> theta_hat that runs estimation
            subsample_size: Size of each subsample (default: N^0.7)
            num_subsamples: Number of subsamples
            beta_indices: Which parameters to report
            seed: Random seed
        """
        if beta_indices is None:
            beta_indices = np.arange(self.num_features, dtype=np.int64)
        
        N = self.num_agents
        if subsample_size is None:
            subsample_size = int(N ** 0.7)  # Standard choice
        b = min(subsample_size, N - 1)
        
        if self.is_root():
            print("\n" + "=" * 70)
            print("STANDARD ERRORS (SUBSAMPLING)")
            print("=" * 70)
            print(f"  Subsamples: {num_subsamples}, size: {b} (N={N})")
            print(f"  Parameters: {len(beta_indices)}")
            
            if seed is not None:
                np.random.seed(seed)
            
            obs_bundles = self.data_manager.full_data["obs_bundles"]
            agent_data = self.data_manager.full_data["agent_data"]
            item_data = self.data_manager.full_data.get("item_data")
            
            theta_subs = []
            for s in range(num_subsamples):
                if (s + 1) % 20 == 0:
                    print(f"  Subsample {s+1}/{num_subsamples}...")
                
                # Sample without replacement
                idx = np.random.choice(N, size=b, replace=False)
                
                sub_data = {
                    "obs_bundle": obs_bundles[idx],
                    "agent_data": {k: v[idx] for k, v in agent_data.items()},
                    "errors": np.random.randn(b, self.num_items),
                }
                if item_data is not None:
                    sub_data["item_data"] = item_data
                
                theta_s = solve_fn(sub_data)
                if theta_s is not None:
                    theta_subs.append(theta_s)
            
            if len(theta_subs) < 10:
                print("  ❌ Too few successful subsamples")
                return None
            
            theta_subs = np.array(theta_subs)
            # Subsampling scaling: SE = sqrt(b/N) * std
            se_all = np.sqrt(b / N) * np.std(theta_subs, axis=0, ddof=1)
            se = se_all[beta_indices]
            theta_beta = theta_hat[beta_indices]
            t_stats = np.where(se > 1e-16, theta_beta / se, np.nan)
            
            print("\n" + "-" * 70)
            print("Standard Errors (Subsampling):")
            print("-" * 70)
            for i, idx in enumerate(beta_indices):
                print(f"  θ[{idx}] = {theta_hat[idx]:.6f}, SE = {se[i]:.6f}, t = {t_stats[i]:.2f}")
            
            return StandardErrorsResult(
                se=se, se_all=se_all, theta_beta=theta_beta,
                beta_indices=beta_indices,
                variance=np.diag(se**2),
                A_matrix=np.eye(len(beta_indices)),
                B_matrix=np.eye(len(beta_indices)),
                t_stats=t_stats,
            )
        return None
    
    def _generate_se_errors(
        self, num_simulations: int, seed: Optional[int], error_sigma: float = 1.0
    ) -> NDArray[np.float64]:
        """Generate errors for SE computation."""
        if self.is_root():
            if seed is not None:
                np.random.seed(seed)
            errors = error_sigma * np.random.normal(0, 1, (num_simulations, self.num_agents, self.num_items))
        else:
            errors = None
        return self.comm.bcast(errors, root=0)
    
    def _compute_B_matrix(
        self,
        theta: NDArray[np.float64],
        errors_all_sims: NDArray[np.float64],
    ) -> Optional[NDArray[np.float64]]:
        """
        Compute B matrix: (1/N) sum_i g_i g_i^T.
        
        g_i = (1/S) sum_s x_{i,B_i^{s,*}} - x_{i,B_i^obs}
        """
        num_simulations = len(errors_all_sims)
        
        if self.is_root():
            print(f"\nComputing B matrix ({self.num_features}×{self.num_features})...")
        
        all_features_per_sim = []
        for s in range(num_simulations):
            if self.is_root():
                print(f"  Simulation {s+1}/{num_simulations}...")
            
            # Update errors for this simulation
            self.data_manager.update_errors(errors_all_sims[s] if self.is_root() else None)
            
            # Solve subproblems
            if self.num_local_agents > 0:
                local_bundles = self.subproblem_manager.solve_local(theta)
            else:
                local_bundles = np.empty((0, self.num_items), dtype=bool)
            
            # Gather features to root
            features_sim = self.feature_manager.compute_gathered_features(local_bundles)
            if self.is_root():
                all_features_per_sim.append(features_sim)
        
        self.comm.Barrier()
        
        if self.is_root():
            # Stack: (S, N, K)
            features_all = np.stack(all_features_per_sim, axis=0)
            avg_simulated = features_all.mean(axis=0)  # (N, K)
            
            # Per-agent subgradient: g_i = avg_sim - obs
            g_i_full = avg_simulated - self._obs_features  # (N, K)
            
            # B = (1/N) sum_i g_i g_i^T
            B_full = (g_i_full.T @ g_i_full) / self.num_agents
            print(f"  B matrix: cond={np.linalg.cond(B_full):.2e}")
            return B_full
        return None
    
    def _compute_B_matrix_subset(
        self,
        theta: NDArray[np.float64],
        errors_all_sims: NDArray[np.float64],
        beta_indices: NDArray[np.int64],
    ) -> Optional[NDArray[np.float64]]:
        """Compute B matrix for subset of parameters only."""
        num_simulations = len(errors_all_sims)
        num_beta = len(beta_indices)
        
        if self.is_root():
            print(f"\nComputing B matrix ({num_beta}×{num_beta})...")
        
        all_features_per_sim = []
        for s in range(num_simulations):
            if self.is_root():
                print(f"  Simulation {s+1}/{num_simulations}...")
            
            self.data_manager.update_errors(errors_all_sims[s] if self.is_root() else None)
            
            if self.num_local_agents > 0:
                local_bundles = self.subproblem_manager.solve_local(theta)
            else:
                local_bundles = np.empty((0, self.num_items), dtype=bool)
            
            features_sim = self.feature_manager.compute_gathered_features(local_bundles)
            if self.is_root():
                all_features_per_sim.append(features_sim)
        
        self.comm.Barrier()
        
        if self.is_root():
            features_all = np.stack(all_features_per_sim, axis=0)
            avg_simulated = features_all.mean(axis=0)[:, beta_indices]
            obs_beta = self._obs_features[:, beta_indices]
            g_i_beta = avg_simulated - obs_beta
            B_beta = (g_i_beta.T @ g_i_beta) / self.num_agents
            print(f"  B matrix: cond={np.linalg.cond(B_beta):.2e}")
            return B_beta
        return None
    
    def _compute_A_matrix(
        self,
        theta: NDArray[np.float64],
        errors_all_sims: NDArray[np.float64],
        step_size: float,
    ) -> Optional[NDArray[np.float64]]:
        """
        Compute A matrix via finite differences.
        
        A[:, k] = (g_bar(theta + h_k e_k) - g_bar(theta - h_k e_k)) / (2 h_k)
        """
        if self.is_root():
            print(f"Computing A matrix ({self.num_features}×{self.num_features}, {self.num_features} columns)...")
            A_full = np.zeros((self.num_features, self.num_features))
        else:
            A_full = None
        
        # Compute for all K parameters
        for k in range(self.num_features):
            if self.is_root():
                print(f"  Column {k+1}/{self.num_features}...")
            
            h_k = compute_adaptive_step_size(theta[k], step_size)
            
            theta_plus = theta.copy()
            theta_plus[k] += h_k
            theta_minus = theta.copy()
            theta_minus[k] -= h_k
            
            g_plus = self._compute_avg_subgradient(theta_plus, errors_all_sims)
            g_minus = self._compute_avg_subgradient(theta_minus, errors_all_sims)
            
            if self.is_root():
                A_full[:, k] = (g_plus - g_minus) / (2 * h_k)
        
        self.comm.Barrier()
        return A_full
    
    def _compute_A_matrix_subset(
        self,
        theta: NDArray[np.float64],
        errors_all_sims: NDArray[np.float64],
        step_size: float,
        beta_indices: NDArray[np.int64],
    ) -> Optional[NDArray[np.float64]]:
        """Compute A matrix for subset of parameters only (more efficient)."""
        num_beta = len(beta_indices)
        
        if self.is_root():
            print(f"Computing A matrix ({num_beta}×{num_beta}, {num_beta} columns)...")
            A_beta = np.zeros((num_beta, num_beta))
        else:
            A_beta = None
        
        for k_idx, k in enumerate(beta_indices):
            if self.is_root():
                print(f"  Column {k_idx+1}/{num_beta} (param {k})...")
            
            h_k = compute_adaptive_step_size(theta[k], step_size)
            
            theta_plus = theta.copy()
            theta_plus[k] += h_k
            theta_minus = theta.copy()
            theta_minus[k] -= h_k
            
            g_plus = self._compute_avg_subgradient_subset(theta_plus, errors_all_sims, beta_indices)
            g_minus = self._compute_avg_subgradient_subset(theta_minus, errors_all_sims, beta_indices)
            
            if self.is_root():
                A_beta[:, k_idx] = (g_plus - g_minus) / (2 * h_k)
        
        self.comm.Barrier()
        return A_beta
    
    def _compute_avg_subgradient_subset(
        self,
        theta: NDArray[np.float64],
        errors_all_sims: NDArray[np.float64],
        beta_indices: NDArray[np.int64],
    ) -> Optional[NDArray[np.float64]]:
        """Compute average subgradient for subset of features only."""
        num_simulations = len(errors_all_sims)
        num_beta = len(beta_indices)
        
        # Use cached mean_obs for this subset if available (doesn't depend on theta)
        cache_key = tuple(beta_indices)
        if self._mean_obs_subset is None:
            self._mean_obs_subset = {}
        
        if cache_key not in self._mean_obs_subset:
            obs_local = self.data_manager.local_data["obs_bundles"]
            obs_feat_local = self.feature_manager.compute_rank_features(obs_local)
            obs_sum_local = obs_feat_local[:, beta_indices].sum(axis=0) if obs_feat_local.size else np.zeros(num_beta)
            
            obs_sum_global = np.zeros(num_beta)
            self.comm.Allreduce(obs_sum_local, obs_sum_global, op=MPI.SUM)
            self._mean_obs_subset[cache_key] = obs_sum_global / self.num_agents
        
        mean_obs = self._mean_obs_subset[cache_key]
        
        sim_sum_local = np.zeros(num_beta)
        for s in range(num_simulations):
            self.data_manager.update_errors(errors_all_sims[s] if self.is_root() else None)
            
            if self.num_local_agents > 0:
                local_bundles = self.subproblem_manager.solve_local(theta)
            else:
                local_bundles = np.empty((0, self.num_items), dtype=bool)
            
            feat_local = self.feature_manager.compute_rank_features(local_bundles)
            if feat_local.size:
                sim_sum_local += feat_local[:, beta_indices].sum(axis=0)
        
        sim_sum_global = np.zeros(num_beta)
        self.comm.Allreduce(sim_sum_local, sim_sum_global, op=MPI.SUM)
        mean_sim = (sim_sum_global / num_simulations) / self.num_agents
        
        if self.is_root():
            return mean_sim - mean_obs
        return None
    
    def _compute_avg_subgradient(
        self,
        theta: NDArray[np.float64],
        errors_all_sims: NDArray[np.float64],
    ) -> Optional[NDArray[np.float64]]:
        """
        Compute average subgradient g_bar(theta) for all features.
        
        Uses MPI Allreduce for efficiency.
        """
        num_simulations = len(errors_all_sims)
        K = self.num_features
        
        # Use cached mean_obs if available (doesn't depend on theta)
        if self._mean_obs_full is None:
            obs_local = self.data_manager.local_data["obs_bundles"]
            obs_feat_local = self.feature_manager.compute_rank_features(obs_local)
            obs_sum_local = obs_feat_local.sum(axis=0) if obs_feat_local.size else np.zeros(K)
            
            obs_sum_global = np.zeros(K)
            self.comm.Allreduce(obs_sum_local, obs_sum_global, op=MPI.SUM)
            self._mean_obs_full = obs_sum_global / self.num_agents
        
        mean_obs = self._mean_obs_full
        
        # Simulated features sum
        sim_sum_local = np.zeros(K)
        for s in range(num_simulations):
            self.data_manager.update_errors(errors_all_sims[s] if self.is_root() else None)
            
            if self.num_local_agents > 0:
                local_bundles = self.subproblem_manager.solve_local(theta)
            else:
                local_bundles = np.empty((0, self.num_items), dtype=bool)
            
            feat_local = self.feature_manager.compute_rank_features(local_bundles)
            if feat_local.size:
                sim_sum_local += feat_local.sum(axis=0)
        
        sim_sum_global = np.zeros(K)
        self.comm.Allreduce(sim_sum_local, sim_sum_global, op=MPI.SUM)
        mean_sim = (sim_sum_global / num_simulations) / self.num_agents
        
        if self.is_root():
            return mean_sim - mean_obs
        return None
    
