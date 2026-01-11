"""Resampling-based standard errors: Bootstrap, Subsampling, Bayesian Bootstrap."""

from typing import Optional, Callable, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

from .result import StandardErrorsResult
from bundlechoice.utils import get_logger

if TYPE_CHECKING:
    from bundlechoice.estimation.row_generation import RowGenerationSolver

logger = get_logger(__name__)


class ResamplingMixin:
    """Mixin providing resampling-based SE methods for StandardErrorsManager."""
    
    def compute_bootstrap(
        self,
        theta_hat: NDArray[np.float64],
        solve_fn: Callable[[dict], Optional[NDArray[np.float64]]],
        num_bootstrap: int = 100,
        beta_indices: Optional[NDArray[np.int64]] = None,
        seed: Optional[int] = None,
    ) -> Optional[StandardErrorsResult]:
        """
        Standard bootstrap: resample agents with replacement.
        
        NOTE: solve_fn must handle MPI - all ranks must call it together.
        """
        if beta_indices is None:
            beta_indices = np.arange(self.num_features, dtype=np.int64)
        
        if self.is_root():
            lines = ["=" * 70, "STANDARD ERRORS (BOOTSTRAP)", "=" * 70]
            lines.append(f"  Samples: {num_bootstrap}, Parameters: {len(beta_indices)}")
            logger.info("\n".join(lines))
            
            if seed is not None:
                np.random.seed(seed)
            
            obs_bundles = self.data_manager.input_data["obs_bundle"]
            agent_data = self.data_manager.input_data["agent_data"]
            item_data = self.data_manager.input_data.get("item_data")
            N = self.num_agents
        
        theta_boots = []
        
        for b in range(num_bootstrap):
            if self.is_root() and (b + 1) % 20 == 0:
                logger.info("  Bootstrap %d/%d...", b + 1, num_bootstrap)
            
            # Root generates bootstrap data, broadcasts to all ranks
            if self.is_root():
                idx = np.random.choice(N, size=N, replace=True)
                boot_data = {
                    "obs_bundle": obs_bundles[idx],
                    "agent_data": {k: v[idx] for k, v in agent_data.items()},
                    "errors": np.random.randn(N, self.num_items),
                }
                if item_data is not None:
                    boot_data["item_data"] = item_data
            else:
                boot_data = None
            
            # All ranks call solve_fn (it handles MPI internally)
            theta_b = solve_fn(boot_data)
            
            if self.is_root() and theta_b is not None:
                theta_boots.append(theta_b)
        
        if not self.is_root():
            return None
        
        return self._finalize_resampling_result(theta_hat, theta_boots, beta_indices, "Bootstrap")
    
    def compute_subsampling(
        self,
        theta_hat: NDArray[np.float64],
        solve_fn: Callable[[dict], Optional[NDArray[np.float64]]],
        subsample_size: Optional[int] = None,
        num_subsamples: int = 100,
        beta_indices: Optional[NDArray[np.int64]] = None,
        seed: Optional[int] = None,
    ) -> Optional[StandardErrorsResult]:
        """
        Subsampling: draw subsamples without replacement, SE = sqrt(b/N) * std.
        
        NOTE: solve_fn must handle MPI - all ranks must call it together.
        """
        if beta_indices is None:
            beta_indices = np.arange(self.num_features, dtype=np.int64)
        
        N = self.num_agents
        b = min(subsample_size or int(N ** 0.7), N - 1)
        
        if self.is_root():
            lines = ["=" * 70, "STANDARD ERRORS (SUBSAMPLING)", "=" * 70]
            lines.append(f"  Subsamples: {num_subsamples}, size: {b} (N={N})")
            logger.info("\n".join(lines))
            
            if seed is not None:
                np.random.seed(seed)
            
            obs_bundles = self.data_manager.input_data["obs_bundle"]
            agent_data = self.data_manager.input_data["agent_data"]
            item_data = self.data_manager.input_data.get("item_data")
        
        theta_subs = []
        
        for s in range(num_subsamples):
            if self.is_root() and (s + 1) % 20 == 0:
                logger.info("  Subsample %d/%d...", s + 1, num_subsamples)
            
            # Root generates subsample data
            if self.is_root():
                idx = np.random.choice(N, size=b, replace=False)
                sub_data = {
                    "obs_bundle": obs_bundles[idx],
                    "agent_data": {k: v[idx] for k, v in agent_data.items()},
                    "errors": np.random.randn(b, self.num_items),
                }
                if item_data is not None:
                    sub_data["item_data"] = item_data
            else:
                sub_data = None
            
            # All ranks call solve_fn
            theta_s = solve_fn(sub_data)
            
            if self.is_root() and theta_s is not None:
                theta_subs.append(theta_s)
        
        if not self.is_root():
            return None
        
        return self._finalize_resampling_result(
            theta_hat, theta_subs, beta_indices, "Subsampling",
            scale_factor=np.sqrt(b / N)
        )
    
    def compute_bayesian_bootstrap(
        self,
        theta_hat: NDArray[np.float64],
        row_generation: "RowGenerationSolver",
        num_bootstrap: int = 100,
        beta_indices: Optional[NDArray[np.int64]] = None,
        seed: Optional[int] = None,
        reuse_constraints: bool = True,
    ) -> Optional[StandardErrorsResult]:
        """
        Bayesian bootstrap: reweight agents with Exp(1) weights instead of resampling.
        
        Args:
            reuse_constraints: If True, warm-start each solve with constraints from previous solve.
                             This significantly speeds up computation since only weights change.
        """
        if beta_indices is None:
            beta_indices = np.arange(self.num_features, dtype=np.int64)
        
        if self.is_root():
            lines = ["=" * 70, "STANDARD ERRORS (BAYESIAN BOOTSTRAP)", "=" * 70]
            lines.append(f"  Samples: {num_bootstrap}, Parameters: {len(beta_indices)}")
            if reuse_constraints:
                lines.append("  Warm-start: reusing constraints across samples")
            logger.info("\n".join(lines))
        
        N = self.num_agents
        theta_boots = []
        constraints = None  # Will hold constraints from previous solve
        
        if seed is not None:
            np.random.seed(seed)
        
        for b in range(num_bootstrap):
            if self.is_root() and (b + 1) % 20 == 0:
                logger.info("  Bayesian bootstrap %d/%d...", b + 1, num_bootstrap)
            
            if self.is_root():
                weights = np.random.exponential(1.0, N)
                weights = weights / weights.mean()
            else:
                weights = None
            weights = self.comm.bcast(weights, root=0)
            
            try:
                # Pass constraints from previous solve for warm-start
                result = row_generation.solve(
                    agent_weights=weights,
                    initial_constraints=constraints if reuse_constraints else None
                )
                if self.is_root():
                    theta_boots.append(result.theta_hat)
                    # Get constraints for next iteration (only on first or if reusing)
                    if reuse_constraints:
                        constraints = row_generation.get_constraints()
                        if b == 0:
                            n_constr = len(constraints.get('indices', [])) if constraints else 0
                            logger.debug("After first solve: got %d constraints for warm-start", n_constr)
            except Exception:
                pass
        
        if not self.is_root():
            return None
        
        return self._finalize_resampling_result(theta_hat, theta_boots, beta_indices, "Bayesian Bootstrap")
    
    def _finalize_resampling_result(
        self,
        theta_hat: NDArray[np.float64],
        theta_samples: list,
        beta_indices: NDArray[np.int64],
        method_name: str,
        scale_factor: float = 1.0,
    ) -> Optional[StandardErrorsResult]:
        """Compute SE from resampling results and format output."""
        if len(theta_samples) < 10:
            logger.error("  Too few successful samples (%d)", len(theta_samples))
            return None
        
        theta_samples = np.array(theta_samples)
        se_all = scale_factor * np.std(theta_samples, axis=0, ddof=1)
        se = se_all[beta_indices]
        theta_beta = theta_hat[beta_indices]
        t_stats = np.where(se > 1e-16, theta_beta / se, np.nan)
        
        lines = ["-" * 70, f"Standard Errors ({method_name}):", "-" * 70]
        for i, idx in enumerate(beta_indices):
            lines.append(f"  θ[{idx}] = {theta_hat[idx]:.6f}, SE = {se[i]:.6f}, t = {t_stats[i]:.2f}")
        logger.info("\n".join(lines))
        
        n_params = len(beta_indices)
        return StandardErrorsResult(
            se=se,
            se_all=se_all,
            theta_beta=theta_beta,
            beta_indices=beta_indices,
            variance=np.diag(se ** 2),
            A_matrix=np.eye(n_params),
            B_matrix=np.eye(n_params),
            t_stats=t_stats,
        )
