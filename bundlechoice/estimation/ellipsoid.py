"""
Ellipsoid method solver for modular bundle choice estimation (v2).
This module implements the ellipsoid method for parameter estimation.
"""
import numpy as np
from numpy.typing import NDArray
from datetime import datetime
from typing import Optional, Callable, Dict, Any
from .base import BaseEstimationManager
from .result import EstimationResult
from bundlechoice.utils import get_logger

logger = get_logger(__name__)


class EllipsoidManager(BaseEstimationManager):
    """
    Implements the ellipsoid method for parameter estimation in modular bundle choice models.

    This solver uses the ellipsoid method to iteratively refine parameter estimates
    based on gradient information and constraint violations.
    """
    
    def __init__(
        self,
        comm_manager: Any,
        dimensions_cfg: Any,
        ellipsoid_cfg: Any,
        data_manager: Any,
        feature_manager: Any,
        subproblem_manager: Any,
        theta_init: Optional[NDArray[np.float64]] = None
    ) -> None:
        """
        Initialize the EllipsoidManager.

        Args:
            comm_manager: Communication manager for MPI operations
            dimensions_cfg: DimensionsConfig instance
            ellipsoid_cfg: EllipsoidConfig instance with method-specific parameters
            data_manager: DataManager instance
            feature_manager: FeatureManager instance
            subproblem_manager: SubproblemManager instance
            theta_init: Optional initial theta for warm start
        """
        super().__init__(
            comm_manager=comm_manager,
            dimensions_cfg=dimensions_cfg,
            data_manager=data_manager,
            feature_manager=feature_manager,
            subproblem_manager=subproblem_manager
        )
        
        self.ellipsoid_cfg = ellipsoid_cfg
        self.theta_init = theta_init
        
        # Initialize ellipsoid-specific attributes
        self.theta_iter = None
        self.B_iter = None
        self.iteration_count = 0
        self.timing_stats = None  # Store detailed timing statistics
        
        # Ellipsoid update coefficients
        n = self.num_features
        self.n = n
        self.alpha = (n**2 / (n**2 - 1))**(1/4)
        self.gamma = self.alpha * ((n-1)/(n+1))**(1/2)

        self.gamma_1 = (n**2 / (n**2 - 1))**(1/2)
        self.gamma_2 = self.gamma_1 * ((2 / (n + 1)))

    def solve(self, callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> EstimationResult:
        """
        Run the ellipsoid method to estimate model parameters.
        
        Returns:
            EstimationResult: Result object containing theta_hat and diagnostics.

        Args:
            callback: Optional callback function called after each iteration.
                     Signature: callback(info: dict) where info contains:
                     - 'iteration': Current iteration number (int)
                     - 'theta': Current parameter estimate (np.ndarray)
                     - 'objective': Current objective value (float)
                     - 'best_objective': Best objective found so far (float or None)

        Returns:
            np.ndarray: Estimated parameter vector.
        """
        logger.info("=== ELLIPSOID METHOD ===")
        tic = datetime.now()
        
        # Initialize subproblem manager
        self.subproblem_manager.initialize_local()
        
        # Initialize ellipsoid
        self._initialize_ellipsoid()
        
        # Main iteration loop
        iteration = 0
        vals = []
        centers = []
        # Use num_iters if provided, otherwise compute from solver_precision or fall back to max_iterations
        if self.ellipsoid_cfg.num_iters is not None:
            num_iters = self.ellipsoid_cfg.num_iters
        else:
            # Compute iterations using formula: n*(n-1)*log(1/precision) where n is num_features
            num_iters = int(self.n * (self.n - 1) * np.log(1.0 / self.ellipsoid_cfg.solver_precision))
        keep_last_n = min(1000, num_iters)  # Limit memory for large num_iters

        # Track timing breakdown
        timing_breakdown = {
            'pricing': [],
            'gradient_compute': [],
            'ellipsoid_update': [],
            'mpi_broadcast': []
        }

        while iteration < num_iters:
            iteration += 1
            logger.info(f"ELLIPSOID ITERATION {iteration}")
            logger.info(f"THETA: {np.round(self.theta_iter, 4)}")
            
            # Check for constraint violations (non-negativity)
            violated = np.where(self.theta_iter < 0.0)[0]

            direction = None
            if violated.size > 0:
                # Handle constraint violation: use negative gradient of violated constraint
                direction = np.zeros_like(self.theta_iter)
                direction[violated[0]] = -1.0
                obj_value = np.inf
            else:
                # Use objective gradient for productive update
                t_gradient = datetime.now()
                obj_value, gradient = self.compute_obj_and_gradient(self.theta_iter)
                timing_breakdown['gradient_compute'].append((datetime.now() - t_gradient).total_seconds())
                direction = gradient
                vals.append(obj_value)
                centers.append(self.theta_iter.copy())
                
                # Limit memory growth for very long runs
                if len(centers) > keep_last_n:
                    vals = vals[-keep_last_n:]
                    centers = centers[-keep_last_n:]

            # Update ellipsoid
            t_update = datetime.now()
            self._update_ellipsoid(direction)
            timing_breakdown['ellipsoid_update'].append((datetime.now() - t_update).total_seconds())
            
            # Prepare buffer on non-root ranks
            if not self.is_root():
                self.theta_iter = np.empty(self.num_features, dtype=np.float64)
            
            t_bcast = datetime.now()
            self.theta_iter = self.comm_manager.broadcast_array(self.theta_iter, root=0)
            timing_breakdown['mpi_broadcast'].append((datetime.now() - t_bcast).total_seconds())
            
            # Call callback if provided
            if callback and self.is_root() and obj_value != np.inf:
                callback({
                    'iteration': iteration,
                    'theta': self.theta_iter.copy() if self.theta_iter is not None else None,
                    'objective': obj_value,
                    'best_objective': min(vals) if vals else None,
                })
            


        elapsed = (datetime.now() - tic).total_seconds()
        logger.info("Ellipsoid method completed %d iterations in %.2f seconds.", 
                   num_iters, elapsed)
        
        # Return best theta found, not last iteration
        if self.is_root():
            if len(vals) > 0:
                best_idx = np.argmin(vals)
                best_theta = np.array(centers)[best_idx]
                best_obj = vals[best_idx]
                best_iter = best_idx + 1
                logger.info(f"Best objective: {vals[best_idx]:.4f} at iteration {best_idx+1}")
            else:
                # All iterations were constraint violations
                best_theta = self.theta_iter
                best_obj = None
                best_iter = None
        else:
            best_theta = np.empty(self.num_features, dtype=np.float64)
            best_obj = None
            best_iter = None
        
        # Broadcast result to all ranks
        best_theta = self.comm_manager.broadcast_array(best_theta, root=0)
        
        # Store timing statistics
        if self.is_root():
            total_gradient = np.sum(timing_breakdown.get('gradient_compute', [0]))
            total_update = np.sum(timing_breakdown.get('ellipsoid_update', [0]))
            total_mpi = np.sum(timing_breakdown.get('mpi_broadcast', [0]))
            
            self.timing_stats = {
                'total_time': elapsed,
                'num_iterations': num_iters,
                'gradient_time': total_gradient,
                'update_time': total_update,
                'mpi_time': total_mpi,
                'gradient_time_pct': 100 * total_gradient / elapsed if elapsed > 0 else 0,
                'update_time_pct': 100 * total_update / elapsed if elapsed > 0 else 0,
                'mpi_time_pct': 100 * total_mpi / elapsed if elapsed > 0 else 0,
            }
            result = EstimationResult(
                theta_hat=best_theta.copy(),
                converged=True,  # Ellipsoid always runs fixed iterations
                num_iterations=num_iters,
                final_objective=best_obj,
                timing=self.timing_stats,
                iteration_history=None,
                warnings=[] if best_obj is not None else ['All iterations were constraint violations'],
                metadata={'best_iteration': best_iter}
            )
        else:
            self.timing_stats = None
            result = EstimationResult(
                theta_hat=best_theta.copy(),  # best_theta already broadcast to all ranks
                converged=True,
                num_iterations=num_iters,
                final_objective=None,
                timing=None,
                iteration_history=None,
                warnings=[],
                metadata={}
            )
        
        return result

    def _initialize_ellipsoid(self) -> None:
        """Initialize the ellipsoid with starting parameters and matrix."""
        if self.theta_init is not None:
            self.theta_iter = self.theta_init.copy()
        else:
            self.theta_iter = 0.1 * np.ones(self.n)
        self.B_iter = self.ellipsoid_cfg.initial_radius * np.eye(self.n)

    def _update_ellipsoid(self, d: NDArray[np.float64]) -> None:
        """
        Update the ellipsoid using the gradient.
        
        Args:
            d: Direction of update.
        """
        if self.is_root():
            dTBd = d.T @ self.B_iter @ d
            if dTBd <= 0 or not np.isfinite(dTBd):
                logger.warning("Ellipsoid update: dTBd <= 0 or non-finite, skipping update")
                return
            b = (self.B_iter @ d) / np.sqrt(dTBd)
            if not np.all(np.isfinite(b)):
                logger.warning("Ellipsoid update: b is non-finite, skipping update")
                return
            self.theta_iter = self.theta_iter - (1/(self.n+1)) * b
            B_new = self.gamma_1 * self.B_iter - self.gamma_2 * np.outer(b, b)
            if np.all(np.isfinite(B_new)):
                self.B_iter = B_new
            else:
                logger.warning("Ellipsoid update: B_new is non-finite, skipping B update")