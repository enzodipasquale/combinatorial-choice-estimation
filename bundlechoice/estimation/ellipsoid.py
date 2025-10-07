"""
Ellipsoid method solver for modular bundle choice estimation (v2).
This module implements the ellipsoid method for parameter estimation.
"""
import numpy as np
from datetime import datetime
from typing import Optional, Tuple
from .base import BaseEstimationSolver
from bundlechoice.utils import get_logger

logger = get_logger(__name__)


class EllipsoidSolver(BaseEstimationSolver):
    """
    Implements the ellipsoid method for parameter estimation in modular bundle choice models.

    This solver uses the ellipsoid method to iteratively refine parameter estimates
    based on gradient information and constraint violations.
    """
    
    def __init__(
        self,
        comm_manager,
        dimensions_cfg,
        ellipsoid_cfg,  
        data_manager,
        feature_manager,
        subproblem_manager,
        theta_init=None
    ):
        """
        Initialize the EllipsoidSolver.

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
        
        # Ellipsoid update coefficients
        n = self.num_features
        self.n = n
        self.alpha = (n**2 / (n**2 - 1))**(1/4)
        self.gamma = self.alpha * ((n-1)/(n+1))**(1/2)

        self.gamma_1 = (n**2 / (n**2 - 1))**(1/2)
        self.gamma_2 = self.gamma_1 * ((2 / (n + 1)))

    def solve(self, callback=None) -> np.ndarray:
        """
        Run the ellipsoid method to estimate model parameters.

        Args:
            callback: Optional callback function called after each iteration with info dict

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
        keep_last_n = min(1000, self.ellipsoid_cfg.num_iters)  # Limit memory for large num_iters

        while iteration < self.ellipsoid_cfg.num_iters:
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
                obj_value, gradient = self.compute_obj_and_gradient(self.theta_iter)
                direction = gradient
                vals.append(obj_value)
                centers.append(self.theta_iter.copy())
                
                # Limit memory growth for very long runs
                if len(centers) > keep_last_n:
                    vals = vals[-keep_last_n:]
                    centers = centers[-keep_last_n:]

            # Update ellipsoid
            self._update_ellipsoid(direction)
            
            # Prepare buffer on non-root ranks
            if not self.is_root():
                self.theta_iter = np.empty(self.num_features, dtype=np.float64)
            
            self.theta_iter = self.comm_manager.broadcast_array(self.theta_iter, root=0)
            
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
                   self.ellipsoid_cfg.num_iters, elapsed)
        
        # Return best theta found, not last iteration
        if self.is_root():
            if len(vals) > 0:
                best_idx = np.argmin(vals)
                best_theta = np.array(centers)[best_idx]
                logger.info(f"Best objective: {vals[best_idx]:.4f} at iteration {best_idx+1}")
            else:
                # All iterations were constraint violations
                best_theta = self.theta_iter
        else:
            best_theta = np.empty(self.num_features, dtype=np.float64)
        
        # Broadcast result to all ranks
        best_theta = self.comm_manager.broadcast_array(best_theta, root=0)
        return best_theta

    def _initialize_ellipsoid(self):
        """Initialize the ellipsoid with starting parameters and matrix."""
        if self.theta_init is not None:
            self.theta_iter = self.theta_init.copy()
        else:
            self.theta_iter = 0.1 * np.ones(self.n)
        self.B_iter = self.ellipsoid_cfg.initial_radius * np.eye(self.n)

    def _update_ellipsoid(self, d: np.ndarray):
        """
        Update the ellipsoid using the gradient.
        
        Args:
            e: direction of update
        """
        if self.is_root():
            b = (self.B_iter @ d) / np.sqrt(d.T @ self.B_iter @ d)
            self.theta_iter = self.theta_iter - (1/(self.n+1)) * b
            self.B_iter = self.gamma_1 * self.B_iter - self.gamma_2 *  np.outer(b, b)



   