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
        subproblem_manager
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
        """
        super().__init__(
            comm_manager=comm_manager,
            dimensions_cfg=dimensions_cfg,
            data_manager=data_manager,
            feature_manager=feature_manager,
            subproblem_manager=subproblem_manager
        )
        
        self.ellipsoid_cfg = ellipsoid_cfg
        
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

    def solve(self) -> np.ndarray:
        """
        Run the ellipsoid method to estimate model parameters.

        Returns:
            np.ndarray: Estimated parameter vector.
        """
        tic = datetime.now()
        
        # Initialize subproblem manager
        self.subproblem_manager.initialize_local()
        
        # Initialize ellipsoid
        self._initialize_ellipsoid()
        
        # Main iteration loop
        iteration = 0
        vals = []
        centers = []

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
                centers.append(self.theta_iter)
                # logger.info(f"Objective: {obj_value:.4f}, Gradient norm: {np.linalg.norm(direction):.4f}")

            # Update ellipsoid
            self._update_ellipsoid(direction)
            self.theta_iter = self.comm_manager.broadcast_from_root(self.theta_iter, root=0)
            


        elapsed = (datetime.now() - tic).total_seconds()
        logger.info("Ellipsoid method completed %d iterations in %.2f seconds.", 
                   self.ellipsoid_cfg.num_iters, elapsed)
        if self.is_root():
            best_theta = np.array(centers)[np.argmin(vals)]
            best_obj = np.min(vals)
            # logger.info(f"Best theta: {best_theta}, Best objective: {best_obj}")
        return self.theta_iter

    def _initialize_ellipsoid(self):
        """Initialize the ellipsoid with starting parameters and matrix."""
        self.theta_iter = 0.1 * np.ones(self.n)
        self.B_iter = self.ellipsoid_cfg.initial_radius * np.eye(self.n)

    def _update_ellipsoid(self, d: np.ndarray):
        """
        Update the ellipsoid using the gradient.
        
        Args:
            e: direction of update
        """
        if self.is_root():
                # print("direction", d)
            # BTd = self.B_iter.T @ d
            # dTBBTd = d.T @ self.B_iter @  BTd
            # logger.info(f"dTBBTd: {dTBBTd}")
            # p = BTd / np.sqrt(dTBBTd)
            # logger.info(f"p: {p}")
            # # print("update",-(1/(self.n+1)) * self.B_iter @ p)
            # self.theta_iter = self.theta_iter - (1/(self.n+1)) * self.B_iter @ p
            # self.B_iter = self.alpha * self.B_iter - self.gamma * (self.B_iter @ np.outer(p, p))

            b = (self.B_iter @ d) / np.sqrt(d.T @ self.B_iter @ d)
            self.theta_iter = self.theta_iter - (1/(self.n+1)) * b
            self.B_iter = self.gamma_1 * self.B_iter - self.gamma_2 *  np.outer(b, b)



   