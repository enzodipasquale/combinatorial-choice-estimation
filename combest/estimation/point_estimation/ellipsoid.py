import time
import math
import numpy as np
from combest.estimation.result import RowGenerationEstimationResult


class EllipsoidSolver:

    def __init__(self, pt_estimation_manager):
        self.pt = pt_estimation_manager

    def solve(self, num_iters=None, precision=1e-4, initial_radius=100.0, verbose=False):
        pt = self.pt
        comm = pt.comm_manager
        n = pt.config.dimensions.n_covariates
        weights = pt.data_manager.local_obs_quantity
        pt.subproblem_manager.initialize_solver()

        if num_iters is None:
            num_iters = int(n * (n - 1) * math.log(1.0 / precision))

        gamma_1 = (n**2 / (n**2 - 1)) ** 0.5
        gamma_2 = gamma_1 * 2 / (n + 1)

        theta = np.zeros(n, dtype=np.float64)
        B = initial_radius**2 * np.eye(n)
        best_obj, best_theta = np.inf, theta.copy()

        t0 = time.perf_counter()
        for it in range(num_iters):
            theta = comm.Bcast(theta)
            obj, grad = pt.compute_polyhedral_obj_and_grad_at_root(theta, weights)

            if comm.is_root():
                obj = float(obj)
                if obj < best_obj:
                    best_obj, best_theta = obj, theta.copy()
                if verbose:
                    print(f"  ellipsoid {it+1}/{num_iters}  obj={obj:.6f}  best={best_obj:.6f}")

                dBd = grad @ B @ grad
                if dBd > 0:
                    b = (B @ grad) / np.sqrt(dBd)
                    theta = theta - b / (n + 1)
                    B = gamma_1 * B - gamma_2 * np.outer(b, b)

        total_time = time.perf_counter() - t0
        best_theta = comm.Bcast(best_theta)

        return RowGenerationEstimationResult(
            theta_hat=best_theta,
            converged=True,
            num_iterations=num_iters,
            final_objective=best_obj if comm.is_root() else None,
            total_time=total_time,
        )
