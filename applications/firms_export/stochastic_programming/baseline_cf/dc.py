import time
import numpy as np
from combest.estimation.result import RowGenerationEstimationResult
from combest.utils import get_logger

logger = get_logger(__name__)


class DCSolver:

    def __init__(self, row_gen_solver, subproblem_solver):
        self.row_gen = row_gen_solver
        self.solver = subproblem_solver

    def solve(self, theta0, max_dc_iters=20, tol=1e-6, verbose=False,
              iteration_callback=None):
        comm = self.row_gen.comm_manager
        n = len(theta0)
        theta_k = comm.Bcast(np.asarray(theta0, dtype=np.float64))
        t0 = time.perf_counter()

        if verbose and comm.is_root():
            logger.info("")
            logger.info(" DC ALGORITHM")
            logger.info(f" max_dc_iters={max_dc_iters}  tol={tol:.0e}")
            logger.info(f" theta0 = {theta_k}")
            logger.info("")

        result = None
        converged = False
        final_obj = None

        for k in range(max_dc_iters):
            if verbose and comm.is_root():
                logger.info(f" DC iter {k+1}/{max_dc_iters}")
                logger.info(f"   theta_k = {theta_k}")

            self.solver.solve_Q(theta_k)

            init_master = (k == 0)
            result = self.row_gen.solve(
                initialize_solver=False,
                initialize_master=init_master,
                iteration_callback=iteration_callback,
                verbose=verbose)

            theta_next = np.empty(n, dtype=np.float64)
            if comm.is_root():
                theta_next[:] = result.theta_hat
                final_obj = result.final_objective
            theta_next = comm.Bcast(theta_next)

            rel_change = np.linalg.norm(theta_next - theta_k) / (
                1.0 + np.linalg.norm(theta_k))

            if verbose and comm.is_root():
                logger.info(f"   theta_next = {theta_next}")
                logger.info(f"   rel_change = {rel_change:.2e}")
                logger.info("")

            if rel_change < tol:
                converged = True
                break

            theta_k = theta_next.copy()

        total_time = time.perf_counter() - t0

        if comm.is_root():
            return RowGenerationEstimationResult(
                theta_hat=theta_next,
                converged=converged,
                num_iterations=k + 1,
                final_objective=final_obj,
                total_time=total_time,
            )
        return None
