import time
import numpy as np
import gurobipy as gp
from combest.estimation.result import RowGenerationEstimationResult


class BundleSolver:

    def __init__(self, pt_estimation_manager):
        self.pt = pt_estimation_manager

    def solve(self, theta0, tau=1.0, gamma=0.1, Gamma=0.9, gamma_tilde=0.9,
              c=1e-5, tol=1e-5, tol_f=1e-6, tol_g=1e-8, max_iters=200,
              max_bundle_size=50, verbose=False):
        """Schramm-Zowe proximal bundle method (Algorithm 1,
        Kuchlbauer-Liers-Stingl).

        Parameters
        ----------
        theta0 : array_like   Starting point.
        tau : float            Initial proximity weight.
        gamma : float          Serious step acceptance threshold (0 < gamma < Gamma).
        Gamma : float          Strong descent threshold for tau decrease (gamma < Gamma < 1).
        gamma_tilde : float    Proximity control threshold for tau increase (gamma < gamma_tilde < 1).
        c : float              Downshift safeguard.
        tol : float            Stopping tolerance on relative step size.
        tol_f : float          Stopping tolerance on relative function decrease.
        tol_g : float          Stopping tolerance on aggregate subgradient norm
                               (Remark 4.17).
        max_iters : int        Maximum oracle calls.
        max_bundle_size : int  Prune bundle to this size.
        verbose : bool         Print iteration log.
        """
        pt = self.pt
        comm = pt.comm_manager
        n = pt.config.dimensions.n_covariates
        weights = pt.data_manager.local_obs_quantity

        # Initialise: broadcast theta0, evaluate oracle
        theta_hat = comm.Bcast(np.asarray(theta0, dtype=np.float64))
        f_hat, g_hat = pt.compute_nonlinear_obj_and_grad_at_root(
            theta_hat, weights)

        thetas = [theta_hat.copy()]
        fs = [f_hat]
        gs = [g_hat.copy() if g_hat is not None else None]

        converged = False
        null_close_count = 0
        flat_count = 0           # consecutive serious steps with tiny f decrease
        t0 = time.perf_counter()

        if verbose and comm.is_root():
            print(f"  bundle init  f_hat={f_hat:.6f}")

        for k in range(max_iters):
            # Prune bundle before QP (keep most recent planes)
            if comm.is_root() and len(thetas) > max_bundle_size:
                thetas = thetas[-max_bundle_size:]
                fs = fs[-max_bundle_size:]
                gs = gs[-max_bundle_size:]

            # Solve QP subproblem (root only)
            theta_next = np.empty(n, dtype=np.float64)
            phi_next = 0.0
            g_star_norm = 0.0
            if comm.is_root():
                theta_next[:], phi_next = self._solve_qp(
                    thetas, fs, gs, theta_hat, f_hat, tau, c, n)
                # Aggregate subgradient: g* = tau * (theta_hat - theta_next)
                # Must use the tau that was passed to the QP, before any update
                g_star_norm = tau * np.linalg.norm(theta_hat - theta_next)
            theta_next = comm.Bcast(theta_next)

            # Oracle at trial point (all ranks participate)
            f_next, g_next = pt.compute_nonlinear_obj_and_grad_at_root(
                theta_next, weights)

            # Append new cutting plane to bundle
            thetas.append(theta_next.copy())
            fs.append(f_next)
            gs.append(g_next.copy() if g_next is not None else None)

            # Acceptance test and parameter update (root only)
            stop = False
            if comm.is_root():
                rel_step = np.linalg.norm(theta_next - theta_hat) / (
                    1.0 + np.linalg.norm(theta_hat))

                denom = f_hat - phi_next
                rho = (f_hat - f_next) / denom if denom > 1e-15 else 0.0

                serious = rho >= gamma
                if serious:
                    # ---- Serious step ----
                    f_hat_old = f_hat
                    theta_hat = theta_next.copy()
                    f_hat = f_next
                    null_close_count = 0

                    # Decrease tau on strong descent (Step 9)
                    if rho >= Gamma:
                        tau = max(tau / 2.0, 1e-3)

                    # Stopping criterion 1: small serious step
                    if rel_step < tol:
                        converged = True
                        stop = True

                    # Stopping criterion 3: flat objective
                    rel_decrease = (f_hat_old - f_hat) / (
                        1.0 + abs(f_hat_old))
                    if rel_decrease < tol_f:
                        flat_count += 1
                    else:
                        flat_count = 0
                    if flat_count >= 5:
                        converged = True
                        stop = True
                else:
                    # ---- Null step ----
                    # Proximity control (Steps 15-17): compute rho_tilde
                    # s_k = downshift for the new cutting plane at theta_next
                    diff_k = theta_hat - theta_next
                    s_k = (max(0.0, f_next + g_next @ diff_k - f_hat)
                           + c * np.dot(diff_k, diff_k))
                    rho_tilde = ((f_hat - (f_next - s_k)) / denom
                                 if denom > 1e-15 else 0.0)

                    # Only increase tau if rho_tilde passes threshold
                    if rho_tilde >= gamma_tilde:
                        tau = min(tau * 2.0, 1e8)

                    # Stopping criterion 2: 3 successive small null steps
                    if rel_step < tol:
                        null_close_count += 1
                    else:
                        null_close_count = 0

                    if null_close_count >= 3:
                        converged = True
                        stop = True

                # Stopping criterion 4 (Remark 4.17): small aggregate
                # subgradient — applies to both serious and null steps
                if g_star_norm < tol_g:
                    converged = True
                    stop = True

                if verbose:
                    tag = "S" if serious else "N"
                    extra = ""
                    if not serious:
                        extra = f"  rho~={rho_tilde:.3f}"
                    else:
                        extra = f"  df={rel_decrease:.1e}  flat={flat_count}"
                    print(f"  bundle {k+1:>4d} [{tag}]  f={f_next:.6f}  "
                          f"f_hat={f_hat:.6f}  tau={tau:.2e}  "
                          f"rho={rho:.3f}  step={rel_step:.2e}"
                          f"  |g*|={g_star_norm:.2e}{extra}")

            # Broadcast decisions to all ranks
            stop = bool(comm.Bcast(np.array(stop)))
            theta_hat = comm.Bcast(theta_hat)
            f_hat = float(comm.Bcast(np.array(
                float(f_hat) if comm.is_root() else 0.0)))

            if stop:
                break

        total_time = time.perf_counter() - t0
        theta_hat = comm.Bcast(theta_hat)

        return RowGenerationEstimationResult(
            theta_hat=theta_hat,
            converged=converged,
            num_iterations=k + 1,
            final_objective=float(f_hat) if comm.is_root() else None,
            total_time=total_time,
        )

    @staticmethod
    def _solve_qp(thetas, fs, gs, theta_hat, f_hat, tau, c, n):
        """Solve the QP subproblem:
            min  t + (tau/2)||z - theta_hat||^2
            s.t. t >= f_i + g_i'(z - theta_i) - s_i   for each i
        where s_i = max(0, f_i + g_i'(theta_hat - theta_i) - f_hat)
                    + c||theta_i - theta_hat||^2
        """
        m = gp.Model()
        m.setParam("OutputFlag", 0)
        z = m.addMVar(n, lb=-gp.GRB.INFINITY, name="z")
        t = m.addVar(lb=-gp.GRB.INFINITY, name="t")
        m.setObjective(
            t + (tau / 2) * (z - theta_hat) @ (z - theta_hat),
            gp.GRB.MINIMIZE)

        for i in range(len(thetas)):
            g_i, f_i, th_i = gs[i], fs[i], thetas[i]
            diff = theta_hat - th_i
            s_i = max(0.0, f_i + g_i @ diff - f_hat) + c * (diff @ diff)
            m.addConstr(t >= f_i + g_i @ (z - th_i) - s_i)

        m.optimize()
        return np.array(z.X), t.X
