import time
import numpy as np
import gurobipy as gp
from combest.estimation.result import RowGenerationEstimationResult


TAU_INIT = 1.0              # initial proximity weight
GAMMA = 0.1                 # serious step acceptance threshold
GAMMA_UP = 0.9              # strong descent threshold for tau decrease
GAMMA_TILDE = 0.9           # proximity control threshold for tau increase
C = 1e-5                    # downshift safeguard
TOL = 1e-8                  # stopping tolerance on relative step size
TOL_G = 1e-8                # stopping tolerance on aggregate subgradient norm
TAU_MAX = 1e8               # upper bound on tau
ZETA = 1e-4                  # positive definiteness bound (Step 11)
MAX_ITERS = 200             # maximum oracle calls
NULL_CLOSE_LIMIT = 10        # consecutive small null steps before stopping


class BundleSolver:

    def __init__(self, pt_estimation_manager):
        self.pt = pt_estimation_manager

    def solve(self, theta0, tau=TAU_INIT, gamma=GAMMA, Gamma=GAMMA_UP,
              gamma_tilde=GAMMA_TILDE, c=C, zeta=ZETA, tol=TOL,
              tol_g=TOL_G, max_iters=MAX_ITERS, prune=True,
              verbose=False):
        """Schramm-Zowe proximal bundle method (Algorithm 1,
        Kuchlbauer-Liers-Stingl).
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
        t0 = time.perf_counter()

        _col_hdr = (" Iter  S/N     f_trial       f_hat"
                    "        tau       rho    rel_step"
                    "      |g*|")
        _col_sep = " " + "─" * 78

        if verbose and comm.is_root():
            print()
            print(" BUNDLE METHOD")
            print(f" tau={tau:.2f}  gamma={gamma}  tol={tol:.0e}"
                  f"  max_iters={max_iters}")
            print(f" f(theta0) = {f_hat:.6f}")
            print()
            print(_col_hdr)
            print(_col_sep)

        for k in range(max_iters):
            # Solve QP subproblem (root only)
            theta_next = np.empty(n, dtype=np.float64)
            phi_next = 0.0
            g_star_norm = 0.0
            lambdas = None
            if comm.is_root():
                theta_next[:], phi_next, lambdas, s_values = self._solve_qp(
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
                    theta_hat = theta_next.copy()
                    f_hat = f_next
                    null_close_count = 0

                    # -- Bundle pruning (Schramm-Zowe reset) --
                    if prune:
                        # lambdas are QP duals for the n_old planes
                        # (before the new exactness plane was appended)
                        n_old = len(lambdas)
                        PRUNE_THRESH = 1e-6
                        active_idx = [i for i in range(n_old)
                                      if lambdas[i] > PRUNE_THRESH]
                        inactive_idx = [i for i in range(n_old)
                                        if lambdas[i] <= PRUNE_THRESH]

                        new_thetas = [thetas[i] for i in active_idx]
                        new_fs = [fs[i] for i in active_idx]
                        new_gs = [gs[i] for i in active_idx]

                        # Aggregate inactive planes into one synthetic plane
                        if inactive_idx:
                            lam_inact = np.array([lambdas[i]
                                                  for i in inactive_idx])
                            lam_sum = lam_inact.sum()
                            if lam_sum > 1e-12:
                                w = lam_inact / lam_sum
                                g_agg = sum(
                                    w[j] * gs[inactive_idx[j]]
                                    for j in range(len(inactive_idx)))
                                # Downshifted linearization at new theta_hat
                                f_agg = sum(
                                    w[j] * (fs[inactive_idx[j]]
                                            + gs[inactive_idx[j]]
                                            @ (theta_hat - thetas[inactive_idx[j]])
                                            - s_values[inactive_idx[j]])
                                    for j in range(len(inactive_idx)))
                                new_thetas.append(theta_hat.copy())
                                new_fs.append(f_agg)
                                new_gs.append(g_agg)

                        # Exactness plane at new stability center
                        new_thetas.append(theta_hat.copy())
                        new_fs.append(f_hat)
                        new_gs.append(g_next.copy())

                        thetas[:] = new_thetas
                        fs[:] = new_fs
                        gs[:] = new_gs

                    # Decrease tau on strong descent (Step 9)
                    if rho >= Gamma:
                        tau = tau / 2.0
                    # Step 11: enforce positive definiteness bound
                    tau = max(tau, zeta)

                    # Stopping criterion 1: small serious step
                    if rel_step < tol:
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

                    # Increase tau if proximity control triggers (Step 17)
                    if rho_tilde >= gamma_tilde:
                        tau = min(tau * 2.0, TAU_MAX)

                    # Stopping criterion 2: successive small null steps
                    if rel_step < tol:
                        null_close_count += 1
                    else:
                        null_close_count = 0

                    if null_close_count >= NULL_CLOSE_LIMIT:
                        converged = True
                        stop = True

                # Stopping criterion 4 (Remark 4.17): small aggregate
                # subgradient — applies to both serious and null steps
                if g_star_norm < tol_g:
                    converged = True
                    stop = True


                if verbose:
                    tag = "S" if serious else "N"
                    if (k + 1) % 50 == 0:
                        print(_col_hdr)
                        print(_col_sep)
                    print(f" {k+1:4d}   {tag}"
                          f"  {f_next:12.4f}  {f_hat:12.4f}"
                          f"  {tau:9.2e}  {rho:7.3f}"
                          f"  {rel_step:9.2e}  {g_star_norm:9.2e}")

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

        s_values = []
        for i in range(len(thetas)):
            g_i, f_i, th_i = gs[i], fs[i], thetas[i]
            diff = theta_hat - th_i
            s_i = max(0.0, f_i + g_i @ diff - f_hat) + c * (diff @ diff)
            s_values.append(s_i)
            m.addConstr(t >= f_i + g_i @ (z - th_i) - s_i)

        m.optimize()
        lambdas = np.array([constr.Pi for constr in m.getConstrs()])
        assert np.all(lambdas >= -1e-10), f"Negative duals: {lambdas}"
        return np.array(z.X), t.X, lambdas, s_values
