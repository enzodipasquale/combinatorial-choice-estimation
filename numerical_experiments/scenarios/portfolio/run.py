"""Zero-noise estimator for the portfolio-choice scenario.

DGP has NO idiosyncratic errors: choices are deterministic given theta*.
beta_0 is pinned to 1.0 for scale normalization. The remaining four
parameters (beta_1, beta_2, gamma, kappa) are free.

With no misspecification, the estimator should recover theta* exactly.
"""

import time
import json
from pathlib import Path
import numpy as np
import combest as ce
from combest.utils import get_logger

from dgp import generate_dgp
from solver import PortfolioSolver
from oracle import covariates_oracle, error_oracle

logger = get_logger(__name__)
BASE = Path(__file__).resolve().parent


def run(seed=42, M=15):
    model = ce.Model()
    is_root = model.is_root()

    t0 = time.perf_counter()

    # --- DGP ---
    X, X_agents, Sigma, theta_star = generate_dgp(seed=seed, M=M)
    M, K = X.shape
    N = X_agents.shape[0]
    n_params = len(theta_star)  # 5: [beta_0, beta_1, beta_2, gamma, kappa]

    if is_root:
        logger.info("PORTFOLIO ZERO-NOISE ESTIMATOR")
        logger.info(f"N={N}, M={M}, K={K}, n_params={n_params}")
        logger.info(f"theta* = {theta_star}")

    # --- Solve DGP at theta* to get observed bundles ---
    # We solve all N agents serially on root, then distribute.
    if is_root:
        import gurobipy as gp
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.start()

        beta = theta_star[:K]
        gamma_val = theta_star[K]
        kappa_val = theta_star[K + 1]

        obs_bundles = np.zeros((N, 2 * M), dtype=float)
        for i in range(N):
            X_i = X_agents[i]
            mu_i = X_i @ beta
            m = gp.Model(env=env)
            m.setParam('OutputFlag', 0)
            s = m.addMVar(M, vtype=gp.GRB.BINARY, name='s')
            w = m.addMVar(M, lb=0.0, ub=1.0, name='w')
            m.addConstr(w.sum() == 1)
            m.addConstr(w <= s)
            m.setObjective(
                mu_i @ w + w @ (-0.5 * gamma_val * Sigma) @ w
                - kappa_val * s.sum(),
                gp.GRB.MAXIMIZE
            )
            m.optimize()
            obs_bundles[i, :M] = np.round(np.array(s.X)).astype(float)
            obs_bundles[i, M:] = np.array(w.X)

        sizes = obs_bundles[:, :M].sum(axis=1)
        logger.info(f"|s| distribution: mean={sizes.mean():.1f}, std={sizes.std():.1f}, "
                     f"min={int(sizes.min())}, max={int(sizes.max())}")
    else:
        obs_bundles = None

    # --- Configure model ---
    # Pin beta_0 = 1.0 via lbs[0] = ubs[0] = 1.0
    model_cfg = {
        'dimensions': {
            'n_obs': N,
            'n_items': 2 * M,  # bundles are (s, w) of length 2M
            'n_covariates': n_params,
            'n_simulations': 1,
        },
        'subproblem': {
            'gurobi_params': {'TimeLimit': 30, 'OutputFlag': 0},
        },
        'row_generation': {
            'max_iters': 200,
            'tolerance': 1e-8,
            'theta_bounds': {
                'lb': -50, 'ub': 50,
                'lbs': {0: 1.0},  # pin beta_0 = 1.0
                'ubs': {0: 1.0},
            },
        },
    }
    model.load_config(model_cfg)

    # --- Distribute data ---
    if is_root:
        input_data = {
            'id_data': {
                'obs_bundles': obs_bundles,
                'X_agents': X_agents,
            },
            'item_data': {
                'Sigma': Sigma,
            },
        }
    else:
        input_data = None

    model.data.load_and_distribute_input_data(input_data)

    # --- Set oracles ---
    model.features.set_covariates_oracle(covariates_oracle)
    model.features.set_error_oracle(error_oracle)

    # --- Load and initialize solver ---
    model.subproblems.load_solver(PortfolioSolver)
    model.subproblems.initialize_solver()

    # --- Run row generation ---
    row_gen = model.point_estimation.n_slack
    result = row_gen.solve(initialize_solver=False, initialize_master=True, verbose=True)

    t_total = time.perf_counter() - t0

    # --- Diagnostics ---
    if is_root and result is not None:
        theta_hat = result.theta_hat
        # True ratios (with beta_0 = 1.0 as numeraire)
        true_ratios = theta_star[1:] / theta_star[0]
        est_ratios = theta_hat[1:] / theta_hat[0]
        ratio_names = ['beta_1/beta_0', 'beta_2/beta_0', 'gamma/beta_0', 'kappa/beta_0']

        logger.info("")
        logger.info("=" * 70)
        logger.info(f"  PORTFOLIO ZERO-NOISE RESULTS (N={N}, M={M})")
        logger.info(f"  beta_0 = {theta_hat[0]:.6f} (pinned)")
        logger.info("=" * 70)
        logger.info(f"  {'Ratio':<16} {'True':>10} {'Est':>10} {'|Err|':>10} {'Err%':>8}")
        logger.info(f"  {'-'*56}")
        for j, name in enumerate(ratio_names):
            err = abs(est_ratios[j] - true_ratios[j])
            err_pct = err / abs(true_ratios[j]) * 100 if true_ratios[j] != 0 else float('inf')
            logger.info(f"  {name:<16} {true_ratios[j]:>10.6f} {est_ratios[j]:>10.6f} "
                         f"{err:>10.2e} {err_pct:>7.1f}%")

        logger.info(f"")
        logger.info(f"  theta_hat = {theta_hat}")
        logger.info(f"  Converged: {result.converged}")
        logger.info(f"  Iterations: {result.num_iterations}")
        logger.info(f"  Final objective: {result.final_objective:.6e}")
        logger.info(f"  Runtime: {t_total:.1f}s")

        out = {
            'theta_true': theta_star.tolist(),
            'theta_hat': theta_hat.tolist(),
            'true_ratios': dict(zip(ratio_names, true_ratios.tolist())),
            'est_ratios': dict(zip(ratio_names, est_ratios.tolist())),
            'converged': result.converged,
            'iterations': result.num_iterations,
            'final_objective': result.final_objective,
            'runtime_s': round(t_total, 2),
        }
        out_path = BASE / 'result.json'
        with open(out_path, 'w') as f:
            json.dump(out, f, indent=2)
        logger.info(f"  Saved to {out_path}")

    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--M', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    run(seed=args.seed, M=args.M)
