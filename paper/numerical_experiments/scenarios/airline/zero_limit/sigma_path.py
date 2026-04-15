"""Zero-limit estimator under misspecification: DGP has errors, estimator doesn't.

Traces bias as sigma -> 0. For each sigma, runs R replications with
fresh errors, estimates with the zero-limit approach (gravity fixed at 1,
no simulation noise), and reports bias/RMSE.
"""

import sys
import json
from pathlib import Path
import numpy as np
import yaml
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from generate_data import (build_geography, build_edges, build_covariates,
                           build_hubs, greedy_demand)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import AirlineGreedySolver, build_zero_limit_covariates_oracle

import combest as ce
from combest.utils import get_logger

logger = get_logger(__name__)

BASE = Path(__file__).resolve().parent
PARENT = BASE.parent
with open(PARENT / 'config.yaml') as f:
    CFG = yaml.safe_load(f)


def run_sigma_path(sigmas=None, n_reps=30, cfg=None):
    if cfg is None:
        cfg = CFG
    if sigmas is None:
        sigmas = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]

    dgp_cfg = cfg['dgp']
    est_cfg = cfg['estimation']

    C = dgp_cfg['C']
    N = dgp_cfg['N']
    pop_log_std = dgp_cfg['pop_log_std']
    hub_pool_frac = dgp_cfg['hub_pool_frac']
    min_hubs = dgp_cfg['min_hubs']
    max_hubs = dgp_cfg['max_hubs']
    fe_mode = dgp_cfg['fe_mode']

    M = C * (C - 1)
    n_p = N + 1  # theta_fc_0..N-1 + theta_gs

    # Fixed geography (same across all sigmas and reps)
    rng_dgp = np.random.default_rng(42)
    _, origin_of, dest_of, _ = build_edges(C)
    locations, dists, populations = build_geography(C, pop_log_std, rng_dgp)
    phi = build_covariates(C, M, origin_of, dest_of, dists, populations, fe_mode)
    hubs = build_hubs(N, C, populations, min_hubs, max_hubs, hub_pool_frac, rng_dgp)

    gravity = phi[:, 0]

    # Fixed true parameters
    theta_rev_true = 1.0
    theta_fc_true = np.array([0.5 + 0.1 * i for i in range(N)])
    theta_gs_true = 2.0
    theta_star = np.concatenate([theta_fc_true, [theta_gs_true]])
    theta_rev_vec = np.array([theta_rev_true])

    logger.info(f"SIGMA PATH: C={C}, M={M}, N={N}")
    logger.info(f"theta_rev={theta_rev_true} (fixed), theta_fc={theta_fc_true}, "
                f"theta_gs={theta_gs_true}")
    logger.info(f"sigmas={sigmas}, n_reps={n_reps}")

    results = {}

    for sigma in sigmas:
        logger.info(f"\n--- sigma = {sigma} ---")
        all_theta_hat = []
        t0 = time.perf_counter()

        for rep in range(n_reps):
            # DGP errors
            if sigma > 0:
                rng_err = np.random.default_rng((42, 55555, rep))
                errors = rng_err.normal(0, sigma, (N, M))
            else:
                errors = np.zeros((N, M))

            # Observed bundles (with errors)
            obs_bundles = np.zeros((N, M), dtype=bool)
            for i in range(N):
                obs_bundles[i] = greedy_demand(
                    phi, theta_rev_vec, theta_fc_true[i], theta_gs_true,
                    hubs[i], origin_of, errors[i], M)

            # Zero-limit estimation (no simulation noise)
            model = ce.Model()
            input_data = {
                'id_data': {
                    'obs_bundles': obs_bundles,
                    'hubs': [list(h) for h in hubs],
                },
                'item_data': {
                    'phi': phi, 'origin_of': origin_of, 'N_firms': N,
                },
            }

            lbs = {i: 0 for i in range(N + 1)}
            model_cfg = {
                'dimensions': {
                    'n_obs': N, 'n_items': M,
                    'n_covariates': n_p, 'n_simulations': 1,
                },
                'subproblem': {
                    'gurobi_params': {
                        'TimeLimit': est_cfg['gurobi_timeout'],
                        'OutputFlag': 0,
                    },
                },
                'row_generation': {
                    'max_iters': est_cfg['max_iters'],
                    'tolerance': est_cfg['tolerance'],
                    'theta_bounds': {'lb': -20, 'ub': 20, 'lbs': lbs},
                },
            }
            model.load_config(model_cfg)
            model.data.load_and_distribute_input_data(input_data)

            ld = model.data.local_data
            ld.id_data['hubs'] = [set(h) for h in ld.id_data['hubs']]
            ld.item_data['obs_ids'] = model.comm_manager.obs_ids

            # Deterministic error: gravity * 1 (no noise)
            n_local = model.comm_manager.num_local_agent
            local_errors = np.tile(gravity, (n_local, 1))
            model.features.local_modular_errors = local_errors
            model.features._error_oracle = lambda b, ids: (
                model.features.local_modular_errors[ids] * b).sum(-1)
            model.features._error_oracle_takes_data = False

            model.features.set_covariates_oracle(
                build_zero_limit_covariates_oracle(N))

            model.subproblems.load_solver(AirlineGreedySolver)
            model.subproblems.initialize_solver()

            row_gen = model.point_estimation.n_slack
            result = row_gen.solve(initialize_solver=False,
                                   initialize_master=True, verbose=False)

            if result is not None:
                all_theta_hat.append(result.theta_hat.copy())

        t_elapsed = time.perf_counter() - t0
        estimates = np.array(all_theta_hat)
        n_ok = len(estimates)

        deviations = estimates - theta_star
        bias = deviations.mean(axis=0)
        std = deviations.std(axis=0, ddof=1) if n_ok > 1 else np.zeros(n_p)
        rmse = np.sqrt((deviations ** 2).mean(axis=0))
        mae_pct = (np.abs(deviations) / np.abs(theta_star) * 100).mean(axis=0)

        fc_names = [f"theta_fc_{i}" for i in range(N)]
        param_names = fc_names + ["theta_gs"]

        # Summaries
        fc_sl = slice(0, N)
        fc_bias = np.abs(bias[fc_sl]).mean()
        fc_rmse = rmse[fc_sl].mean()
        fc_mae = mae_pct[fc_sl].mean()
        gs_bias = abs(bias[-1])
        gs_rmse = rmse[-1]
        gs_mae = mae_pct[-1]

        results[sigma] = {
            'n_ok': n_ok,
            'bias': bias.tolist(),
            'std': std.tolist(),
            'rmse': rmse.tolist(),
            'mae_pct': mae_pct.tolist(),
            'fc_avg_abs_bias': float(fc_bias),
            'fc_avg_rmse': float(fc_rmse),
            'fc_avg_mae_pct': float(fc_mae),
            'gs_abs_bias': float(gs_bias),
            'gs_rmse': float(gs_rmse),
            'gs_mae_pct': float(gs_mae),
            'time_s': round(t_elapsed, 1),
        }

        print(f"  sigma={sigma:>5.2f}: {n_ok}/{n_reps} reps, "
              f"avg|bias(fc)|={fc_bias:.4f}, RMSE(fc)={fc_rmse:.4f}, "
              f"|bias(gs)|={gs_bias:.4f}, RMSE(gs)={gs_rmse:.4f}, "
              f"{t_elapsed:.0f}s")

    # Summary table
    print(f"\n{'='*80}")
    print(f"  SIGMA PATH: zero-limit estimator under misspecification")
    print(f"  C={C}, M={M}, N={N}, {n_reps} reps per sigma")
    print(f"{'='*80}")
    print(f"  {'sigma':>7} | {'avg|bias(fc)|':>14} {'RMSE(fc)':>10} {'MAE%(fc)':>10} | "
          f"{'|bias(gs)|':>11} {'RMSE(gs)':>10} {'MAE%(gs)':>10}")
    print(f"  {'-'*78}")
    for sigma in sigmas:
        r = results[sigma]
        print(f"  {sigma:>7.2f} | {r['fc_avg_abs_bias']:>14.4f} {r['fc_avg_rmse']:>10.4f} "
              f"{r['fc_avg_mae_pct']:>9.1f}% | "
              f"{r['gs_abs_bias']:>11.4f} {r['gs_rmse']:>10.4f} "
              f"{r['gs_mae_pct']:>9.1f}%")
    print()

    out_path = BASE / 'sigma_path.json'
    with open(out_path, 'w') as f:
        json.dump({
            'theta_true': theta_star.tolist(),
            'theta_rev_fixed': theta_rev_true,
            'sigmas': sigmas,
            'results': {str(s): v for s, v in results.items()},
            'config': {'C': C, 'M': M, 'N': N, 'n_reps': n_reps},
        }, f, indent=2)
    print(f"  Saved to {out_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-reps', type=int, default=30)
    args = parser.parse_args()

    run_sigma_path(
        sigmas=[0.0, 0.1, 0.25, 0.5, 1.0, 2.0],
        n_reps=args.n_reps,
    )
