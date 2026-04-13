"""Monte Carlo: variance of the estimator over geography + firms.

Fixes theta* and draws fresh geography, populations, hubs, and errors
each replication. Measures the distribution of theta_hat around the
fixed truth — capturing the full DGP variance, not just error variance.
"""

import sys
import json
from pathlib import Path
import numpy as np
import yaml
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_data import (build_geography, build_edges, build_covariates,
                           build_hubs, greedy_demand, n_shared_covariates, n_params)
from oracle import make_find_best_item, build_covariates_oracle

import combest as ce
from combest.subproblems.registry.greedy import GreedySolver
from combest.utils import get_logger

logger = get_logger(__name__)

BASE = Path(__file__).resolve().parent
with open(BASE / 'config.yaml') as f:
    CFG = yaml.safe_load(f)


class AirlineGreedySolver(GreedySolver):
    find_best_item = staticmethod(make_find_best_item())


def run_monte_carlo(n_reps=30, base_cfg=None, theta_star=None):
    if base_cfg is None:
        base_cfg = CFG

    dgp_cfg = base_cfg['dgp']
    est_cfg = base_cfg['estimation']

    C = dgp_cfg['C']
    N = dgp_cfg['N']
    fe_mode = dgp_cfg['fe_mode']
    sigma = dgp_cfg['sigma']
    pop_log_std = dgp_cfg['pop_log_std']
    hub_pool_frac = dgp_cfg['hub_pool_frac']
    min_hubs = dgp_cfg['min_hubs']
    max_hubs = dgp_cfg['max_hubs']

    M = C * (C - 1)
    n_shared = n_shared_covariates(C, fe_mode)
    n_p = n_params(C, N, fe_mode)

    # Fixed theta*: if not provided, draw one from the healthy search
    if theta_star is None:
        healthy_cfg = base_cfg['healthy_dgp']
        seeds = base_cfg['seeds']
        from generate_data import generate_data
        theta_star, _, _, _ = generate_data(dgp_cfg, healthy_cfg, seeds, verbose=True)

    theta_rev = theta_star[:n_shared]
    theta_fc = theta_star[n_shared:n_shared + N]
    theta_gs_val = theta_star[-1]

    logger.info(f"MONTE CARLO: C={C}, M={M}, N={N}, n_params={n_p}, n_reps={n_reps}")
    logger.info(f"Fixed theta* = {theta_star}")
    logger.info(f"Drawing fresh geography + firms + errors each replication")

    all_theta_hat = []
    all_converged = []
    all_iters = []
    all_bundle_sizes = []
    t0 = time.perf_counter()

    for rep in range(n_reps):
        rng_rep = np.random.default_rng((42, 99999, rep))

        # Fresh geography
        locations, dists, populations = build_geography(C, pop_log_std, rng_rep)
        _, origin_of, dest_of, M_check = build_edges(C)
        phi = build_covariates(C, M, origin_of, dest_of, dists, populations, fe_mode)

        # Fresh hubs
        hubs = build_hubs(N, C, populations, min_hubs, max_hubs, hub_pool_frac, rng_rep)

        # Fresh errors
        rng_err = np.random.default_rng((42, 88888, rep))
        errors = rng_err.normal(0, sigma, (N, M))

        # Observed bundles at theta*
        obs_bundles = np.zeros((N, M), dtype=bool)
        for i in range(N):
            obs_bundles[i] = greedy_demand(
                phi, theta_rev, theta_fc[i], theta_gs_val,
                hubs[i], origin_of, errors[i], M)

        sizes = obs_bundles.sum(axis=1)
        all_bundle_sizes.append(sizes.tolist())

        # Skip if degenerate (all empty or all full)
        if sizes.min() == 0 or sizes.max() == M:
            print(f"  Rep {rep:>3d}: DEGENERATE, skipping")
            continue

        # Estimation
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

        lbs = {0: 0}
        for i in range(N):
            lbs[n_shared + i] = 0
        lbs[n_shared + N] = 0

        model_cfg = {
            'dimensions': {
                'n_obs': N, 'n_items': M, 'n_covariates': n_p,
                'n_simulations': est_cfg['n_simulations'],
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

        err_seed = (42, 77777, rep)
        model.features.build_local_modular_error_oracle(seed=err_seed, sigma=sigma)
        model.features.set_covariates_oracle(build_covariates_oracle(N))

        model.subproblems.load_solver(AirlineGreedySolver)
        model.subproblems.initialize_solver()

        row_gen = model.point_estimation.n_slack
        result = row_gen.solve(initialize_solver=False, initialize_master=True,
                               verbose=False)

        if result is not None:
            all_theta_hat.append(result.theta_hat.copy())
            all_converged.append(result.converged)
            all_iters.append(result.num_iterations)
            err_pct = np.abs(result.theta_hat - theta_star) / np.abs(theta_star) * 100
            print(f"  Rep {rep:>3d}: converged={result.converged}, "
                  f"iters={result.num_iterations:>3d}, "
                  f"bundles={sizes.min()}-{sizes.max()}, "
                  f"max_err={err_pct.max():.1f}%")

    t_total = time.perf_counter() - t0

    if len(all_theta_hat) == 0:
        print("No successful replications!")
        return

    estimates = np.array(all_theta_hat)
    n_ok = len(estimates)
    n_converged = sum(all_converged)

    deviations = estimates - theta_star
    hat_means = estimates.mean(axis=0)
    hat_stds = estimates.std(axis=0, ddof=1)
    hat_ses = hat_stds / np.sqrt(n_ok)
    bias = deviations.mean(axis=0)
    rmse = np.sqrt((deviations ** 2).mean(axis=0))
    mae_pct = (np.abs(deviations) / np.abs(theta_star) * 100).mean(axis=0)
    median_ae_pct = np.median(
        np.abs(deviations) / np.abs(theta_star) * 100, axis=0)

    shared_names = ["theta_rev"]
    if fe_mode == "origin":
        shared_names += [f"theta_fe_{j}" for j in range(C)]
    fc_names = [f"theta_fc_{i}" for i in range(N)]
    param_names = shared_names + fc_names + ["theta_gs"]

    W = 84
    print(f"\n{'=' * W}")
    print(f"  MONTE CARLO (full DGP variation)")
    print(f"  C={C}, M={M}, N={N}, {n_p} params, S={est_cfg['n_simulations']}")
    print(f"  {n_ok}/{n_reps} reps, {n_converged} converged, "
          f"{t_total:.0f}s ({t_total/max(n_ok,1):.1f}s/rep)")
    print(f"  Mean iters: {np.mean(all_iters):.1f}")
    print(f"{'=' * W}")
    header = (f"  {'Param':<14} {'True':>8} {'Mean':>8} {'Bias':>8} "
              f"{'Std':>8} {'RMSE':>8} {'MAE%':>7} {'MdAE%':>7}")
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    for j, name in enumerate(param_names):
        print(f"  {name:<14} {theta_star[j]:>8.4f} {hat_means[j]:>8.4f} "
              f"{bias[j]:>+8.4f} {hat_stds[j]:>8.4f} "
              f"{rmse[j]:>8.4f} {mae_pct[j]:>6.1f}% {median_ae_pct[j]:>6.1f}%")

    if N > 1:
        print(f"  {'-' * (len(header) - 2)}")
        fc_sl = slice(n_shared, n_shared + N)
        print(f"  {'avg(fc)':<14} {'':>8} {'':>8} "
              f"{bias[fc_sl].mean():>+8.4f} {hat_stds[fc_sl].mean():>8.4f} "
              f"{rmse[fc_sl].mean():>8.4f} {mae_pct[fc_sl].mean():>6.1f}% "
              f"{median_ae_pct[fc_sl].mean():>6.1f}%")
    print()

    # Save
    out = {
        'config': {
            'C': C, 'M': M, 'N': N, 'fe_mode': fe_mode, 'sigma': sigma,
            'n_params': n_p, 'n_reps': n_ok, 'n_converged': n_converged,
            'n_simulations': est_cfg['n_simulations'],
            'runtime_total_s': round(t_total, 1),
        },
        'theta_true': theta_star.tolist(),
        'param_names': param_names,
        'mean': hat_means.tolist(),
        'bias': bias.tolist(),
        'std': hat_stds.tolist(),
        'rmse': rmse.tolist(),
        'mae_pct': mae_pct.tolist(),
        'median_ae_pct': median_ae_pct.tolist(),
    }
    out_path = BASE / 'results' / 'monte_carlo.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"  Saved to {out_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-reps', type=int, default=30)
    parser.add_argument('--N', type=int, default=None)
    parser.add_argument('--S', type=int, default=None)
    args = parser.parse_args()

    cfg = {k: dict(v) if isinstance(v, dict) else v for k, v in CFG.items()}
    if args.N is not None:
        cfg['dgp'] = dict(cfg['dgp'])
        cfg['dgp']['N'] = args.N
    if args.S is not None:
        cfg['estimation'] = dict(cfg['estimation'])
        cfg['estimation']['n_simulations'] = args.S

    run_monte_carlo(n_reps=args.n_reps, base_cfg=cfg)
