"""Run multiple replications with FIXED true theta, varying only errors."""

import sys
import json
from pathlib import Path
import numpy as np
import yaml
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_data import (generate_data, build_edges, build_geography,
                           build_covariates, build_hubs, greedy_demand,
                           n_modular, n_params)
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


def run_replications(n_reps=30, base_cfg=None):
    if base_cfg is None:
        base_cfg = CFG

    dgp_cfg = base_cfg['dgp']
    healthy_cfg = base_cfg['healthy_dgp']
    est_cfg = base_cfg['estimation']
    seeds = base_cfg['seeds']

    C = dgp_cfg['C']
    N = dgp_cfg['N']
    fe_mode = dgp_cfg['fe_mode']
    sigma = dgp_cfg['sigma']

    # --- Fixed DGP: geography, hubs, theta* (drawn once) ---
    theta_star, _, dgp_data, dgp_diag = generate_data(
        dgp_cfg, healthy_cfg, seeds, verbose=True)

    phi = dgp_data['phi']
    hubs = dgp_data['hubs']
    origin_of = dgp_data['origin_of']
    M = dgp_data['M']
    n_p = n_params(C, fe_mode)
    n_mod = n_modular(C, fe_mode)

    theta_mod = theta_star[:-1]
    theta_gs = theta_star[-1]

    logger.info(f"Fixed theta_star = {theta_star}")
    logger.info(f"Running {n_reps} replications (varying errors only)")

    all_theta_hat = []
    n_converged = 0
    t0 = time.perf_counter()

    for rep in range(n_reps):
        # --- Draw fresh DGP errors for this replication ---
        rng_dgp_err = np.random.default_rng(seeds['dgp'] + 999 + rep)
        errors = rng_dgp_err.normal(0, sigma, (N, M))

        # Compute observed bundles at theta_star with these errors
        obs_bundles = np.zeros((N, M), dtype=bool)
        for i in range(N):
            obs_bundles[i] = greedy_demand(
                phi, theta_mod, theta_gs, hubs[i], origin_of, errors[i], M)

        # --- Estimation ---
        model = ce.Model()

        input_data = {
            'id_data': {
                'obs_bundles': obs_bundles,
                'hubs': [list(h) for h in hubs],
            },
            'item_data': {
                'phi': phi,
                'origin_of': origin_of,
            },
        }

        theta_bounds = {
            'lb': -20, 'ub': 20,
            'lbs': {0: 0, 1: 0, n_mod: 0},
        }

        model_cfg = {
            'dimensions': {
                'n_obs': N, 'n_items': M,
                'n_covariates': n_p,
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
                'theta_bounds': theta_bounds,
            },
        }
        model.load_config(model_cfg)
        model.data.load_and_distribute_input_data(input_data)

        ld = model.data.local_data
        ld.id_data['hubs'] = [set(h) for h in ld.id_data['hubs']]

        # Fresh simulation errors (independent of DGP errors)
        err_seed = seeds['error'] + rep * 1000
        model.features.build_local_modular_error_oracle(seed=err_seed, sigma=sigma)

        cov_oracle = build_covariates_oracle()
        model.features.set_covariates_oracle(cov_oracle)

        model.subproblems.load_solver(AirlineGreedySolver)
        model.subproblems.initialize_solver()

        row_gen = model.point_estimation.n_slack
        result = row_gen.solve(initialize_solver=False, initialize_master=True,
                               verbose=False)

        if result is not None:
            all_theta_hat.append(result.theta_hat.copy())
            if result.converged:
                n_converged += 1
            err_pct = np.abs(result.theta_hat - theta_star) / np.abs(theta_star) * 100
            print(f"  Rep {rep}: converged={result.converged}, "
                  f"err%=[{', '.join(f'{e:.1f}' for e in err_pct)}]")

    t_total = time.perf_counter() - t0

    if len(all_theta_hat) == 0:
        print("No successful replications!")
        return

    estimates = np.array(all_theta_hat)
    n_ok = len(estimates)

    errors = estimates - theta_star
    bias = errors.mean(axis=0)
    rmse = np.sqrt((errors ** 2).mean(axis=0))
    mae_pct = (np.abs(errors) / np.abs(theta_star) * 100).mean(axis=0)
    hat_means = estimates.mean(axis=0)
    bias_ses = errors.std(axis=0, ddof=1) / np.sqrt(n_ok)

    base_names = ["theta_rev", "theta_fc"]
    if fe_mode == "origin":
        base_names += [f"theta_fe_{j}" for j in range(C)]
    param_names = base_names + ["theta_gs"]

    print(f"\n{'='*80}")
    print(f"  N={N}, C={C}, M={M}, {n_ok}/{n_reps} reps, "
          f"{n_converged} converged, {t_total:.1f}s")
    print(f"{'='*80}")
    print(f"  {'Param':<14} {'True':>8} {'E[θ̂]':>8} {'Bias':>8} {'SE(bias)':>8} "
          f"{'RMSE':>8} {'MAE%':>8}")
    print(f"  {'-'*64}")
    for j, name in enumerate(param_names):
        print(f"  {name:<14} {theta_star[j]:>8.4f} {hat_means[j]:>8.4f} "
              f"{bias[j]:>8.4f} {bias_ses[j]:>8.4f} "
              f"{rmse[j]:>8.4f} {mae_pct[j]:>7.1f}%")
    print()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-reps', type=int, default=30)
    parser.add_argument('--N', type=int, default=None)
    args = parser.parse_args()

    cfg = {k: dict(v) if isinstance(v, dict) else v for k, v in CFG.items()}
    if args.N is not None:
        cfg['dgp'] = dict(cfg['dgp'])
        cfg['dgp']['N'] = args.N

    run_replications(n_reps=args.n_reps, base_cfg=cfg)
