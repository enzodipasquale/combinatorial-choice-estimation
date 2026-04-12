"""Run multiple replications with FIXED true theta, varying only errors."""

import sys
from pathlib import Path
import numpy as np
import yaml
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_data import (generate_data, greedy_demand,
                           n_shared_covariates, n_params)
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
    n_p = n_params(C, N, fe_mode)
    n_shared = n_shared_covariates(C, fe_mode)

    # Unpack theta_star: [theta_rev..., theta_fc_0, ..., theta_fc_{N-1}, theta_gs]
    theta_rev = theta_star[:n_shared]
    theta_fc = theta_star[n_shared:n_shared + N]
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
                phi, theta_rev, theta_fc[i], theta_gs,
                hubs[i], origin_of, errors[i], M)

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
                'N_firms': N,
            },
        }

        lbs = {0: 0}
        for i in range(N):
            lbs[n_shared + i] = 0
        lbs[n_shared + N] = 0
        theta_bounds = {'lb': -20, 'ub': 20, 'lbs': lbs}

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

        err_seed = seeds['error'] + rep * 1000
        model.features.build_local_modular_error_oracle(seed=err_seed, sigma=sigma)

        cov_oracle = build_covariates_oracle(N)
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
                  f"max_err%={err_pct.max():.1f}%")

    t_total = time.perf_counter() - t0

    if len(all_theta_hat) == 0:
        print("No successful replications!")
        return

    estimates = np.array(all_theta_hat)
    n_ok = len(estimates)

    errors = estimates - theta_star
    bias = errors.mean(axis=0)
    hat_means = estimates.mean(axis=0)
    hat_ses = estimates.std(axis=0, ddof=1) / np.sqrt(n_ok)

    shared_names = ["theta_rev"]
    if fe_mode == "origin":
        shared_names += [f"theta_fe_{j}" for j in range(C)]
    fc_names = [f"theta_fc_{i}" for i in range(N)]
    param_names = shared_names + fc_names + ["theta_gs"]

    print(f"\n{'='*72}")
    print(f"  N={N}, C={C}, M={M}, {n_ok}/{n_reps} reps, "
          f"{n_converged} converged, {t_total:.1f}s")
    print(f"{'='*72}")
    print(f"  {'Param':<14} {'True':>10} {'E[θ̂]':>10} {'SE':>10} {'t':>8}")
    print(f"  {'-'*52}")
    for j, name in enumerate(param_names):
        t_stat = hat_means[j] / hat_ses[j] if hat_ses[j] > 0 else float('inf')
        print(f"  {name:<14} {theta_star[j]:>10.4f} {hat_means[j]:>10.4f} "
              f"{hat_ses[j]:>10.4f} {t_stat:>8.1f}")
    print()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-reps', type=int,
                        default=CFG.get('replications', {}).get('n_reps', 30))
    parser.add_argument('--N', type=int, default=None)
    args = parser.parse_args()

    cfg = {k: dict(v) if isinstance(v, dict) else v for k, v in CFG.items()}
    if args.N is not None:
        cfg['dgp'] = dict(cfg['dgp'])
        cfg['dgp']['N'] = args.N

    run_replications(n_reps=args.n_reps, base_cfg=cfg)
