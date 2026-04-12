"""End-to-end: DGP -> estimation -> result.json for airline / GS scenario."""

import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
import yaml
import combest as ce
from combest.utils import get_logger
from combest.subproblems.registry.greedy import GreedySolver

logger = get_logger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_data import generate_data, n_params, n_covariates
from oracle import make_find_best_item, build_covariates_oracle

BASE = Path(__file__).resolve().parent
with open(BASE / 'config.yaml') as f:
    CFG = yaml.safe_load(f)


class AirlineGreedySolver(GreedySolver):
    """GreedySolver with custom find_best_item for airline GS."""
    find_best_item = staticmethod(make_find_best_item())


def run(C, N, fe_mode, sigma, dgp_seed, search_seed, err_seed,
        max_candidates=200, max_iters=200, tolerance=1e-6,
        gurobi_timeout=30, verbose=True):

    model = ce.Model()
    is_root = model.is_root()

    # --- DGP ---
    t0 = time.perf_counter()
    theta_star, obs_bundles, dgp_data, dgp_diag = generate_data(
        C=C, N=N, fe_mode=fe_mode, sigma=sigma,
        dgp_seed=dgp_seed, search_seed=search_seed,
        max_candidates=max_candidates, verbose=verbose and is_root)

    M = dgp_data['M']
    n_p = n_params(C, fe_mode)
    n_mod = n_covariates(C, fe_mode)

    if is_root:
        logger.info(f"DGP: C={C}, M={M}, N={N}, fe_mode={fe_mode}, n_params={n_p}")
        logger.info(f"theta_star = {theta_star}")

    # --- Prepare input data ---
    if is_root:
        input_data = {
            'id_data': {
                'obs_bundles': obs_bundles,
                'hubs': [list(h) for h in dgp_data['hubs']],
            },
            'item_data': {
                'phi': dgp_data['phi'],
                'origin_of': dgp_data['origin_of'],
            },
        }
    else:
        input_data = None

    # --- Configure model ---
    # theta = [theta_mod_0, ..., theta_mod_{n_mod-1}, theta_gs]
    # theta_gs >= 0 (last parameter)
    theta_bounds = {
        'lb': -20, 'ub': 20,
        'lbs': {n_mod: 0},  # theta_gs >= 0
    }

    cfg = {
        'dimensions': {
            'n_obs': N, 'n_items': M,
            'n_covariates': n_p, 'n_simulations': 1,
        },
        'subproblem': {
            'gurobi_params': {'TimeLimit': gurobi_timeout, 'OutputFlag': 0},
        },
        'row_generation': {
            'max_iters': max_iters,
            'tolerance': tolerance,
            'theta_bounds': theta_bounds,
        },
    }
    model.load_config(cfg)
    model.data.load_and_distribute_input_data(input_data)

    # Convert hubs from lists to sets in local data (needed by oracles)
    ld = model.data.local_data
    ld.id_data['hubs'] = [set(h) for h in ld.id_data['hubs']]

    # --- Build simulation errors (FRESH, independent of DGP) ---
    model.features.build_local_modular_error_oracle(seed=err_seed, sigma=sigma)

    # --- Set covariates oracle ---
    cov_oracle = build_covariates_oracle()
    model.features.set_covariates_oracle(cov_oracle)

    # --- Load and initialize solver ---
    model.subproblems.load_solver(AirlineGreedySolver)
    model.subproblems.initialize_solver()

    # --- Run estimation ---
    row_gen = model.point_estimation.n_slack
    result = row_gen.solve(initialize_solver=False, initialize_master=True,
                           verbose=verbose)

    t_total = time.perf_counter() - t0

    # --- Report ---
    if is_root and result is not None:
        theta_hat = result.theta_hat
        error_pct = np.abs(theta_hat - theta_star) / np.abs(theta_star) * 100

        logger.info("")
        logger.info("=" * 60)
        logger.info(f"  Results (C={C}, M={M}, N={N}, fe_mode={fe_mode})")
        logger.info("=" * 60)
        param_names = [f"theta_mod_{j}" for j in range(n_mod)] + ["theta_gs"]
        for j, name in enumerate(param_names):
            logger.info(f"  {name:<20} true={theta_star[j]:>8.4f}  "
                        f"hat={theta_hat[j]:>8.4f}  err={error_pct[j]:>6.1f}%")
        logger.info(f"")
        logger.info(f"  Converged: {result.converged}")
        logger.info(f"  Row-gen iterations: {result.num_iterations}")
        logger.info(f"  Runtime: {t_total:.1f}s")

        # --- Write result.json ---
        out = {
            'theta_true': theta_star.tolist(),
            'theta_hat': theta_hat.tolist(),
            'error_pct': error_pct.tolist(),
            'n_cities': C,
            'n_edges': M,
            'n_airlines': N,
            'fe_mode': fe_mode,
            'runtime_seconds': round(t_total, 2),
            'row_generation_iters': result.num_iterations,
            'converged': result.converged,
            'dgp_healthy_check': dgp_diag,
        }
        out_path = BASE / 'results' / 'result.json'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(out, f, indent=2)
        logger.info(f"  Result written to {out_path}")

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Airline / GS estimation')
    parser.add_argument('--C', type=int, default=CFG['dgp']['C'])
    parser.add_argument('--N', type=int, default=CFG['dgp']['N'])
    parser.add_argument('--fe-mode', type=str, default=CFG['dgp']['fe_mode'])
    parser.add_argument('--sigma', type=float, default=CFG['dgp']['sigma'])
    parser.add_argument('--dgp-seed', type=int, default=CFG['seeds']['dgp'])
    parser.add_argument('--search-seed', type=int, default=CFG['seeds']['search'])
    parser.add_argument('--err-seed', type=int, default=CFG['seeds']['error'])
    parser.add_argument('--max-iters', type=int,
                        default=CFG['estimation']['max_iters'])
    args = parser.parse_args()

    run(C=args.C, N=args.N, fe_mode=args.fe_mode, sigma=args.sigma,
        dgp_seed=args.dgp_seed, search_seed=args.search_seed,
        err_seed=args.err_seed, max_iters=args.max_iters)
