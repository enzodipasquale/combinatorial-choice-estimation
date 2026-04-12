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
from generate_data import generate_data, n_params, n_modular
from oracle import make_find_best_item, build_covariates_oracle

BASE = Path(__file__).resolve().parent
with open(BASE / 'config.yaml') as f:
    CFG = yaml.safe_load(f)


class AirlineGreedySolver(GreedySolver):
    """GreedySolver with custom find_best_item for airline GS."""
    find_best_item = staticmethod(make_find_best_item())


def run(cfg=None):
    if cfg is None:
        cfg = CFG

    dgp_cfg = cfg['dgp']
    seeds = cfg['seeds']
    est_cfg = cfg['estimation']
    healthy_cfg = cfg['healthy_dgp']

    C = dgp_cfg['C']
    N = dgp_cfg['N']
    fe_mode = dgp_cfg['fe_mode']
    sigma = dgp_cfg['sigma']

    model = ce.Model()
    is_root = model.is_root()

    # --- DGP ---
    t0 = time.perf_counter()
    theta_star, obs_bundles, dgp_data, dgp_diag = generate_data(
        dgp_cfg, healthy_cfg, seeds, verbose=is_root)

    M = dgp_data['M']
    n_p = n_params(C, fe_mode)
    n_mod = n_modular(C, fe_mode)

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
    # theta = [theta_rev, theta_fc, ..., theta_gs]
    # theta_rev >= 0 (index 0), theta_fc >= 0 (index 1), theta_gs >= 0 (last)
    theta_bounds = {
        'lb': -20, 'ub': 20,
        'lbs': {0: 0, 1: 0, n_mod: 0},
    }

    model_cfg = {
        'dimensions': {
            'n_obs': N, 'n_items': M,
            'n_covariates': n_p, 'n_simulations': est_cfg['n_simulations'],
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

    # Convert hubs from lists to sets in local data
    ld = model.data.local_data
    ld.id_data['hubs'] = [set(h) for h in ld.id_data['hubs']]

    # --- Build simulation errors (FRESH, independent of DGP) ---
    model.features.build_local_modular_error_oracle(seed=seeds['error'], sigma=sigma)

    # --- Set covariates oracle ---
    cov_oracle = build_covariates_oracle()
    model.features.set_covariates_oracle(cov_oracle)

    # --- Load and initialize solver ---
    model.subproblems.load_solver(AirlineGreedySolver)
    model.subproblems.initialize_solver()

    # --- Run estimation ---
    row_gen = model.point_estimation.n_slack
    result = row_gen.solve(initialize_solver=False, initialize_master=True,
                           verbose=True)

    t_total = time.perf_counter() - t0

    # --- Report ---
    if is_root and result is not None:
        theta_hat = result.theta_hat
        error_pct = np.abs(theta_hat - theta_star) / np.abs(theta_star) * 100

        logger.info("")
        logger.info("=" * 60)
        logger.info(f"  Results (C={C}, M={M}, N={N}, fe_mode={fe_mode})")
        logger.info("=" * 60)
        base_names = ["theta_rev", "theta_fc"]
        if fe_mode == "origin":
            base_names += [f"theta_fe_{j}" for j in range(C)]
        param_names = base_names + ["theta_gs"]
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
    parser.add_argument('--C', type=int, default=None)
    parser.add_argument('--N', type=int, default=None)
    parser.add_argument('--fe-mode', type=str, default=None)
    parser.add_argument('--sigma', type=float, default=None)
    parser.add_argument('--max-hubs', type=int, default=None)
    args = parser.parse_args()

    cfg = CFG.copy()
    cfg['dgp'] = dict(cfg['dgp'])
    if args.C is not None:
        cfg['dgp']['C'] = args.C
    if args.N is not None:
        cfg['dgp']['N'] = args.N
    if args.fe_mode is not None:
        cfg['dgp']['fe_mode'] = args.fe_mode
    if args.sigma is not None:
        cfg['dgp']['sigma'] = args.sigma
    if args.max_hubs is not None:
        cfg['dgp']['max_hubs'] = args.max_hubs

    run(cfg)
