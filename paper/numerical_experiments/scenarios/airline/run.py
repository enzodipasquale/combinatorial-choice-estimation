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
from generate_data import generate_data, n_params, n_shared_covariates
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
    n_p = n_params(C, N, fe_mode)
    n_shared = n_shared_covariates(C, fe_mode)

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
                'N_firms': N,
            },
        }
    else:
        input_data = None

    # --- Configure model ---
    # theta = [theta_rev..., theta_fc_0, ..., theta_fc_{N-1}, theta_gs]
    # theta_rev >= 0, all theta_fc >= 0, theta_gs >= 0
    lbs = {0: 0}  # theta_rev >= 0
    for i in range(N):
        lbs[n_shared + i] = 0  # theta_fc_i >= 0
    lbs[n_shared + N] = 0  # theta_gs >= 0
    theta_bounds = {'lb': -20, 'ub': 20, 'lbs': lbs}

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
    cov_oracle = build_covariates_oracle(N)
    model.features.set_covariates_oracle(cov_oracle)

    # --- Load and initialize solver ---
    model.subproblems.load_solver(AirlineGreedySolver)
    model.subproblems.initialize_solver()

    # --- Point estimation ---
    se_result = model.standard_errors.compute_bootstrap(
        num_bootstrap=est_cfg.get('num_bootstrap', 100),
        seed=seeds.get('bootstrap', 999),
        method='bayesian',
        verbose=True,
    )

    t_total = time.perf_counter() - t0

    # --- Report ---
    if is_root and se_result is not None:
        theta_hat = se_result.mean
        theta_se = se_result.se
        t_stats = se_result.t_stats
        error_pct = np.abs(theta_hat - theta_star) / np.abs(theta_star) * 100

        logger.info("")
        logger.info("=" * 80)
        logger.info(f"  Results (C={C}, M={M}, N={N}, fe_mode={fe_mode})")
        logger.info("=" * 80)
        shared_names = ["theta_rev"]
        if fe_mode == "origin":
            shared_names += [f"theta_fe_{j}" for j in range(C)]
        fc_names = [f"theta_fc_{i}" for i in range(N)]
        param_names = shared_names + fc_names + ["theta_gs"]
        logger.info(f"  {'Param':<20} {'True':>8} {'Est':>8} {'SE':>8} "
                    f"{'t':>8} {'Err%':>7}")
        logger.info(f"  {'-'*59}")
        for j, name in enumerate(param_names):
            logger.info(f"  {name:<20} {theta_star[j]:>8.4f} {theta_hat[j]:>8.4f} "
                        f"{theta_se[j]:>8.4f} {t_stats[j]:>8.1f} "
                        f"{error_pct[j]:>6.1f}%")
        logger.info(f"")
        logger.info(f"  Bootstrap samples: {se_result.n_samples}")
        logger.info(f"  Runtime: {t_total:.1f}s")

        # --- Write result.json ---
        out = {
            'theta_true': theta_star.tolist(),
            'theta_hat': theta_hat.tolist(),
            'theta_se': theta_se.tolist(),
            't_stats': t_stats.tolist(),
            'error_pct': error_pct.tolist(),
            'ci_lower': se_result.ci_lower.tolist(),
            'ci_upper': se_result.ci_upper.tolist(),
            'n_cities': C,
            'n_edges': M,
            'n_airlines': N,
            'fe_mode': fe_mode,
            'n_bootstrap': se_result.n_samples,
            'runtime_seconds': round(t_total, 2),
            'dgp_healthy_check': dgp_diag,
        }
        out_path = BASE / 'results' / 'result.json'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(out, f, indent=2)
        logger.info(f"  Result written to {out_path}")

    return se_result


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
