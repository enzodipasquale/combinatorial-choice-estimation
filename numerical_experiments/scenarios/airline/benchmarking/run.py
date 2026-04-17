"""Benchmark: BGK moment inequality estimator vs combest.

Generates data, runs BGK to get identified set [LB, UB],
compares width with combest point estimates.
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from generate_data import (build_geography, build_edges, build_covariates,
                           build_hubs, greedy_demand)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bgk_estimator import estimate_identified_set

from combest.utils import get_logger

logger = get_logger(__name__)

BASE = Path(__file__).resolve().parent
PARENT = BASE.parent
with open(PARENT / 'config.yaml') as f:
    CFG = yaml.safe_load(f)


def run(cfg=None, sigma=0.0):
    if cfg is None:
        cfg = CFG

    dgp_cfg = cfg['dgp']
    C = dgp_cfg['C']
    N = dgp_cfg['N']
    pop_log_std = dgp_cfg['pop_log_std']
    hub_pool_frac = dgp_cfg['hub_pool_frac']
    min_hubs = dgp_cfg['min_hubs']
    max_hubs = dgp_cfg['max_hubs']
    fe_mode = dgp_cfg['fe_mode']

    M = C * (C - 1)

    rng_dgp = np.random.default_rng(42)
    _, origin_of, dest_of, _ = build_edges(C)
    locations, dists, populations = build_geography(C, pop_log_std, rng_dgp)
    phi = build_covariates(C, M, origin_of, dest_of, dists, populations, fe_mode)
    hubs = build_hubs(N, C, populations, min_hubs, max_hubs, hub_pool_frac, rng_dgp)

    theta_rev_true = 1.0
    theta_fc_true = np.array([0.5 + 0.1 * i for i in range(N)])
    theta_gs_true = 2.0
    theta_star = np.concatenate([theta_fc_true, [theta_gs_true]])
    theta_rev_vec = np.array([theta_rev_true])

    if sigma > 0:
        rng_err = np.random.default_rng(1041)
        errors = rng_err.normal(0, sigma, (N, M))
    else:
        errors = np.zeros((N, M))

    obs_bundles = np.zeros((N, M), dtype=bool)
    for i in range(N):
        obs_bundles[i] = greedy_demand(
            phi, theta_rev_vec, theta_fc_true[i], theta_gs_true,
            hubs[i], origin_of, errors[i], M)

    sizes = obs_bundles.sum(axis=1)
    logger.info(f"BENCHMARK: C={C}, M={M}, N={N}, sigma={sigma}")
    logger.info(f"Bundle sizes: min={sizes.min()}, max={sizes.max()}, "
                f"mean={sizes.mean():.0f}")

    # --- BGK ---
    t0 = time.perf_counter()
    bgk_results, n_ineqs = estimate_identified_set(
        obs_bundles, phi, hubs, origin_of, N, M, theta_rev_true)
    t_bgk = time.perf_counter() - t0

    param_names = [f"theta_fc_{i}" for i in range(N)] + ["theta_gs"]

    logger.info(f"\n{'='*74}")
    logger.info(f"  BGK identified set ({n_ineqs} inequalities, {t_bgk:.2f}s)")
    logger.info(f"{'='*74}")
    logger.info(f"  {'Param':<14} {'True':>8} {'LB':>10} {'UB':>10} "
                f"{'Width':>8} {'Cover':>6}")
    logger.info(f"  {'-'*56}")

    widths = []
    all_cover = True
    for j, name in enumerate(param_names):
        r = bgk_results[name]
        lb = r['lb'] if r['lb'] is not None else float('-inf')
        ub = r['ub'] if r['ub'] is not None else float('inf')
        width = ub - lb if np.isfinite(ub) and np.isfinite(lb) else float('inf')
        covers = lb - 1e-8 <= theta_star[j] <= ub + 1e-8
        if not covers:
            all_cover = False
        widths.append(width)
        lb_s = f"{lb:>10.4f}" if np.isfinite(lb) else f"{'  -inf':>10}"
        ub_s = f"{ub:>10.4f}" if np.isfinite(ub) else f"{'   inf':>10}"
        w_s = f"{width:>8.4f}" if np.isfinite(width) else f"{'  inf':>8}"
        logger.info(f"  {name:<14} {theta_star[j]:>8.4f} {lb_s} {ub_s} "
                    f"{w_s} {'Yes' if covers else ' NO':>6}")

    fc_widths = widths[:N]
    avg_fc_width = np.mean([w for w in fc_widths if np.isfinite(w)])
    gs_width = widths[-1]
    logger.info(f"  {'-'*56}")
    logger.info(f"  avg FC width: {avg_fc_width:.4f}, "
                f"GS width: {gs_width:.4f}")
    logger.info(f"  All covered: {all_cover}")

    out = {
        'sigma': sigma,
        'theta_true': theta_star.tolist(),
        'param_names': param_names,
        'bgk': {name: bgk_results[name] for name in param_names},
        'n_inequalities': n_ineqs,
        'bgk_time_s': round(t_bgk, 2),
        'avg_fc_width': float(avg_fc_width),
        'gs_width': float(gs_width),
        'all_covered': all_cover,
        'config': {'C': C, 'M': M, 'N': N},
    }
    out_path = BASE / 'result.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    logger.info(f"  Saved to {out_path}")

    return bgk_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sigma', type=float, default=0.0)
    args = parser.parse_args()
    run(sigma=args.sigma)
