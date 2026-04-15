"""Zero-limit estimator for airline / GS scenario.

DGP has NO modular errors: choices are deterministic given theta*.
The gravity covariate is moved into the error term with coefficient
fixed to 1. Only theta_fc (firm-specific) and theta_gs are estimated.

With no misspecification, the estimator should recover theta* exactly.
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
import yaml
import combest as ce
from combest.utils import get_logger
from combest.subproblems.registry.greedy import GreedySolver

logger = get_logger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from generate_data import (build_geography, build_edges, build_covariates,
                           build_hubs, greedy_demand)
from oracle import make_find_best_item

BASE = Path(__file__).resolve().parent
PARENT = BASE.parent
with open(PARENT / 'config.yaml') as f:
    CFG = yaml.safe_load(f)


class AirlineGreedySolver(GreedySolver):
    find_best_item = staticmethod(make_find_best_item())


def build_zero_limit_covariates_oracle(N_firms):
    """Covariates oracle for zero-limit: NO gravity column.

    theta layout: [theta_fc_0, ..., theta_fc_{N-1}, theta_gs]

    Gravity is in the error term (fixed coeff=1), not estimated.
    """
    def covariates_oracle(bundles, ids, data):
        origin_of = data.item_data['origin_of']
        hubs_list = data.id_data['hubs']
        obs_ids = data.item_data.get('obs_ids', None)

        n_agents = bundles.shape[0]
        n_total = N_firms + 1  # firm FCs + congestion
        features = np.zeros((n_agents, n_total))

        for i_local, idx in enumerate(ids):
            obs_idx = int(obs_ids[idx]) if obs_ids is not None else idx
            b = bundles[i_local]
            # Firm-specific fixed cost: -bundle_size
            features[i_local, obs_idx] = -float(b.sum())
            # Congestion: -sum_h n_h^2
            hubs_i = hubs_list[obs_idx]
            cong = 0.0
            for h in hubs_i:
                n_h = int((b & (origin_of == h)).sum())
                cong += n_h ** 2
            features[i_local, -1] = -cong

        return features

    return covariates_oracle


def make_zero_limit_find_best_item():
    """find_best_item for zero-limit: gravity is in the error, not in theta.

    theta layout: [theta_fc_0, ..., theta_fc_{N-1}, theta_gs]
    modular_error contains the gravity values (fixed coeff=1).
    """
    def find_best_item(local_id, bundle, items_left, theta, best_val,
                       data, modular_error, cache=None):
        if cache is None:
            cache = {}

        if 'base' not in cache:
            origin_of = data.item_data['origin_of']
            obs_ids = data.item_data.get('obs_ids', None)
            obs_idx = int(obs_ids[local_id]) if obs_ids is not None else local_id
            hubs_i = data.id_data['hubs'][obs_idx]
            N_firms = data.item_data['N_firms']

            theta_fc_i = theta[obs_idx]

            # base = gravity (in modular_error) - firm FC
            cache['base'] = modular_error - theta_fc_i
            cache['hub_masks'] = {h: (origin_of == h) for h in hubs_i}
            cache['hub_counts'] = {h: 0 for h in hubs_i}

        base = cache['base']
        hub_masks = cache['hub_masks']
        hub_counts = cache['hub_counts']
        theta_gs = theta[-1]

        marginals = base.copy()
        if theta_gs > 0:
            for h, mask in hub_masks.items():
                marginals[mask] -= theta_gs * (2 * hub_counts[h] + 1)
        marginals[~items_left] = -np.inf

        best_item = int(np.argmax(marginals))
        best_marginal = marginals[best_item]

        if best_marginal <= 0:
            return -1, best_val

        for h, mask in hub_masks.items():
            if mask[best_item]:
                hub_counts[h] += 1
                break

        return best_item, best_val + best_marginal

    return find_best_item


# Use the zero-limit find_best_item
AirlineGreedySolver.find_best_item = staticmethod(make_zero_limit_find_best_item())


def run(cfg=None):
    if cfg is None:
        cfg = CFG

    dgp_cfg = cfg['dgp']
    est_cfg = cfg['estimation']

    C = dgp_cfg['C']
    N = dgp_cfg['N']
    fe_mode = dgp_cfg['fe_mode']
    sigma = dgp_cfg['sigma']
    pop_log_std = dgp_cfg['pop_log_std']
    hub_pool_frac = dgp_cfg['hub_pool_frac']
    min_hubs = dgp_cfg['min_hubs']
    max_hubs = dgp_cfg['max_hubs']

    model = ce.Model()
    is_root = model.is_root()

    t0 = time.perf_counter()

    # --- Build geography ---
    rng_dgp = np.random.default_rng(42)
    _, origin_of, dest_of, M = build_edges(C)
    locations, dists, populations = build_geography(C, pop_log_std, rng_dgp)
    phi = build_covariates(C, M, origin_of, dest_of, dists, populations, fe_mode)
    hubs = build_hubs(N, C, populations, min_hubs, max_hubs, hub_pool_frac, rng_dgp)

    # --- True parameters ---
    # theta_rev is fixed to 1 (goes into the error term)
    # We set it to 1.0 and put gravity * 1.0 = gravity into the "error"
    gravity = phi[:, 0]  # (M,)
    theta_rev_true = 1.0
    theta_fc_true = np.array([0.5 + 0.1 * i for i in range(N)])  # spread of FCs
    theta_gs_true = 2.0

    # Full theta for reporting: [theta_rev=1, theta_fc_0, ..., theta_fc_{N-1}, theta_gs]
    theta_star_full = np.concatenate([[theta_rev_true], theta_fc_true, [theta_gs_true]])
    # Estimated theta: [theta_fc_0, ..., theta_fc_{N-1}, theta_gs]
    theta_star_est = np.concatenate([theta_fc_true, [theta_gs_true]])
    n_p = N + 1

    if is_root:
        logger.info(f"ZERO-LIMIT ESTIMATOR")
        logger.info(f"C={C}, M={M}, N={N}, n_estimated_params={n_p}")
        logger.info(f"theta_rev = {theta_rev_true} (FIXED, in error term)")
        logger.info(f"theta_fc = {theta_fc_true}")
        logger.info(f"theta_gs = {theta_gs_true}")

    # --- DGP: NO errors, deterministic ---
    obs_bundles = np.zeros((N, M), dtype=bool)
    for i in range(N):
        theta_rev_vec = np.array([theta_rev_true])
        obs_bundles[i] = greedy_demand(
            phi, theta_rev_vec, theta_fc_true[i], theta_gs_true,
            hubs[i], origin_of,
            np.zeros(M),  # NO errors
            M)

    sizes = obs_bundles.sum(axis=1)
    if is_root:
        logger.info(f"Bundle sizes: {sizes.tolist()}")

    # --- Estimation ---
    if is_root:
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
    else:
        input_data = None

    # Bounds: all theta_fc >= 0, theta_gs >= 0
    lbs = {}
    for i in range(N):
        lbs[i] = 0
    lbs[N] = 0

    model_cfg = {
        'dimensions': {
            'n_obs': N, 'n_items': M,
            'n_covariates': n_p,
            'n_simulations': 1,  # deterministic, one "simulation"
        },
        'subproblem': {
            'gurobi_params': {'TimeLimit': est_cfg['gurobi_timeout'], 'OutputFlag': 0},
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

    # --- Error oracle: gravity with coefficient 1 (deterministic) ---
    # Instead of random errors, we put gravity * theta_rev (=1) into
    # the error term. This is the "zero limit" — no noise.
    n_local = model.comm_manager.num_local_agent
    local_errors = np.zeros((n_local, M))
    for i in range(n_local):
        local_errors[i] = gravity * theta_rev_true  # = gravity
    model.features.local_modular_errors = local_errors
    model.features._error_oracle = lambda b, ids: (
        model.features.local_modular_errors[ids] * b).sum(-1)
    model.features._error_oracle_takes_data = False

    # --- Covariates oracle: only FC + congestion (no gravity) ---
    cov_oracle = build_zero_limit_covariates_oracle(N)
    model.features.set_covariates_oracle(cov_oracle)

    # --- Solver ---
    model.subproblems.load_solver(AirlineGreedySolver)
    model.subproblems.initialize_solver()

    # --- Run ---
    row_gen = model.point_estimation.n_slack
    result = row_gen.solve(initialize_solver=False, initialize_master=True,
                           verbose=True)

    t_total = time.perf_counter() - t0

    # --- Report ---
    if is_root and result is not None:
        theta_hat = result.theta_hat
        err = np.abs(theta_hat - theta_star_est)
        err_pct = err / np.abs(theta_star_est) * 100

        fc_names = [f"theta_fc_{i}" for i in range(N)]
        param_names = fc_names + ["theta_gs"]

        logger.info("")
        logger.info("=" * 70)
        logger.info(f"  ZERO-LIMIT RESULTS (C={C}, M={M}, N={N})")
        logger.info(f"  theta_rev = {theta_rev_true} (fixed)")
        logger.info("=" * 70)
        logger.info(f"  {'Param':<14} {'True':>10} {'Est':>10} {'|Err|':>10} {'Err%':>8}")
        logger.info(f"  {'-'*52}")
        for j, name in enumerate(param_names):
            logger.info(f"  {name:<14} {theta_star_est[j]:>10.6f} "
                        f"{theta_hat[j]:>10.6f} {err[j]:>10.2e} "
                        f"{err_pct[j]:>7.1f}%")
        logger.info(f"")
        logger.info(f"  Max |error|: {err.max():.2e}")
        logger.info(f"  Converged: {result.converged}")
        logger.info(f"  Iterations: {result.num_iterations}")
        logger.info(f"  Runtime: {t_total:.1f}s")

        out = {
            'theta_rev_fixed': theta_rev_true,
            'theta_true': theta_star_est.tolist(),
            'theta_hat': theta_hat.tolist(),
            'abs_error': err.tolist(),
            'max_abs_error': float(err.max()),
            'converged': result.converged,
            'iterations': result.num_iterations,
            'runtime_s': round(t_total, 2),
            'n_cities': C, 'n_edges': M, 'n_airlines': N,
        }
        out_path = BASE / 'result.json'
        with open(out_path, 'w') as f:
            json.dump(out, f, indent=2)
        logger.info(f"  Saved to {out_path}")

    return result


if __name__ == '__main__':
    run()
