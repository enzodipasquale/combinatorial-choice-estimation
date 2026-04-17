"""Airline with item FEs and endogenous gravity index.

DGP:
  geo_j = (pop_o * pop_d / dist)  standardized        (exogenous geography)
  z_j ~ N(0,1)^K_z                                     (instruments)
  xi_j ~ N(0,1)                                        (unobserved quality)
  grav_index_j = geo_j + alpha_1*(z_j@1) + alpha_2*xi_j  (endogenous)
  delta_j = beta * grav_index_j + xi_j                 (route value)

  V_i(b) = sum_j b_j*delta_j - theta_fc*|b| - theta_gs*congestion + nu_ij

Second stage: delta_hat = a + beta * grav_index + xi
  OLS biased (grav_index contains xi via alpha_2)
  2SLS with z corrects (z independent of xi)
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
from generate_data import (build_geography, build_edges, build_hubs, greedy_demand)

BASE = Path(__file__).resolve().parent
PARENT = BASE.parent
with open(PARENT / 'config.yaml') as f:
    CFG = yaml.safe_load(f)


def make_endog_find_best_item():
    def find_best_item(local_id, bundle, items_left, theta, best_val,
                       data, modular_error, cache=None):
        if cache is None:
            cache = {}
        if 'base' not in cache:
            M = data.item_data['M']
            origin_of = data.item_data['origin_of']
            obs_ids = data.item_data.get('obs_ids', None)
            obs_idx = int(obs_ids[local_id]) if obs_ids is not None else local_id
            hubs_i = data.id_data['hubs'][obs_idx]
            cache['base'] = theta[:M] - theta[M] + modular_error
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
        if marginals[best_item] <= 0:
            return -1, best_val
        for h, mask in hub_masks.items():
            if mask[best_item]:
                hub_counts[h] += 1
                break
        return best_item, best_val + marginals[best_item]
    return find_best_item


class EndogGreedySolver(GreedySolver):
    find_best_item = staticmethod(make_endog_find_best_item())


def build_endog_covariates_oracle(M):
    def covariates_oracle(bundles, ids, data):
        origin_of = data.item_data['origin_of']
        hubs_list = data.id_data['hubs']
        obs_ids = data.item_data.get('obs_ids', None)
        n_agents = bundles.shape[0]
        features = np.zeros((n_agents, M + 2))
        for i_local, idx in enumerate(ids):
            obs_idx = int(obs_ids[idx]) if obs_ids is not None else idx
            b = bundles[i_local]
            features[i_local, :M] = b.astype(float)
            features[i_local, M] = -float(b.sum())
            hubs_i = hubs_list[obs_idx]
            cong = sum(int((b & (origin_of == h)).sum())**2 for h in hubs_i)
            features[i_local, M+1] = -float(cong)
        return features
    return covariates_oracle


def run(sigma=1.0, alpha_1=1.0, alpha_2=0.5, beta=0.8,
        C=10, N=200, K_z=5):
    cfg = CFG
    est_cfg = cfg['estimation']
    M = C * (C - 1)
    t0 = time.perf_counter()

    # ---- Geography ----
    rng_dgp = np.random.default_rng(42)
    _, origin_of, dest_of, _ = build_edges(C)
    locations, dists, populations = build_geography(
        C, cfg['dgp']['pop_log_std'], rng_dgp)
    hubs = build_hubs(N, C, populations,
                      cfg['dgp']['min_hubs'], cfg['dgp']['max_hubs'],
                      cfg['dgp']['hub_pool_frac'], rng_dgp)

    d_safe = np.maximum(dists[origin_of, dest_of], 1e-6)
    geo_raw = populations[origin_of] * populations[dest_of] / d_safe
    geo = (geo_raw - geo_raw.mean()) / max(geo_raw.std(), 1e-8)

    # ---- Instruments and unobserved quality ----
    rng_blp = np.random.default_rng(77)
    z = rng_blp.normal(0, 1, (M, K_z))
    xi = rng_blp.normal(0, 1, M)

    # ---- Endogenous gravity index ----
    alpha_z = alpha_1 * np.ones(K_z)
    grav_index = geo + z @ alpha_z + alpha_2 * xi

    # ---- Delta ----
    delta_true = beta * grav_index + xi
    delta_true -= delta_true.mean()

    # ---- Parameters ----
    theta_fc_true = 0.3
    theta_gs_true = 0.5
    K = M + 2

    logger.info(f"ENDOGENEITY: C={C}, M={M}, N={N}, sigma={sigma}")
    logger.info(f"alpha_1={alpha_1}, alpha_2={alpha_2}, beta={beta}, K_z={K_z}")
    logger.info(f"delta std={delta_true.std():.2f}, grav_index std={grav_index.std():.2f}")

    # ---- DGP errors ----
    rng_err = np.random.default_rng(1041)
    nu = rng_err.normal(0, sigma, (N, M))

    # ---- Observed bundles ----
    obs_bundles = np.zeros((N, M), dtype=bool)
    for i in range(N):
        obs_bundles[i] = greedy_demand(
            delta_true[:, None], np.array([1.0]),
            theta_fc_true, theta_gs_true,
            hubs[i], origin_of, nu[i], M)

    sizes = obs_bundles.sum(axis=1)
    route_counts = obs_bundles.sum(axis=0)
    n_var = ((route_counts > 0) & (route_counts < N)).sum()
    logger.info(f"Bundles: min={sizes.min()}, max={sizes.max()}, mean={sizes.mean():.0f}")
    logger.info(f"Routes with variation: {n_var}/{M}")

    if sizes.min() == 0 or sizes.max() == M:
        logger.info("DEGENERATE"); return None

    # ---- First stage: combest ----
    model = ce.Model()
    input_data = {
        'id_data': {'obs_bundles': obs_bundles,
                    'hubs': [list(h) for h in hubs]},
        'item_data': {'origin_of': origin_of, 'M': M, 'N_firms': N},
    }

    model_cfg = {
        'dimensions': {'n_obs': N, 'n_items': M, 'n_covariates': K,
                       'n_simulations': est_cfg['n_simulations']},
        'subproblem': {'gurobi_params': {'TimeLimit': est_cfg['gurobi_timeout'],
                                         'OutputFlag': 0}},
        'row_generation': {'max_iters': 300, 'tolerance': est_cfg['tolerance'],
                           'theta_bounds': {'lb': -20, 'ub': 20,
                                            'lbs': {M+1: 0}}},
    }
    model.load_config(model_cfg)
    model.data.load_and_distribute_input_data(input_data)

    ld = model.data.local_data
    ld.id_data['hubs'] = [set(h) for h in ld.id_data['hubs']]
    ld.item_data['obs_ids'] = model.comm_manager.obs_ids

    model.features.build_local_modular_error_oracle(seed=2042, sigma=sigma)
    model.features.set_covariates_oracle(build_endog_covariates_oracle(M))
    model.subproblems.load_solver(EndogGreedySolver)
    model.subproblems.initialize_solver()

    result = model.point_estimation.n_slack.solve(
        initialize_solver=False, initialize_master=True, verbose=True)

    t_first = time.perf_counter() - t0
    if result is None: return None

    delta_hat = result.theta_hat[:M]
    gs_hat = result.theta_hat[M+1]

    delta_hat_dm = delta_hat - delta_hat.mean()
    delta_true_dm = delta_true - delta_true.mean()
    delta_corr = np.corrcoef(delta_hat_dm, delta_true_dm)[0, 1]
    delta_rmse = np.sqrt(((delta_hat_dm - delta_true_dm)**2).mean())

    logger.info(f"\n{'='*70}")
    logger.info(f"  FIRST STAGE ({t_first:.1f}s, {result.num_iterations} iters)")
    logger.info(f"{'='*70}")
    logger.info(f"  delta: corr={delta_corr:.4f}, RMSE(dm)={delta_rmse:.4f}")
    logger.info(f"  GS: true={theta_gs_true:.4f}, hat={gs_hat:.4f}")

    # ---- Second stage: delta = a + beta * grav_index + xi ----
    # OLS
    X = np.column_stack([np.ones(M), grav_index])
    beta_ols = np.linalg.lstsq(X, delta_hat_dm, rcond=None)[0][1]

    # 2SLS
    Z = np.column_stack([np.ones(M), z])
    pi_hat = np.linalg.lstsq(Z, grav_index, rcond=None)[0]
    grav_hat = Z @ pi_hat
    r2 = 1 - ((grav_index - grav_hat)**2).sum() / \
             ((grav_index - grav_index.mean())**2).sum()

    X2 = np.column_stack([np.ones(M), grav_hat])
    beta_2sls = np.linalg.lstsq(X2, delta_hat_dm, rcond=None)[0][1]

    # On true delta
    beta_ols_true = np.linalg.lstsq(X, delta_true_dm, rcond=None)[0][1]
    beta_2sls_true = np.linalg.lstsq(X2, delta_true_dm, rcond=None)[0][1]

    logger.info(f"\n{'='*70}")
    logger.info(f"  SECOND STAGE: delta = a + beta*grav_index + xi")
    logger.info(f"{'='*70}")
    logger.info(f"  First stage R²:     {r2:.4f}")
    logger.info(f"  beta (true):        {beta:.4f}")
    logger.info(f"  On TRUE delta:")
    logger.info(f"    OLS:  {beta_ols_true:.4f} (bias={beta_ols_true-beta:+.4f})")
    logger.info(f"    2SLS: {beta_2sls_true:.4f} (bias={beta_2sls_true-beta:+.4f})")
    logger.info(f"  On ESTIMATED delta:")
    logger.info(f"    OLS:  {beta_ols:.4f} (bias={beta_ols-beta:+.4f})")
    logger.info(f"    2SLS: {beta_2sls:.4f} (bias={beta_2sls-beta:+.4f})")
    logger.info(f"  Endogeneity correction: {abs(beta_ols - beta_2sls):.4f}")

    out = {
        'sigma': sigma, 'alpha_1': alpha_1, 'alpha_2': alpha_2,
        'beta': beta, 'C': C, 'M': M, 'N': N, 'K_z': K_z,
        'delta_corr': float(delta_corr), 'delta_rmse': float(delta_rmse),
        'gs_hat': float(gs_hat), 'gs_true': theta_gs_true,
        'first_stage_r2': float(r2),
        'beta_ols_true': float(beta_ols_true),
        'beta_2sls_true': float(beta_2sls_true),
        'beta_ols': float(beta_ols), 'beta_2sls': float(beta_2sls),
        'converged': result.converged,
        'runtime_s': round(t_first, 1),
    }
    out_path = BASE / 'result.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    logger.info(f"  Saved to {out_path}")
    return out


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--alpha-1', type=float, default=1.0)
    parser.add_argument('--alpha-2', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.8)
    parser.add_argument('--C', type=int, default=10)
    parser.add_argument('--N', type=int, default=200)
    parser.add_argument('--K-z', type=int, default=5)
    args = parser.parse_args()
    run(sigma=args.sigma, alpha_1=args.alpha_1, alpha_2=args.alpha_2,
        beta=args.beta, C=args.C, N=args.N, K_z=args.K_z)
