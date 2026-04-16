#!/bin/env python
"""Distributed bootstrap for FCC-calibrated quadratic knapsack.

Loads DGP artifacts from data/, packs into combest's
QuadraticKnapsackGRB format, runs point estimation + bootstrap.

theta layout: [theta_mod_1..6, delta_1..delta_M, lambda]  (K = M + 7)
"""

import json
import sys
import argparse
from pathlib import Path
import numpy as np
import yaml
from scipy import sparse
import combest as ce
from combest.utils import get_logger
from combest.estimation.callbacks import adaptive_gurobi_timeout

logger = get_logger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_data import N_MODULAR, REGRESSOR_NAMES, n_params

BASE = Path(__file__).resolve().parent


def main(boot_config_path):
    with open(boot_config_path) as f:
        boot_cfg = yaml.safe_load(f)
    with open(BASE / 'config.yaml') as f:
        main_cfg = yaml.safe_load(f)

    size_name = boot_cfg['size']
    size_cfg = main_cfg['sizes'][size_name]
    N, M = size_cfg['N'], size_cfg['M']
    K = n_params(M)

    est_cfg = boot_cfg['estimation']
    bstrap = boot_cfg['bootstrap']
    callbacks = boot_cfg['callbacks']
    sigma = main_cfg['sigma']

    results_dir = BASE / 'results' / Path(boot_config_path).stem
    results_dir.mkdir(parents=True, exist_ok=True)

    model = ce.Model()

    # ---- Load DGP (root only) ----

    if model.is_root():
        obs_bundles = np.load(BASE / 'data' / 'obs_bundles.npy')
        x_modular = np.load(BASE / 'data' / 'x_modular.npy')     # (N, M, 6)
        weights = np.load(BASE / 'data' / 'weights.npy')
        capacities = np.load(BASE / 'data' / 'capacities.npy')
        Q_dense = sparse.load_npz(BASE / 'data' / 'Q_sparse.npz').toarray()

        assert obs_bundles.shape == (N, M), \
            f"Shape mismatch: {obs_bundles.shape} != ({N}, {M})"

        input_data = {
            'id_data': {
                'obs_bundles': obs_bundles,
                'modular': x_modular,                    # (N, M, 6)
                'capacity': capacities,
            },
            'item_data': {
                'modular': -np.eye(M),                   # (M, M) for delta FEs
                'quadratic': (0.5 * Q_dense)[:, :, None],  # (M, M, 1)
                'weight': weights,
            },
        }
        logger.info(f"Loaded DGP: N={N}, M={M}, K={K}")
    else:
        input_data = None

    # ---- Configure model ----

    # Theta bounds matching boot_6.yaml style
    theta_bounds = {
        'lb': -200,
        'ub': 2000,
        'lbs': {i: -2000 for i in range(N_MODULAR)},
        'ubs': {0: 3000},   # elig_pop upper
    }
    theta_bounds['lbs'][K - 1] = 0  # lambda >= 0

    model_cfg = {
        'dimensions': {
            'n_obs': N, 'n_items': M,
            'n_covariates': K,
            'n_simulations': est_cfg['n_simulations'],
        },
        'subproblem': {
            'gurobi_params': {
                'MIPGap': main_cfg['solver']['MIPGap'],
                'TimeLimit': size_cfg['gurobi_timeout'],
                'OutputFlag': 0,
            },
        },
        'row_generation': {
            'max_iters': est_cfg['max_iters'],
            'tolerance': est_cfg['tolerance'],
            'theta_bounds': theta_bounds,
        },
        'standard_errors': {
            'rowgen_tol': bstrap.get('rowgen_tol', 1e-3),
            'rowgen_max_iters': bstrap.get('rowgen_max_iters', 270),
        },
    }
    model.load_config(model_cfg)
    model.data.load_and_distribute_input_data(input_data)

    # ---- Oracles ----

    model.features.build_local_modular_error_oracle(
        seed=main_cfg['seeds']['error'], sigma=sigma)
    model.features.build_quadratic_covariates_from_data()

    # ---- Solver ----

    from combest.subproblems.registry.quadratic_obj.quadratic_knapsack import (
        QuadraticKnapsackGRBSolver,
    )
    model.subproblems.load_solver(QuadraticKnapsackGRBSolver)
    model.subproblems.initialize_solver()

    # ---- Bootstrap ----

    pt_cb, _ = adaptive_gurobi_timeout(callbacks['row_gen'])
    _, dist_cb = adaptive_gurobi_timeout(callbacks['boot'])

    def boot_callback(it, boot, master):
        dist_cb(it, boot, master)
        strip = callbacks.get('boot_strip')
        if master is not None and it == 0 and strip:
            master.strip_slack_constraints(
                percentile=strip['percentile'],
                hard_threshold=strip['hard_threshold'])

    if model.is_root():
        logger.info("=" * 60)
        logger.info(f"  Distributed bootstrap (K={K}, S={est_cfg['n_simulations']}, "
                    f"B={bstrap['num_samples']})")
        logger.info("=" * 60)

    se = model.standard_errors.compute_distributed_bootstrap(
        num_bootstrap=bstrap['num_samples'],
        seed=bstrap['seed'],
        verbose=True,
        pt_estimate_callbacks=(None, pt_cb),
        bootstrap_callback=boot_callback,
        method='bayesian',
        save_model_dir=str(results_dir / 'checkpoints'),
        load_model_dir=str(results_dir / 'checkpoints'),
    )

    # ---- Report ----

    if model.is_root() and se is not None:
        param_names = list(REGRESSOR_NAMES) + \
            [f'delta_{j+1}' for j in range(M)] + ['lambda']

        delta_star = np.load(BASE / 'data' / 'delta_star.npy')
        with open(BASE / 'results' / 'result.json') as f:
            res = json.load(f)
        theta_star = np.concatenate([
            [res['theta_star']['modular'][n] for n in REGRESSOR_NAMES],
            delta_star,
            [res['theta_star']['lambda']],
        ])

        logger.info("=" * 60)
        logger.info(f"  Results (N={N}, M={M}, K={K})")
        logger.info("=" * 60)
        logger.info(f"  {'Param':<25} {'True':>8} {'Mean':>8} {'SE':>8} {'t':>8}")
        logger.info(f"  {'-'*53}")
        for k, name in enumerate(REGRESSOR_NAMES):
            logger.info(f"  {name:<25} {theta_star[k]:>8.2f} {se.mean[k]:>8.2f} "
                        f"{se.se[k]:>8.2f} {se.t_stats[k]:>8.2f}")
        logger.info(f"  {'lambda':<25} {theta_star[-1]:>8.4f} {se.mean[-1]:>8.4f} "
                    f"{se.se[-1]:>8.4f} {se.t_stats[-1]:>8.2f}")
        logger.info(f"  (delta FEs: {M} parameters omitted)")

        n_conv = int(se.converged.sum()) if se.converged is not None else '?'
        logger.info(f"  Converged: {n_conv} / {bstrap['num_samples']}")

        out = {
            'scenario': 'quadratic_knapsack_fcc_calibrated',
            'size': {'N': N, 'M': M, 'K': K},
            'theta_true': theta_star.tolist(),
            'theta_hat': se.mean.tolist(),
            'se': se.se.tolist(),
            't_stats': se.t_stats.tolist(),
            'ci_lower': se.ci_lower.tolist(),
            'ci_upper': se.ci_upper.tolist(),
            'bootstrap_thetas': se.samples.tolist(),
            'converged': se.converged.tolist() if se.converged is not None else None,
            'param_names': param_names,
            'config': boot_cfg,
        }
        if se.u_samples is not None:
            out['bootstrap_u_hat'] = se.u_samples.tolist()

        out_path = results_dir / 'bootstrap_result.json'
        with open(out_path, 'w') as f:
            json.dump(out, f, indent=2)
        logger.info(f"  Saved -> {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', nargs='?', default='boot_config.yaml')
    main(parser.parse_args().config)
