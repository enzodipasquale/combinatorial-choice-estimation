#!/bin/env python
"""Bootstrap for multi-stage production — distributed or serial.

Mode is selected by `bootstrap.method` in the config YAML:
  - "distributed": calls combest.standard_errors.compute_distributed_bootstrap.
        Requires num_bootstrap <= comm_size (one sample per rank). HPC path.
  - "serial":      calls combest.standard_errors.compute_bootstrap.
        Samples run sequentially, each warm-started from the converged
        point-estimate master. Agents are parallelized across ranks within
        each sample. Use when B > comm_size (e.g. local 4-8 rank setup
        with B=50-150).

Phases 1 and 2 (DGP solve at theta_true + point estimation) are identical
across modes.
"""

import json
import sys
import argparse
from pathlib import Path
import yaml
import numpy as np
import combest as ce
from combest.utils import get_logger
from combest.estimation.callbacks import adaptive_gurobi_timeout

logger = get_logger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_data import generate_synthetic_data
from solver import MultiStageSolver, pack_theta, THETA_NAMES
from oracles import build_oracles, N_PARAMS
from run_experiment import draw_errors, load_dgp_errors

BASE = Path(__file__).resolve().parent


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    dgp_cfg = cfg['dgp']
    theta_true = cfg['theta_true']
    sigma = cfg['sigma']
    seed = cfg.get('monte_carlo', {}).get('seed', 42)
    n_simulations = cfg['estimation']['n_simulations']
    boot_cfg = cfg.get('bootstrap', {})
    callbacks = cfg.get('callbacks', {})
    method = boot_cfg.get('method', 'distributed')
    if method not in ('distributed', 'serial'):
        raise ValueError(f"bootstrap.method must be 'distributed' or 'serial', "
                         f"got {method!r}")

    results_dir = BASE / 'results' / Path(config_path).stem
    results_dir.mkdir(parents=True, exist_ok=True)

    model = ce.Model()

    # ---- Phase 1: Generate DGP and solve at theta_true ----

    if model.is_root():
        geo, firms, dgp_errors, _ = generate_synthetic_data(
            seed=seed, dgp=dgp_cfg, sourcing_coefs=None,
            theta_true=theta_true, sigma=sigma)

        nf = len(firms)
        ng_max = max(len(f['ln_xi_1']) for f in firms)
        P_max = max(f['n_platforms'] for f in firms)
        nm_max = max(f['n_models'] for f in firms)
        L1, L2, N = geo['L1'], geo['L2'], geo['n_markets']
        n_items = ng_max * L1 + P_max * L2 + nm_max * N

        input_data = {
            'id_data': {
                'obs_bundles': np.zeros((nf, n_items), dtype=bool),
                'firms': firms, 'dgp_errors': dgp_errors,
                'x_Q': np.zeros((nf, nm_max, N, L1, L2)),
            },
            'item_data': {
                'geo': geo,
                'ng_max': ng_max, 'P_max': P_max, 'nm_max': nm_max,
            },
        }
    else:
        input_data = None
        nf, n_items = 0, 0

    dgp_cfg_combest = {
        'dimensions': {
            'n_obs': nf, 'n_items': n_items,
            'n_covariates': N_PARAMS, 'n_simulations': 1,
        },
        'subproblem': {},
        'row_generation': {
            'max_iters': 200, 'tolerance': 1e-6,
            'theta_bounds': {
                'lb': 0, 'ub': 10,
                'lbs': {6: -5, 7: -5, 8: -5, 9: -5},
                'ubs': {6: 5, 7: 5, 8: 5, 9: 5},
            },
        },
    }
    model.load_config(dgp_cfg_combest)
    model.data.load_and_distribute_input_data(input_data)

    ld = model.data.local_data
    geo = ld.item_data['geo']
    firms = ld.id_data['firms']
    dgp_errors = ld.id_data['dgp_errors']
    ng_max = ld.item_data['ng_max']
    P_max = ld.item_data['P_max']
    nm_max = ld.item_data['nm_max']
    L1, L2, N = geo['L1'], geo['L2'], geo['n_markets']
    n_items = ng_max * L1 + P_max * L2 + nm_max * N

    phi1_dgp, phi2_dgp, nu_dgp = load_dgp_errors(
        model, dgp_errors, ng_max, P_max, nm_max, L1, L2, N)
    ld.errors['phi1'] = phi1_dgp
    ld.errors['phi2'] = phi2_dgp
    ld.errors['nu'] = nu_dgp

    cov_oracle, err_oracle = build_oracles(
        model, geo, firms, ng_max, P_max, nm_max)
    model.features.set_covariates_oracle(cov_oracle)
    model.features.set_error_oracle(err_oracle)

    model.subproblems.load_solver(MultiStageSolver)
    model.subproblems.initialize_solver()

    theta_true_vec = pack_theta(theta_true)

    if model.is_root():
        logger.info("Phase 1: generating observed bundles at theta_true...")
    obs_bundles_local = model.subproblems.subproblem_solver.solve(theta_true_vec)

    comm = model.comm_manager
    obs_bundles_global = comm.Gatherv_by_row(obs_bundles_local)
    x_Q_local = ld.id_data['policies']['x_V'].copy()
    x_Q_global = comm.Gatherv_by_row(x_Q_local)

    if model.is_root():
        active_mask = obs_bundles_global.any(axis=1)
        n_active = active_mask.sum()
        if n_active < len(firms):
            logger.info(f"Filtering: {len(firms) - n_active} opt-out firms removed")
            firms = [f for f, a in zip(firms, active_mask) if a]
            dgp_errors = [e for e, a in zip(dgp_errors, active_mask) if a]
            obs_bundles_global = obs_bundles_global[active_mask]
            x_Q_global = x_Q_global[active_mask]

        nf = len(firms)
        input_data = {
            'id_data': {
                'obs_bundles': obs_bundles_global, 'firms': firms,
                'dgp_errors': dgp_errors, 'x_Q': x_Q_global,
            },
            'item_data': {
                'geo': geo,
                'ng_max': ng_max, 'P_max': P_max, 'nm_max': nm_max,
            },
        }
    else:
        input_data = None
        nf = 0

    # ---- Phase 2: Set up estimation model with S simulations ----

    est_cfg = {
        'dimensions': {
            'n_obs': nf, 'n_items': n_items,
            'n_covariates': N_PARAMS, 'n_simulations': n_simulations,
        },
        'subproblem': {},
        'row_generation': {
            'max_iters': cfg['estimation'].get('max_iters', 200),
            'tolerance': cfg['estimation'].get('tolerance', 1e-6),
            'theta_bounds': {
                'lb': 0, 'ub': 10,
                'lbs': {6: -5, 7: -5, 8: -5, 9: -5},
                'ubs': {6: 5, 7: 5, 8: 5, 9: 5},
            },
        },
        'standard_errors': {
            'rowgen_tol': boot_cfg.get('rowgen_tol', 1e-3),
            'rowgen_max_iters': boot_cfg.get('rowgen_max_iters',
                                             200 if method == 'distributed' else 100),
        },
    }

    model2 = ce.Model()
    model2.load_config(est_cfg)
    model2.data.load_and_distribute_input_data(input_data)

    ld2 = model2.data.local_data
    geo = ld2.item_data['geo']
    firms = ld2.id_data['firms']
    ng_max = ld2.item_data['ng_max']
    P_max = ld2.item_data['P_max']
    nm_max = ld2.item_data['nm_max']
    L1, L2, N = geo['L1'], geo['L2'], geo['n_markets']

    phi1, phi2, nu = draw_errors(
        model2, firms, ng_max, P_max, nm_max, L1, L2, N,
        sigma, err_seed=seed + 1000)
    ld2.errors['phi1'] = phi1
    ld2.errors['phi2'] = phi2
    ld2.errors['nu'] = nu

    cov_oracle, err_oracle = build_oracles(
        model2, geo, firms, ng_max, P_max, nm_max)
    model2.features.set_covariates_oracle(cov_oracle)
    model2.features.set_error_oracle(err_oracle)

    model2.subproblems.load_solver(MultiStageSolver)
    model2.subproblems.initialize_solver()

    # ---- Phase 3: Point estimation + bootstrap ----

    pt_cb, _ = adaptive_gurobi_timeout(callbacks['row_gen'])

    if model2.is_root():
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"  {method.capitalize()} bootstrap (K={N_PARAMS}, "
                    f"S={n_simulations}, B={boot_cfg.get('num_samples', 150)})")
        logger.info("=" * 60)

    if method == 'distributed':
        _, dist_cb = adaptive_gurobi_timeout(callbacks['boot'])

        def boot_callback(it, boot, master):
            dist_cb(it, boot, master)
            strip = callbacks.get('boot_strip')
            if master is not None and it == 0 and strip:
                master.strip_slack_constraints(
                    percentile=strip['percentile'],
                    hard_threshold=strip['hard_threshold'])

        se = model2.standard_errors.compute_distributed_bootstrap(
            num_bootstrap=boot_cfg.get('num_samples', 150),
            seed=boot_cfg.get('seed', 7777),
            verbose=True,
            pt_estimate_callbacks=(None, pt_cb),
            bootstrap_callback=boot_callback,
            method='bayesian',
            save_model_dir=str(results_dir / 'checkpoints'),
            load_model_dir=str(results_dir / 'checkpoints'),
        )
    else:  # serial
        # Serial uses a single pt_cb for both point estimation and each
        # bootstrap sample. The iteration counter resets per row_gen.solve()
        # call, so the timeout schedule applies fresh to every sample
        # (warm-started from the base model's cuts).
        se = model2.standard_errors.compute_bootstrap(
            num_bootstrap=boot_cfg.get('num_samples', 150),
            seed=boot_cfg.get('seed', 7777),
            verbose=True,
            pt_estimate_callbacks=(None, pt_cb),
            method='bayesian',
            checkpoint_dir=str(results_dir),
            checkpoint_every=boot_cfg.get('checkpoint_every', 5),
        )

    if model2.is_root() and se is not None:
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"  Bootstrap results (N={nf}, S={n_simulations}, "
                    f"B={boot_cfg.get('num_samples', 150)})")
        logger.info("=" * 60)
        logger.info(f"  {'Param':<12} {'True':>8} {'Mean':>8} {'SE':>8} "
                    f"{'t-stat':>8} {'CI_lo':>8} {'CI_hi':>8}")
        logger.info(f"  {'-'*60}")
        for j, name in enumerate(THETA_NAMES):
            true_val = theta_true[name]
            logger.info(f"  {name:<12} {true_val:>8.4f} {se.mean[j]:>8.4f} "
                        f"{se.se[j]:>8.4f} {se.t_stats[j]:>8.2f} "
                        f"{se.ci_lower[j]:>8.4f} {se.ci_upper[j]:>8.4f}")
        n_conv = se.converged.sum() if se.converged is not None else '?'
        logger.info(f"\n  Converged samples: {n_conv}")

        out = {
            'theta_true': {name: theta_true[name] for name in THETA_NAMES},
            'theta_hat': se.mean.tolist(),
            'se': se.se.tolist(),
            't_stats': se.t_stats.tolist(),
            'ci_lower': se.ci_lower.tolist(),
            'ci_upper': se.ci_upper.tolist(),
            'bootstrap_thetas': se.samples.tolist(),
            'converged': se.converged.tolist(),
            'param_names': THETA_NAMES,
            'method': method,
            'config': cfg,
        }
        if se.u_samples is not None:
            out['bootstrap_u_hat'] = se.u_samples.tolist()

        out_path = results_dir / 'bootstrap_result.json'
        json.dump(out, open(out_path, 'w'), indent=2)
        logger.info(f"  Saved -> {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to bootstrap config YAML')
    main(parser.parse_args().config)
