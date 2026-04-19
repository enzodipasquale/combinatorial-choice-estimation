#!/bin/env python
"""Row generation estimation for multi-stage production (linear parametrization)."""

import sys
import argparse
from pathlib import Path
import yaml
import numpy as np
import combest as ce
from combest.utils import get_logger

logger = get_logger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_data import generate_synthetic_data
from solver import MultiStageSolver, pack_theta, THETA_NAMES
from oracles import build_oracles, N_PARAMS
from combest.estimation.callbacks import adaptive_gurobi_timeout

BASE = Path(__file__).resolve().parent
with open(BASE / 'config.yaml') as f:
    CFG = yaml.safe_load(f)


def draw_errors(model, firms, ng_max, P_max, nm_max, L1, L2, N,
                sigma, err_seed=0):
    """Draw per-(agent, simulation) errors using agent_ids as seeds."""
    n_local = model.comm_manager.num_local_agent
    obs_ids = model.comm_manager.obs_ids

    phi1 = np.zeros((n_local, ng_max, L1))
    phi2 = np.zeros((n_local, P_max, L2))
    nu = np.zeros((n_local, nm_max, N, L1, L2))

    for i, gid in enumerate(model.comm_manager.agent_ids):
        oid = obs_ids[i]
        firm = firms[oid]
        ng = len(firm['ln_xi_1'])
        P = firm['n_platforms']
        nm = firm['n_models']
        rng = np.random.default_rng((err_seed, int(gid)))
        phi1[i, :ng] = rng.normal(0, sigma['phi'], (ng, L1))
        phi2[i, :P] = rng.normal(0, sigma['phi'], (P, L2))
        nu[i, :nm] = rng.normal(0, sigma['nu'], (nm, N, L1, L2))

    return phi1, phi2, nu


def load_dgp_errors(model, dgp_errors, ng_max, P_max, nm_max, L1, L2, N):
    """Load DGP errors into combest's error arrays (S=1, so agent_id == obs_id)."""
    n_local = model.comm_manager.num_local_agent
    obs_ids = model.comm_manager.obs_ids

    phi1 = np.zeros((n_local, ng_max, L1))
    phi2 = np.zeros((n_local, P_max, L2))
    nu = np.zeros((n_local, nm_max, N, L1, L2))

    for i in range(n_local):
        oid = obs_ids[i]
        err = dgp_errors[oid]
        ng, P = err['phi1'].shape[0], err['phi2'].shape[0]
        nm = err['nu'].shape[0]
        phi1[i, :ng] = err['phi1']
        phi2[i, :P] = err['phi2']
        nu[i, :nm] = err['nu']

    return phi1, phi2, nu


def run(dgp_cfg, seed=42, n_simulations=1, verbose=True):

    model = ce.Model()

    # Generate DGP on root only (no Gurobi — just geography, firms, errors)
    if model.is_root():
        geo, firms, dgp_errors, theta_true = generate_synthetic_data(
            seed=seed, dgp=dgp_cfg,
            sourcing_coefs=None,
            theta_true=CFG['theta_true'],
            sigma=CFG['sigma'],
        )

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

    # --- Phase 1: Solve DGP at theta_true using combest (S=1) ---
    dgp_combest_cfg = {
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
    model.load_config(dgp_combest_cfg)
    model.data.load_and_distribute_input_data(input_data)

    ld = model.data.local_data
    geo = ld.item_data['geo']
    firms = ld.id_data['firms']
    dgp_errors = ld.id_data['dgp_errors']
    theta_true = CFG['theta_true']
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
        logger.info("Generating observed bundles at theta_true via combest solver...")
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

    # --- Phase 2: Row generation estimation ---
    est_cfg = {
        'dimensions': {
            'n_obs': nf, 'n_items': n_items,
            'n_covariates': N_PARAMS, 'n_simulations': n_simulations,
        },
        'subproblem': {},
        'row_generation': {
            'max_iters': CFG['estimation'].get('max_iters', 200),
            'tolerance': CFG['estimation'].get('tolerance', 1e-6),
            'theta_bounds': {
                'lb': 0, 'ub': 10,
                'lbs': {6: -5, 7: -5, 8: -5, 9: -5},
                'ubs': {6: 5, 7: 5, 8: 5, 9: 5},
            },
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
        CFG['sigma'], err_seed=seed + 1000)
    ld2.errors['phi1'] = phi1
    ld2.errors['phi2'] = phi2
    ld2.errors['nu'] = nu

    cov_oracle, err_oracle = build_oracles(
        model2, geo, firms, ng_max, P_max, nm_max)
    model2.features.set_covariates_oracle(cov_oracle)
    model2.features.set_error_oracle(err_oracle)

    model2.subproblems.load_solver(MultiStageSolver)
    model2.subproblems.initialize_solver()

    row_gen = model2.point_estimation.n_slack

    if model2.is_root():
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"  Row generation ({N_PARAMS} params, S={n_simulations})")
        logger.info("=" * 60)

    pt_schedule = CFG['estimation'].get('gurobi_timeout_schedule', [
        {'iters': 20, 'timeout': 5, 'retire': True},
        {'timeout': 30, 'retire': True},
    ])
    pt_callback, _ = adaptive_gurobi_timeout(pt_schedule)

    result = row_gen.solve(
        initialize_solver=False,
        initialize_master=True,
        iteration_callback=pt_callback,
        verbose=verbose)

    if model2.is_root() and result is not None:
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"  Results (N={nf}, S={n_simulations}, seed={seed})")
        logger.info("=" * 60)
        logger.info(f"  {'Parameter':<15} {'True':>10} {'Estimated':>10} {'Error%':>10}")
        logger.info(f"  {'-'*45}")
        for j, name in enumerate(THETA_NAMES):
            true_val = theta_true[name]
            est_val = result.theta_hat[j]
            err_pct = 100 * (est_val - true_val) / abs(true_val) if true_val != 0 else float('nan')
            logger.info(f"  {name:<15} {true_val:>10.4f} {est_val:>10.4f} {err_pct:>9.1f}%")
        logger.info(f"")
        logger.info(f"  Converged: {result.converged}  "
                    f"Iters: {result.num_iterations}  "
                    f"Time: {result.total_time:.1f}s")

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-firms', type=int, default=CFG['dgp']['n_firms'])
    parser.add_argument('--seed', type=int,
                        default=CFG.get('monte_carlo', {}).get('seed', 42))
    parser.add_argument('--n-simulations', type=int,
                        default=CFG['estimation']['n_simulations'])
    args = parser.parse_args()

    dgp = dict(CFG['dgp'])
    dgp['n_firms'] = args.n_firms

    run(dgp, seed=args.seed, n_simulations=args.n_simulations)
