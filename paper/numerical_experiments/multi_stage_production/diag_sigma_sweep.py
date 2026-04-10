#!/bin/env python
"""Sigma sweep: identification as a function of noise level.

For each sigma level (A-D), runs 3 seeds (42,43,44) warm-started at
theta_true (N=100, S=1, 14 params, bounds=[-100,100]).
Also evaluates f(theta_true) vs f(theta_hat) for seed 42.
"""

import sys
from pathlib import Path
import yaml
import numpy as np
import combest as ce

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_data import generate_synthetic_data
from solver import MultiStageSolver, flatten_bundle, pack_theta, THETA_NAMES
from oracles import build_oracles, N_PARAMS
from dc import DCSolver

BASE = Path(__file__).resolve().parent
with open(BASE / 'config.yaml') as f:
    CFG = yaml.safe_load(f)

SIGMA_LEVELS = {
    'A': {'phi_1': 0.001, 'phi_2': 0.001, 'nu': 0.001},
    'B': {'phi_1': 0.005, 'phi_2': 0.005, 'nu': 0.005},
    'C': {'phi_1': 0.02,  'phi_2': 0.02,  'nu': 0.01},
    'D': {'phi_1': 0.1,   'phi_2': 0.1,   'nu': 0.05},
}

N_FIRMS = 100
SEEDS = [42, 43, 44]
MAX_DC_ITERS = 10
N_PARAMS_LOCAL = len(THETA_NAMES)  # 14

DGP = dict(
    n_firms=N_FIRMS, n_markets=6, n_continents=3,
    l1_per_continent=[2, 2, 2],
    l2_per_continent=[2, 2, 2],
    n_groups_cells=2, n_platforms=4,
    models_range=[3, 8],
)


def pad_x_Q(bundles, firms, nm_max, N, L1, L2):
    nf = len(firms)
    x_Q = np.zeros((nf, nm_max, N, L1, L2))
    for i, bun in enumerate(bundles):
        nm = bun['x'].shape[0]
        x_Q[i, :nm] = bun['x']
    return x_Q


def draw_simulation_errors(model, firms, ng_max, P_max, nm_max, L1, L2, N,
                           sigma, err_seed=0):
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
        phi1[i, :ng] = rng.normal(0, sigma['phi_1'], (ng, L1))
        phi2[i, :P] = rng.normal(0, sigma['phi_2'], (P, L2))
        nu[i, :nm] = rng.normal(0, sigma['nu'], (nm, N, L1, L2))
    return phi1, phi2, nu


def run_one(seed, sigma):
    """Run DC for one (seed, sigma). Returns (result, model, solver, oracles, theta_true_vec)."""
    geo, firms, bundles, theta_true = generate_synthetic_data(
        seed=seed, dgp=DGP,
        sourcing_coefs=CFG['sourcing_coefs'],
        theta_true=CFG['theta_true'],
        sigma=sigma,
    )
    nf = len(firms)
    ng_max = max(len(f['ln_xi_1']) for f in firms)
    P_max = max(f['n_platforms'] for f in firms)
    nm_max = max(f['n_models'] for f in firms)
    L1, L2, N = geo['L1'], geo['L2'], geo['n_markets']
    n_items = ng_max * L1 + P_max * L2 + nm_max * N

    obs_bundles = np.stack([
        flatten_bundle(bun, ng_max, P_max, nm_max, L1, L2, N)
        for bun in bundles
    ])
    x_Q_global = pad_x_Q(bundles, firms, nm_max, N, L1, L2)

    model = ce.Model()
    input_data = {
        'id_data': {'obs_bundles': obs_bundles, 'firms': firms, 'x_Q': x_Q_global},
        'item_data': {
            'geo': geo, 'sourcing_coefs': CFG['sourcing_coefs'],
            'ng_max': ng_max, 'P_max': P_max, 'nm_max': nm_max,
        },
    }
    cfg = {
        'dimensions': {
            'n_obs': nf, 'n_items': n_items,
            'n_covariates': N_PARAMS, 'n_simulations': 1,
        },
        'subproblem': {'gurobi_params': {'TimeLimit': 60}},
        'row_generation': {
            'max_iters': 200, 'tolerance': 1e-6,
            'theta_bounds': {'lb': -100, 'ub': 100},
        },
    }
    model.load_config(cfg)
    model.data.load_and_distribute_input_data(input_data)

    ld = model.data.local_data
    phi1, phi2, nu = draw_simulation_errors(
        model, firms, ng_max, P_max, nm_max, L1, L2, N,
        sigma, err_seed=seed + 1000)
    ld.errors['phi1'] = phi1
    ld.errors['phi2'] = phi2
    ld.errors['nu'] = nu

    cov_oracle, err_oracle = build_oracles(model, geo, firms, ng_max, P_max, nm_max)
    model.features.set_covariates_oracle(cov_oracle)
    model.features.set_error_oracle(err_oracle)

    model.subproblems.load_solver(MultiStageSolver)
    model.subproblems.initialize_solver()

    solver = model.subproblems.subproblem_solver
    row_gen = model.point_estimation.n_slack
    theta_true_vec = pack_theta(theta_true)

    dc = DCSolver(row_gen, solver)
    result = dc.solve(theta_true_vec, max_dc_iters=MAX_DC_ITERS, tol=1e-3, verbose=False)

    return result, model, solver, cov_oracle, err_oracle, theta_true_vec


def eval_objective(solver, cov_oracle, err_oracle, model, theta_vec):
    """Evaluate f(theta) = sol_term - obs_term."""
    ld = model.data.local_data
    local_obs = model.data.local_obs_bundles
    n = model.comm_manager.num_local_agent
    ids = np.arange(n)

    solver.set_q_linearization(theta_vec)
    solver_bundles = solver.solve(theta_vec)

    cov_obs = cov_oracle(local_obs, ids, ld)
    err_obs = err_oracle(local_obs, ids, ld)
    obs_term = (cov_obs @ theta_vec + err_obs).sum()

    cov_v = cov_oracle(solver_bundles, ids, ld)
    err_v = err_oracle(solver_bundles, ids, ld)
    sol_term = (cov_v @ theta_vec + err_v).sum()

    return sol_term - obs_term


# ─── Main sweep ──────────────────────────────────────────────────────────────

all_estimates = {}   # level -> list of theta_hat arrays (nan if failed)
all_f = {}           # level -> (f_true, f_hat) for seed 42
all_times = {}       # level -> list of times

for level, sigma in SIGMA_LEVELS.items():
    print(f"\n{'#'*70}", flush=True)
    print(f"  SIGMA LEVEL {level}: phi_1={sigma['phi_1']}, phi_2={sigma['phi_2']}, "
          f"nu={sigma['nu']}", flush=True)
    print(f"{'#'*70}", flush=True)

    estimates = []
    times = []
    f_info = None

    for seed in SEEDS:
        print(f"\n  --- seed={seed} ---", flush=True)
        try:
            result, model, solver, cov_oracle, err_oracle, theta_true_vec = run_one(seed, sigma)
            if result is not None:
                estimates.append(result.theta_hat)
                times.append(result.total_time)
                print(f"  converged={result.converged}, iters={result.num_iterations}, "
                      f"time={result.total_time:.1f}s", flush=True)
                if seed == 42:
                    f_true = eval_objective(solver, cov_oracle, err_oracle, model, theta_true_vec)
                    f_hat = eval_objective(solver, cov_oracle, err_oracle, model, result.theta_hat)
                    f_info = (f_true, f_hat)
                    print(f"  f(theta_true)={f_true:.6f}  f(theta_hat)={f_hat:.6f}  "
                          f"diff={f_hat - f_true:.6f}", flush=True)
            else:
                estimates.append(np.full(N_PARAMS_LOCAL, np.nan))
                times.append(np.nan)
                print(f"  FAILED (result=None)", flush=True)
        except Exception as e:
            estimates.append(np.full(N_PARAMS_LOCAL, np.nan))
            times.append(np.nan)
            print(f"  EXCEPTION: {e}", flush=True)

    all_estimates[level] = estimates
    all_f[level] = f_info
    all_times[level] = times

# ─── Summary tables ──────────────────────────────────────────────────────────

theta_true_vec = pack_theta(CFG['theta_true'])

print(f"\n\n{'='*80}")
print(f"  SIGMA SWEEP SUMMARY  (N={N_FIRMS}, S=1, init=theta_true, seeds={SEEDS})")
print(f"{'='*80}")

for level, sigma in SIGMA_LEVELS.items():
    ests = all_estimates[level]
    valid = [e for e in ests if not np.any(np.isnan(e))]
    arr = np.array(valid) if valid else np.full((1, N_PARAMS_LOCAL), np.nan)
    times = [t for t in all_times[level] if not np.isnan(t)]
    avg_time = np.mean(times) if times else np.nan

    print(f"\n  Level {level}: phi_1={sigma['phi_1']}, phi_2={sigma['phi_2']}, "
          f"nu={sigma['nu']}   ({len(valid)}/{len(SEEDS)} seeds OK, "
          f"avg_time={avg_time:.1f}s)")
    print(f"  {'Parameter':<15} {'True':>8} {'Mean':>8} {'Std':>8} {'MeanErr%':>10}")
    print(f"  {'-'*53}")
    for j, name in enumerate(THETA_NAMES):
        t = theta_true_vec[j]
        m = arr[:, j].mean()
        s = arr[:, j].std()
        err = 100 * (m - t) / abs(t) if t != 0 else float('nan')
        print(f"  {name:<15} {t:>8.4f} {m:>8.4f} {s:>8.4f} {err:>9.1f}%")

    f_info = all_f[level]
    if f_info is not None:
        f_true, f_hat = f_info
        print(f"\n  Seed 42 objective:  f(theta_true)={f_true:.6f}  "
              f"f(theta_hat)={f_hat:.6f}  "
              f"diff={f_hat - f_true:.6f}  "
              f"f_hat<f_true: {f_hat < f_true}")
    else:
        print(f"\n  Seed 42 objective:  N/A")

print()
