"""DGP generation and validation for the quadratic knapsack / auction scenario.

No estimation is run. The deliverable is:
  1. Healthy DGP at showcase size (N=250, M=500).
  2. Brute-force verification at tiny size (N=10, M=15).
  3. 2SLS smoke test on delta_star.
  4. result.json + data artifacts.
"""

import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
import yaml
from scipy import sparse
from combest.utils import get_logger

logger = get_logger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_data import generate_data
from oracle import brute_force_demand, compute_utility, twosls_smoke_test

BASE = Path(__file__).resolve().parent
with open(BASE / 'config.yaml') as f:
    CFG = yaml.safe_load(f)


def run(cfg=None, size_name='pilot'):
    if cfg is None:
        cfg = CFG

    size_cfg = cfg['sizes'][size_name]
    dgp_cfg = cfg['dgp']
    healthy_cfg = cfg['healthy_dgp']
    solver_cfg = cfg['solver']
    seeds = cfg['seeds']
    defaults = cfg.get('defaults', {})

    # Override default alpha into dgp_cfg for generate_data
    dgp_cfg_full = dict(dgp_cfg)
    dgp_cfg_full['alpha'] = defaults.get('alpha', 0.1)

    N = size_cfg['N']
    M = size_cfg['M']

    t0 = time.perf_counter()

    # --- Generate healthy DGP ---
    logger.info(f"Generating DGP: size={size_name}, N={N}, M={M}")
    theta_star, bundles, dgp_data, diag = generate_data(
        size_cfg, dgp_cfg_full, healthy_cfg, solver_cfg, seeds)

    alpha = theta_star[0]
    delta = theta_star[1:M + 1]
    lambda_ = theta_star[-1]

    logger.info(f"DGP generated: alpha={alpha:.4f}, lambda={lambda_:.4f}, "
                f"delta std={delta.std():.3f}")
    logger.info(f"Healthy checks: {diag}")

    # --- Brute-force verification (tiny size only) ---
    bf_result = None
    if M <= 20:
        n_verify = min(cfg.get('brute_force', {}).get('n_agents_verify', 3), N)
        logger.info(f"Running brute-force verification for {n_verify} agents (M={M})...")
        bf_match = True
        for i in range(n_verify):
            bf_bundle, bf_val = brute_force_demand(
                dgp_data['x'][i], delta, dgp_data['Q_dense'],
                lambda_, alpha, dgp_data['errors'][i],
                dgp_data['weights'], dgp_data['capacities'][i], M)
            qkp_val = compute_utility(
                bundles[i], dgp_data['x'][i], delta, dgp_data['Q_dense'],
                lambda_, alpha, dgp_data['errors'][i])

            match = np.abs(bf_val - qkp_val) < 1e-6
            if not match:
                logger.warning(f"  Agent {i}: BF val={bf_val:.6f} vs QKP val={qkp_val:.6f} MISMATCH")
                bf_match = False
            else:
                logger.info(f"  Agent {i}: match (val={qkp_val:.6f})")

        bf_result = bf_match
        logger.info(f"Brute-force verification: {'PASS' if bf_match else 'FAIL'}")

    # --- 2SLS smoke test ---
    blp = dgp_data['blp']
    beta_hat, max_2sls_err = twosls_smoke_test(
        blp['delta'], blp['phi'], blp['z'], blp['beta_star'])
    # With M observations, 2SLS has finite-sample bias.
    # Tolerance scales with 1/sqrt(M): generous for smoke test.
    twosls_tol = 5.0 / np.sqrt(M)
    twosls_ok = max_2sls_err < twosls_tol
    logger.info(f"2SLS smoke test: max|beta_hat - beta_star| = {max_2sls_err:.2e} "
                f"({'PASS' if twosls_ok else 'FAIL'})")

    t_total = time.perf_counter() - t0

    # --- Save data artifacts ---
    data_dir = BASE / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(data_dir / 'phi.npy', blp['phi'])
    np.save(data_dir / 'z.npy', blp['z'])
    np.save(data_dir / 'xi.npy', blp['xi'])
    np.save(data_dir / 'delta_star.npy', blp['delta'])
    np.save(data_dir / 'beta_star.npy', blp['beta_star'])
    np.save(data_dir / 'prices.npy', blp['prices'])
    sparse.save_npz(data_dir / 'Q_sparse.npz', dgp_data['Q_sparse'])
    np.save(data_dir / 'weights.npy', dgp_data['weights'])
    np.save(data_dir / 'capacities.npy', dgp_data['capacities'])
    np.save(data_dir / 'x_modular.npy', dgp_data['x'])
    np.save(data_dir / 'obs_bundles.npy', bundles)

    # --- Write result.json ---
    result = {
        'scenario': 'quadratic_knapsack',
        'size': {
            'N': N, 'M': M,
            'avg_degree': diag['avg_degree'],
            'nnz_Q': diag['nnz_Q'],
        },
        'theta_star': {
            'alpha': float(alpha),
            'lambda': float(lambda_),
            'delta_summary': {
                'mean': float(delta.mean()),
                'std': float(delta.std()),
                'min': float(delta.min()),
                'max': float(delta.max()),
            },
        },
        'dgp_paths': {
            'phi': 'data/phi.npy',
            'z': 'data/z.npy',
            'xi': 'data/xi.npy',
            'delta_star': 'data/delta_star.npy',
            'beta_star': 'data/beta_star.npy',
            'prices': 'data/prices.npy',
            'Q': 'data/Q_sparse.npz',
            'weights': 'data/weights.npy',
            'capacities': 'data/capacities.npy',
            'x_modular': 'data/x_modular.npy',
            'obs_bundles': 'data/obs_bundles.npy',
        },
        'healthy_checks': diag,
        'smoke_tests': {
            'brute_force_match_M15': bf_result,
            'twosls_recovers_beta_star': twosls_ok,
        },
        'runtime_seconds': round(t_total, 2),
    }

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    out_path = BASE / 'results' / 'result.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    logger.info(f"Result written to {out_path}")

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quadratic knapsack DGP pilot')
    parser.add_argument('--size', type=str, default='showcase',
                        choices=['tiny', 'pilot', 'intermediate', 'showcase'])
    args = parser.parse_args()
    run(size_name=args.size)
