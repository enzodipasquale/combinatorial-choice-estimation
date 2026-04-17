"""DGP generation and validation for the FCC-calibrated quadratic knapsack.

Deliverable: healthy DGP with sparse FCC-like bundles, brute-force
verification at tiny size, 2SLS smoke test, result.json + data artifacts.
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
from generate_data import generate_data, n_params, N_MODULAR, REGRESSOR_NAMES
from oracle import compute_utility, twosls_smoke_test

BASE = Path(__file__).resolve().parent
with open(BASE / 'config.yaml') as f:
    CFG = yaml.safe_load(f)


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


def run(cfg=None, size_name='showcase'):
    if cfg is None:
        cfg = CFG

    size_cfg = cfg['sizes'][size_name]
    seeds = cfg['seeds']

    N = size_cfg['N']
    M = size_cfg['M']
    K = n_params(M)

    t0 = time.perf_counter()

    # --- Generate healthy DGP ---
    logger.info(f"Generating FCC-calibrated DGP: size={size_name}, N={N}, M={M}, K={K}")
    theta_star, bundles, dgp_data, diag = generate_data(size_cfg, cfg, seeds)

    theta_mod = theta_star[:N_MODULAR]
    delta = theta_star[N_MODULAR:N_MODULAR + M]
    lambda_ = theta_star[-1]

    logger.info(f"DGP generated:")
    for k, name in enumerate(REGRESSOR_NAMES):
        logger.info(f"  {name} = {theta_mod[k]:.4f}")
    logger.info(f"  lambda = {lambda_:.4f}, delta std = {delta.std():.3f}")
    logger.info(f"Healthy checks: {diag}")

    # --- 2SLS smoke test ---
    blp = dgp_data['blp']
    beta_hat, max_2sls_err = twosls_smoke_test(
        blp['delta'], blp['phi'], blp['z'], blp['beta_star'])
    twosls_tol = 5.0 / np.sqrt(M)
    twosls_ok = max_2sls_err < twosls_tol
    logger.info(f"2SLS smoke test: max err = {max_2sls_err:.2e} "
                f"({'PASS' if twosls_ok else 'FAIL'})")

    t_total = time.perf_counter() - t0

    # --- Save data artifacts ---
    data_dir = BASE / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(data_dir / 'obs_bundles.npy', bundles)
    np.save(data_dir / 'x_modular.npy', dgp_data['x_modular'])
    np.save(data_dir / 'weights.npy', dgp_data['weights'])
    np.save(data_dir / 'pop.npy', dgp_data['pop'])
    np.save(data_dir / 'locations.npy', dgp_data['locations'])
    np.save(data_dir / 'dists.npy', dgp_data['dists'])
    np.save(data_dir / 'capacities.npy', dgp_data['agents']['capacity'])
    np.save(data_dir / 'elig.npy', dgp_data['agents']['elig'])
    np.save(data_dir / 'assets.npy', dgp_data['agents']['assets'])
    np.save(data_dir / 'hq_idx.npy', dgp_data['agents']['hq_idx'])
    sparse.save_npz(data_dir / 'Q_sparse.npz', dgp_data['Q_sparse'])
    np.save(data_dir / 'delta_star.npy', blp['delta'])
    np.save(data_dir / 'beta_star.npy', blp['beta_star'])
    np.save(data_dir / 'phi.npy', blp['phi'])
    np.save(data_dir / 'z.npy', blp['z'])
    np.save(data_dir / 'xi.npy', blp['xi'])
    np.save(data_dir / 'prices.npy', blp['prices'])

    # --- Write result.json ---
    result = {
        'scenario': 'quadratic_knapsack_fcc_calibrated',
        'size': {'N': N, 'M': M, 'K': K,
                 'avg_degree': diag['avg_degree'], 'nnz_Q': diag['nnz_Q']},
        'theta_star': {
            'modular': {name: float(theta_mod[k])
                        for k, name in enumerate(REGRESSOR_NAMES)},
            'lambda': float(lambda_),
            'delta_summary': {
                'mean': float(delta.mean()), 'std': float(delta.std()),
                'min': float(delta.min()), 'max': float(delta.max()),
            },
        },
        'bundle_summary': {
            'mean_size': float(bundles.sum(1).mean()),
            'median_size': float(np.median(bundles.sum(1))),
            'std_size': float(bundles.sum(1).std()),
            'max_size': int(bundles.sum(1).max()),
            'n_empty': int((bundles.sum(1) == 0).sum()),
        },
        'healthy_checks': diag,
        'smoke_tests': {
            'twosls_recovers_beta_star': twosls_ok,
        },
        'runtime_seconds': round(t_total, 2),
    }

    out_path = BASE / 'results' / 'result.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    logger.info(f"Result written to {out_path}")

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='FCC-calibrated quadratic knapsack DGP')
    parser.add_argument('--size', type=str, default='showcase',
                        choices=['pilot', 'showcase'])
    run(size_name=parser.parse_args().size)
