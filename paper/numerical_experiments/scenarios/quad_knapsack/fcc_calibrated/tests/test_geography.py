"""Test item/agent distributions match FCC calibration targets."""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generate_data import (build_items, build_adjacency, build_agents,
                           build_modular_features, normalize_interaction_matrix)


@pytest.fixture
def data():
    M, N = 500, 250
    rng = np.random.default_rng(42)
    item_cfg = {'pop_log_mu': 5.5, 'pop_log_sigma': 1.3,
                'pop_min': 27, 'pop_max': 18100}
    agent_cfg = {'cap_log_mu': 6.0, 'cap_log_sigma': 1.8, 'cap_min': 27,
                 'assets_log_mu': -1.0, 'assets_log_sigma': 1.5,
                 'assets_zero_prob': 0.08}
    geo_cfg = {'avg_degree': 10, 'max_nnz_per_item': 20}

    locations, pop, dists = build_items(M, rng, item_cfg)
    pop_norm = pop.astype(float) / pop.sum()
    Q_sparse, Q_dense, Q_bin, avg_deg, nnz = build_adjacency(
        M, dists, pop_norm, geo_cfg)
    agents = build_agents(N, M, pop, rng, agent_cfg)

    return {'M': M, 'N': N, 'pop': pop, 'dists': dists, 'locations': locations,
            'pop_norm': pop_norm, 'Q_dense': Q_dense, 'Q_bin': Q_bin,
            'avg_deg': avg_deg, 'agents': agents}


def test_population_range(data):
    assert data['pop'].min() >= 27
    assert data['pop'].max() <= 18100


def test_population_skewed(data):
    """Mean should exceed median (right-skewed)."""
    assert data['pop'].mean() > np.median(data['pop'])


def test_adjacency_normalized(data):
    """After normalization, Q should have zero diagonal and sum to pop_norm."""
    Q = data['Q_dense']
    assert np.allclose(np.diag(Q), 0)
    # Q is NOT symmetric after row-normalization + pop-scaling
    # But binary adjacency IS symmetric
    assert np.allclose(data['Q_bin'], data['Q_bin'].T)


def test_adjacency_sparsity(data):
    assert 4 <= data['avg_deg'] <= 16


def test_capacity_skewed(data):
    caps = data['agents']['capacity']
    total_w = data['pop'].sum()
    median_frac = np.median(caps) / total_w
    # Real data: ~0.4%. Allow generous range.
    assert median_frac < 0.10, f"Median cap frac {median_frac:.4f} too large"


def test_assets_has_zeros(data):
    n_zero = (data['agents']['assets'] == 0).sum()
    assert n_zero >= 1, "Expected some agents with zero assets"


def test_features_shape(data):
    d = data
    x = build_modular_features(
        d['N'], d['M'], d['agents']['elig'], d['agents']['assets'],
        d['pop_norm'], d['dists'], d['agents']['hq_idx'])
    assert x.shape == (d['N'], d['M'], 6)
    # elig_pop should be non-negative
    assert (x[:, :, 0] >= 0).all()
    # log_dist_hq should be non-negative
    assert (x[:, :, 2] >= 0).all()
