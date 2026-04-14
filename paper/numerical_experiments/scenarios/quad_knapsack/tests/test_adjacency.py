"""Verify adjacency matrix properties."""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generate_data import build_items, build_adjacency


@pytest.mark.parametrize("M", [15, 50, 200])
def test_adjacency_symmetric_binary(M):
    """Q must be symmetric, binary, with zero diagonal."""
    rng = np.random.default_rng(42)
    locations, dists = build_items(M, rng)
    _, Q_dense, _, _ = build_adjacency(M, dists, 8, 15)

    assert np.allclose(Q_dense, Q_dense.T), "Q not symmetric"
    assert set(np.unique(Q_dense)).issubset({0.0, 1.0}), "Q not binary"
    assert np.allclose(np.diag(Q_dense), 0), "Q diagonal not zero"


@pytest.mark.parametrize("M", [50, 200])
def test_adjacency_sparsity(M):
    """Realized average degree should be near target, nnz <= 15*M."""
    rng = np.random.default_rng(42)
    locations, dists = build_items(M, rng)
    _, _, avg_deg, nnz = build_adjacency(M, dists, 8, 15)

    assert 2 <= avg_deg <= 20, f"avg_degree={avg_deg} out of range"
    assert nnz <= 15 * M, f"nnz={nnz} exceeds ceiling {15 * M}"
