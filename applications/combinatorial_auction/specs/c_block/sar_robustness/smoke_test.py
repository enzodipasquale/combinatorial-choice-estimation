"""
Smoke test for SAR error injection. No MPI, no row generation.
Verifies that:
  - SAR covariance is built correctly
  - build_local_modular_error_oracle is callable with covariance_matrix
  - local_modular_errors has the right shape and finite values
  - At rho=0 the errors match iid errors (Cholesky of identity is a no-op)

Usage:
    python applications/combinatorial_auction/specs/c_block/sar_robustness/smoke_test.py
"""
import sys, numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import warnings
warnings.filterwarnings("ignore")

from applications.combinatorial_auction.data.loaders import load_bta_data, build_context
from applications.combinatorial_auction.data.prepare import prepare
from applications.combinatorial_auction.specs.c_block.sar_robustness.sar_covariance import build_sar_covariance

import yaml, combest as ce

# ── Build SAR covariance ───────────────────────────────────────────────────
raw = load_bta_data()
ctx = build_context(raw)
adj = ctx["bta_adjacency"]
adj = ((adj + adj.T) > 0).astype(float)
np.fill_diagonal(adj, 0)

print("Testing rho=0.0 (should match iid):")
sar_cov_0 = build_sar_covariance(adj, rho=0.0)
print(f"  Shape: {sar_cov_0.shape}")
print(f"  Max off-diagonal: {sar_cov_0[np.triu_indices(sar_cov_0.shape[0], k=1)].max():.2e}")
assert np.allclose(sar_cov_0, np.eye(sar_cov_0.shape[0]), atol=1e-12), "rho=0 must give identity"
print("  ✓ Identity check passed")

print("\nTesting rho=0.4:")
sar_cov_4 = build_sar_covariance(adj, rho=0.4)
off = sar_cov_4[np.triu_indices(sar_cov_4.shape[0], k=1)]
print(f"  Shape: {sar_cov_4.shape}")
print(f"  Max off-diagonal: {off.max():.4f}")
assert off.max() > 0.1, f"Expected off-diagonal > 0.1 at rho=0.4, got {off.max():.4f}"
print("  ✓ Meaningful correlation check passed")

# ── Build model and inject errors ──────────────────────────────────────────
config_path = Path(__file__).parent / "configs/config_sar_rho04.yaml"
config = yaml.safe_load(open(config_path))

input_data, meta = prepare(
    dataset="c_block",
    modular_regressors=["elig_pop"],
    quadratic_regressors=["adjacency", "pop_centroid_delta4"],
    quadratic_id_regressors=["elig_adjacency"],
    item_modular="fe",
)
meta.pop("raw", None)
config["dimensions"].update(
    n_obs=meta["n_obs"], n_items=meta["n_items"],
    n_covariates=meta["n_covariates"], covariate_names=meta["covariate_names"],
)
config["application"].update(
    n_id_mod=meta["n_id_mod"], n_item_mod=meta["n_item_mod"],
    n_id_quad=meta["n_id_quad"], n_item_quad=meta["n_item_quad"],
)

# Dimension check
assert sar_cov_4.shape[0] == meta["n_items"], \
    f"SAR covariance {sar_cov_4.shape} != n_items={meta['n_items']}"
print(f"\n✓ Dimension check: SAR cov ({sar_cov_4.shape[0]}) == n_items ({meta['n_items']})")

model = ce.Model()
model.load_config(config)
model.data.load_and_distribute_input_data(input_data)
model.features.build_quadratic_covariates_from_data()

# ── Inject rho=0 (identity) and verify matches iid ─────────────────────────
SEED = 2006
model.features.build_local_modular_error_oracle(seed=SEED, covariance_matrix=sar_cov_0)
errors_rho0 = model.features.local_modular_errors.copy()

# Build iid errors directly (no covariance_matrix)
model.features.build_local_modular_error_oracle(seed=SEED, covariance_matrix=None)
errors_iid = model.features.local_modular_errors.copy()

max_diff = np.abs(errors_rho0 - errors_iid).max()
print(f"\nrho=0 vs iid max absolute difference: {max_diff:.2e}")
assert max_diff < 1e-10, f"rho=0 errors differ from iid by {max_diff:.2e} — SAR injection bug!"
print("✓ rho=0 Cholesky path matches iid exactly")

# ── Inject rho=0.4 and verify shape/values ────────────────────────────────
model.features.build_local_modular_error_oracle(seed=SEED, covariance_matrix=sar_cov_4)
errors_rho4 = model.features.local_modular_errors.copy()

print(f"\nrho=0.4 errors:")
print(f"  Shape: {errors_rho4.shape}")
print(f"  All finite: {np.isfinite(errors_rho4).all()}")
print(f"  Mean: {errors_rho4.mean():.4f}, Std: {errors_rho4.std():.4f}")
n_total_rows = errors_rho4.shape[0]
assert errors_rho4.shape[1] == meta["n_items"], f"Wrong n_items: {errors_rho4.shape[1]} != {meta['n_items']}"
assert n_total_rows % meta["n_obs"] == 0, f"Total rows {n_total_rows} not divisible by n_obs {meta['n_obs']}"
assert np.isfinite(errors_rho4).all(), "NaN/Inf in rho=0.4 errors"
assert not np.allclose(errors_rho4, errors_iid), "rho=0.4 errors should differ from iid"
print("✓ rho=0.4 errors have correct shape, finite values, and differ from iid")

print("\n✓ All smoke tests passed. Safe to launch on HPC.")
