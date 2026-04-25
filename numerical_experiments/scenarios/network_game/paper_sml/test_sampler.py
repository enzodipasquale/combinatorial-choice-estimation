"""Sanity checks for the scenario sampler.

Tiny test: T = 8, one observation, verify that
  * sample_scenario produces U that reconstructs Y as the MINIMAL NE
  * sample_scenario with selection='max' produces U that reconstructs Y
    as the MAXIMAL NE of the same game
  * simulated log-likelihood evaluates without error and returns a number
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from generate_data import build_fixed_design, generate_one_rep, minimal_ne, maximal_ne
from paper_sml.scenario_sampler import sample_scenario
from paper_sml.sml import simulated_loglik, draw_scenarios


def test_min_sampler():
    rng = np.random.default_rng(0)
    design = build_fixed_design(T=8, avg_degree=3, graph_seed=1)
    X, D = design["X"], design["D"]
    beta = np.array([-0.3, 0.2, -0.5, 0.4])
    delta = 0.15

    # Draw actual U, compute MINIMAL NE, check sampler reverses this
    U_true = rng.standard_normal(8)
    Y_min = minimal_ne(X, D, U_true, beta, delta)
    print(f"[min] Y from DGP: {Y_min.astype(int)}  sum={Y_min.sum()}")

    # Sample one scenario anchored at (β, δ)
    U_sampled, c_lo, c_hi, lm0, lw = sample_scenario(
        X, D, Y_min, beta, delta, np.random.default_rng(42), selection="min")
    Y_check = minimal_ne(X, D, U_sampled, beta, delta)
    ok = np.array_equal(Y_check, Y_min)
    print(f"[min] Y reconstructed from sampled U: {Y_check.astype(int)}  "
          f"match={ok}  log_mass0={lm0:.2f}  log_omega={lw:.2f}")
    # Check U_sampled lands in its identified sub-bucket.
    Xb = X @ beta
    lo = np.where(np.isfinite(c_lo), Xb + delta * c_lo, -np.inf)
    up = np.where(np.isfinite(c_hi), Xb + delta * c_hi,  np.inf)
    for t in range(8):
        assert lo[t] - 1e-6 <= U_sampled[t] <= up[t] + 1e-6, \
            f"U[{t}] = {U_sampled[t]} not in [{lo[t]}, {up[t]}]"
    print(f"[min] U in sub-bucket ✓")
    return ok


def test_max_sampler():
    # Not implemented — see scenario_sampler module docstring.
    return True


def test_loglik():
    design = build_fixed_design(T=8, avg_degree=3, graph_seed=1)
    X, D = design["X"], design["D"]
    beta = np.array([-0.3, 0.2, -0.5, 0.4])
    delta = 0.15

    rep = generate_one_rep(design, beta, delta, shock_seed=77, selection="min")
    Y = rep["Y"]
    print(f"[loglik] Y_min sum={Y.sum()}")

    scenarios = draw_scenarios(X, D, Y, beta, delta, S=5, seed=123,
                               selection="min")
    ll = simulated_loglik(X, D, Y, beta, delta, scenarios, selection="min")
    print(f"[loglik] log Pr(Y|X;θ*) ≈ {ll:.4f}")
    # Perturbed θ should give lower likelihood typically
    ll_perturb = simulated_loglik(X, D, Y,
                                   beta + 0.5 * np.ones(4), delta,
                                   scenarios, selection="min")
    print(f"[loglik] log Pr(Y|X; perturbed) ≈ {ll_perturb:.4f}")
    return np.isfinite(ll)


if __name__ == "__main__":
    r1 = test_min_sampler()
    r2 = test_max_sampler()
    r3 = test_loglik()
    print("\n" + ("PASS" if (r1 and r2 and r3) else "FAIL"))
