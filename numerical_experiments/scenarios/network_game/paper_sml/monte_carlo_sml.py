"""Monte Carlo experiment: simulated MLE (Graham and Gonzalez 2023) on the
network peer-effects game (Panel B: T players, N = 1 game).

Per-rep output: one JSON in `results/<run_name>/sml/rep_<i>.json` with
`theta_hat`, observed bundle + shocks, simulated log-likelihoods at
{theta_hat, theta_true, (beta_hat, delta_0)} with a common scenario draw
for LR-test reproducibility, and the numerical Hessian of the simulated
log-likelihood at theta_hat. These raw statistics let any downstream
inference quantity (Wald SE, LR p-value, Wald CI coverage, MC std,
...) be reconstructed without re-running.

Aggregate: `results/<run_name>/sml/summary.json` with per-param mean,
bias, std, rmse across converged reps, plus pooled runtime.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import yaml

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from generate_data import build_fixed_design, generate_one_rep
from paper_sml.sml import (
    draw_scenarios,
    fit_sml,
    fit_sml_constrained_delta,
    numerical_hessian,
    simulated_loglik,
)

PARAM_NAMES = ["beta_1", "beta_2", "beta_3", "beta_4", "delta"]


def _fit_one(design, theta_true, shock_seed, S, eval_seed,
             hessian_h=1e-4):
    beta_true = theta_true[:4]
    delta_true = float(theta_true[4])

    # Paper DGP: Y = minimal NE
    rep = generate_one_rep(design, beta_true, delta_true,
                           shock_seed=shock_seed, selection="min")
    Y, U = rep["Y"], rep["U"]

    # -------- unconstrained fit (warm-start at truth; paper's anchor too) --
    t0 = time.perf_counter()
    theta_hat, res, fit_scenarios = fit_sml(
        design["X"], design["D"], Y,
        S=S, seed=shock_seed + 10_000, selection="min",
        theta_init=theta_true.copy(),
        beta0=beta_true, delta0=delta_true, verbose=False)
    runtime_fit = time.perf_counter() - t0

    # -------- evaluate all log-likelihoods on one common scenario set --
    # Fresh scenarios drawn once for the LR test + Hessian so every
    # comparison is on the same simulated rectangles.
    t0 = time.perf_counter()
    eval_scenarios = draw_scenarios(
        design["X"], design["D"], Y,
        beta0=beta_true, delta0=delta_true,
        S=S, seed=eval_seed, selection="min")
    ll_at_hat  = simulated_loglik(design["X"], design["D"], Y,
                                  theta_hat[:4], float(theta_hat[4]),
                                  eval_scenarios, selection="min")
    ll_at_true = simulated_loglik(design["X"], design["D"], Y,
                                  beta_true, delta_true,
                                  eval_scenarios, selection="min")

    # -------- constrained fit at H0: delta = delta_true -----------------
    beta_H0, res_H0, _ = fit_sml_constrained_delta(
        design["X"], design["D"], Y,
        delta_fixed=delta_true,
        S=S, seed=shock_seed + 20_000, selection="min",
        beta_init=theta_hat[:4].copy(),
        beta0=beta_true, delta0=delta_true, verbose=False)
    ll_at_H0 = simulated_loglik(design["X"], design["D"], Y,
                                beta_H0, delta_true,
                                eval_scenarios, selection="min")

    # -------- numerical Hessian of ll at theta_hat ---------------------
    def ll_at(th):
        return simulated_loglik(design["X"], design["D"], Y,
                                th[:4], float(max(th[4], 0.0)),
                                eval_scenarios, selection="min")
    hessian = numerical_hessian(ll_at, theta_hat, h=hessian_h)
    runtime_extra = time.perf_counter() - t0

    return {
        "theta_hat":   np.asarray(theta_hat, dtype=float),
        "theta_init":  theta_true.copy(),
        "beta_H0":     np.asarray(beta_H0, dtype=float),
        "Y_obs":       Y.astype(bool),
        "U_obs":       U.astype(float),
        "bundle_size": int(Y.sum()),
        "loglik_at_hat":  float(ll_at_hat),
        "loglik_at_true": float(ll_at_true),
        "loglik_at_H0":   float(ll_at_H0),     # H0: delta = delta_true
        "hessian_at_hat": hessian.astype(float),  # ∇² log-lik at theta_hat
        "n_fev":         int(res.nfev),
        "n_fev_H0":      int(res_H0.nfev),
        "optimizer_success":    bool(res.success),
        "optimizer_success_H0": bool(res_H0.success),
        "runtime_fit_s":   runtime_fit,
        "runtime_extra_s": runtime_extra,
        "runtime_total_s": runtime_fit + runtime_extra,
    }


# ---------------------------------------------------------------------------
# Per-rep JSON helpers
# ---------------------------------------------------------------------------

def _to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def _write_rep(path: Path, rec: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({k: _to_serializable(v) for k, v in rec.items()},
                  f, indent=2)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(n_reps, S, T=None, run_name=None, cfg_path=None):
    cfg_path = cfg_path or (BASE / "config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    dgp = cfg["dgp"]
    T = int(T if T is not None else dgp["T"])
    beta_true = np.asarray(dgp["beta"], dtype=float)
    delta_true = float(dgp["delta_true"])
    theta_true = np.concatenate([beta_true, [delta_true]])

    design = build_fixed_design(T=T, avg_degree=dgp["avg_degree"],
                                graph_seed=int(cfg["seeds"]["graph"]))
    mc_base = int(cfg["seeds"]["mc_base"])

    avg_deg = float(design["D"].sum(1).mean())
    print(f"[paper_sml_mc] T={T}  avg_deg={avg_deg:.2f}  "
          f"n_reps={n_reps}  S={S}  selection='min'")
    print(f"               theta_true = {theta_true.tolist()}")

    run_name = run_name or f"panel_b_T{T}_S{S}_{n_reps}reps"
    out_dir = BASE / "results" / run_name / "sml"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_rep = []
    t_total = time.perf_counter()
    for i in range(n_reps):
        shock_seed = mc_base + i
        eval_seed  = shock_seed + 30_000
        t_rep = time.perf_counter()
        out = _fit_one(design, theta_true, shock_seed, S, eval_seed)
        out.update({"rep_index": i,
                    "shock_seed": shock_seed,
                    "eval_seed":  eval_seed,
                    "S": S, "T": T, "avg_degree": avg_deg})
        _write_rep(out_dir / f"rep_{i:03d}.json", out)
        per_rep.append(out)
        rep_runtime = time.perf_counter() - t_rep
        print(f"  rep {i+1:>3}/{n_reps}: "
              f"theta_hat={np.round(out['theta_hat'], 4).tolist()}  "
              f"|Y|={out['bundle_size']}  "
              f"ll@hat={out['loglik_at_hat']:.3f}  "
              f"LR_stat={2 * (out['loglik_at_hat'] - out['loglik_at_H0']):.3f}  "
              f"t={rep_runtime:.0f}s", flush=True)

    total_s = time.perf_counter() - t_total

    thetas = np.array([r["theta_hat"] for r in per_rep])
    bias = thetas.mean(axis=0) - theta_true
    std  = thetas.std(axis=0, ddof=1) if len(per_rep) > 1 else np.full(5, np.nan)
    rmse = np.sqrt(((thetas - theta_true) ** 2).mean(axis=0))

    print("\n" + "=" * 66)
    print(f"  paper SML MC summary (n_reps={len(per_rep)}, S={S})")
    print("=" * 66)
    print(f"  {'param':<8} {'true':>9} {'mean':>9} {'bias':>9} {'std':>9} {'rmse':>9}")
    for j, name in enumerate(PARAM_NAMES):
        print(f"  {name:<8} {theta_true[j]:>+9.4f} "
              f"{thetas.mean(axis=0)[j]:>+9.4f} "
              f"{bias[j]:>+9.4f} {std[j]:>9.4f} {rmse[j]:>9.4f}")
    print(f"\n  total runtime: {total_s:.1f}s "
          f"({total_s / max(1, len(per_rep)):.1f}s/rep)")

    summary = {
        "scenario": "network_game",
        "estimator": "paper_sml",
        "selection": "min",
        "theta_true": theta_true.tolist(),
        "param_names": PARAM_NAMES,
        "T": T, "S": S, "n_reps": len(per_rep), "avg_degree": avg_deg,
        "config": {"dgp": dgp, "seeds": cfg["seeds"]},
        "summary": {
            "mean": thetas.mean(axis=0).tolist(),
            "bias": bias.tolist(),
            "std":  std.tolist(),
            "rmse": rmse.tolist(),
        },
        "runtime_total_s": round(total_s, 2),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  wrote {(out_dir / 'summary.json').relative_to(BASE)}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-reps", type=int, default=5)
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--S", type=int, default=10,
                        help="scenario draws per rep (paper: 1, 10, 100)")
    parser.add_argument("--run-name", type=str, default=None,
                        help="subdirectory under results/ "
                             "(default: panel_b_T<T>_S<S>_<n>reps)")
    args = parser.parse_args()
    run(n_reps=args.n_reps, S=args.S, T=args.T, run_name=args.run_name)
