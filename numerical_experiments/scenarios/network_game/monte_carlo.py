"""Monte Carlo experiment: combest row-generation estimator on the
network peer-effects game.

Per-rep output: one JSON in `results/<run_name>/combest/rep_<i>.json`
with `theta_hat`, observed bundle + shocks, final objective, number of
row-generation iterations, per-observation slack `u_hat`, and two V-gap
diagnostics measuring how far Y_obs falls short of the combinatorial
argmax of the game potential at (theta_true) and (theta_hat). These raw
statistics let any downstream inference quantity (MC std, Manski-style
bounds, ...) be reconstructed without re-running.

Aggregate: `results/<run_name>/combest/summary.json` with per-param mean,
bias, std, rmse across converged reps, plus pooled runtime and the
per-rep misspecification gaps.
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

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from generate_data import (
    argmax_potential,
    build_fixed_design,
    generate_one_rep,
)
from oracle import build_combest_input

import combest as ce
from combest.subproblems.registry.quadratic_obj.quadratic_supermodular.min_cut import (
    QuadraticSupermodularMinCutSolver,
)

PARAM_NAMES = ["beta_1", "beta_2", "beta_3", "beta_4", "delta"]


# ---------------------------------------------------------------------------
# Potential and misspecification diagnostics
# ---------------------------------------------------------------------------

def potential(y, X, D, U, beta, delta):
    y = y.astype(float)
    return float((X @ beta - U) @ y + 0.5 * delta * (y @ D @ y))


def _gap(Y, X, D, U, beta, delta):
    """V(argmax V; theta, U) - V(Y; theta, U). Zero iff Y globally maximizes
    V at the given theta; positive otherwise."""
    Y_star = argmax_potential(X, D, U, beta, delta)
    return (potential(Y_star, X, D, U, beta, delta)
            - potential(Y,      X, D, U, beta, delta))


# ---------------------------------------------------------------------------
# One MC replication
# ---------------------------------------------------------------------------

def _fit_one(design, theta_true, shock_seed, est_cfg):
    beta_true, delta_true = theta_true[:4], float(theta_true[4])
    T = design["X"].shape[0]

    model = ce.Model()
    is_root = model.is_root()

    rep = generate_one_rep(design, beta_true, delta_true,
                           shock_seed=shock_seed, selection="min")
    Y, U = rep["Y"], rep["U"]

    input_data = (build_combest_input(Y, design["X"], design["D"])
                  if is_root else None)

    model.load_config({
        "dimensions": {"n_obs": 1, "n_items": T, "n_covariates": 5,
                       "n_simulations": est_cfg["n_simulations"]},
        "row_generation": {
            "max_iters": est_cfg["max_iters"],
            "tolerance": est_cfg["tolerance"],
            "theta_bounds": est_cfg["theta_bounds"],
        },
    })
    model.data.load_and_distribute_input_data(input_data)
    model.features.build_local_modular_error_oracle(
        seed=42 + shock_seed, sigma=1.0, distribution="normal")
    model.subproblems.load_solver(QuadraticSupermodularMinCutSolver)
    model.subproblems.initialize_solver()
    model.features.build_quadratic_covariates_from_data()

    t0 = time.perf_counter()
    result = model.point_estimation.n_slack.solve(
        initialize_solver=False, verbose=False)
    runtime = time.perf_counter() - t0

    if not is_root:
        return None

    theta_hat = np.asarray(result.theta_hat, dtype=float)
    beta_hat, delta_hat = theta_hat[:4], float(theta_hat[4])

    gap_at_truth = _gap(Y, design["X"], design["D"], U, beta_true, delta_true)
    gap_at_hat   = _gap(Y, design["X"], design["D"], U, beta_hat,  delta_hat)

    u_hat = result.u_hat
    if u_hat is None:
        u_hat_arr = np.full(1, np.nan)
    else:
        u_hat_arr = np.asarray(u_hat, dtype=float).ravel()

    return {
        "theta_hat":        theta_hat,
        "Y_obs":            Y.astype(bool),
        "U_obs":            U.astype(float),
        "bundle_size":      int(Y.sum()),
        "gap_at_truth":     float(gap_at_truth),
        "gap_at_hat":       float(gap_at_hat),
        "final_objective":  float(result.final_objective),
        "n_constraints":    int(getattr(result, "n_constraints", 0) or 0),
        "final_reduced_cost": float(getattr(result, "final_reduced_cost", np.nan)),
        "u_hat":            u_hat_arr,
        "iters":            int(result.num_iterations),
        "runtime_s":        runtime,
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

def run(selection, n_reps, S=None, max_iters=None, tolerance=None,
        T=None, run_name=None, cfg_path=None):
    cfg_path = cfg_path or (BASE / "config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    dgp = cfg["dgp"]
    T = int(T if T is not None else dgp["T"])
    beta_true = np.asarray(dgp["beta"], dtype=float)
    delta_true = float(dgp["delta_true"])
    theta_true = np.concatenate([beta_true, [delta_true]])

    est = dict(cfg["estimation"])
    if S is not None:          est["n_simulations"] = int(S)
    if max_iters is not None:  est["max_iters"]     = int(max_iters)
    if tolerance is not None:  est["tolerance"]     = float(tolerance)

    design = build_fixed_design(T=T, avg_degree=dgp["avg_degree"],
                                graph_seed=int(cfg["seeds"]["graph"]))
    mc_base = int(cfg["seeds"]["mc_base"])

    probe = ce.Model()
    is_root = probe.is_root()
    del probe

    avg_deg = float(design["D"].sum(1).mean()) if is_root else 0.0

    if is_root:
        print(f"[combest_mc] T={T}  avg_deg={avg_deg:.2f}  "
              f"selection={selection!r}  n_reps={n_reps}  "
              f"S={est['n_simulations']}")
        print(f"            theta_true = {theta_true.tolist()}")

    run_name = run_name or (
        f"panel_b_T{T}_S{est['n_simulations']}_{n_reps}reps")
    out_dir = BASE / "results" / run_name / "combest" if is_root else None
    if is_root:
        out_dir.mkdir(parents=True, exist_ok=True)

    per_rep = []
    t_total = time.perf_counter()
    for i in range(n_reps):
        shock_seed = mc_base + i
        out = _fit_one(design, theta_true, shock_seed, est)
        if out is None:
            continue
        out.update({"rep_index": i,
                    "shock_seed": shock_seed,
                    "selection":  selection,
                    "S":          est["n_simulations"],
                    "T":          T,
                    "avg_degree": avg_deg})
        _write_rep(out_dir / f"rep_{i:03d}.json", out)
        per_rep.append(out)
        if is_root:
            print(f"  rep {i+1:>3}/{n_reps}: "
                  f"theta_hat={np.round(out['theta_hat'], 4).tolist()}  "
                  f"|Y|={out['bundle_size']}  "
                  f"gap@truth={out['gap_at_truth']:.3f}  "
                  f"iters={out['iters']}  t={out['runtime_s']:.0f}s",
                  flush=True)
    total_s = time.perf_counter() - t_total

    if not is_root:
        return None

    thetas = np.array([r["theta_hat"] for r in per_rep])
    bias = thetas.mean(axis=0) - theta_true
    std  = thetas.std(axis=0, ddof=1) if len(per_rep) > 1 else np.full(5, np.nan)
    rmse = np.sqrt(((thetas - theta_true) ** 2).mean(axis=0))
    gaps = np.array([r["gap_at_truth"] for r in per_rep])

    print("\n" + "=" * 66)
    print(f"  combest MC summary (selection={selection!r}, "
          f"n_reps={len(per_rep)}, S={est['n_simulations']})")
    print("=" * 66)
    print(f"  {'param':<8} {'true':>9} {'mean':>9} {'bias':>9} {'std':>9} {'rmse':>9}")
    for j, name in enumerate(PARAM_NAMES):
        print(f"  {name:<8} {theta_true[j]:>+9.4f} "
              f"{thetas.mean(axis=0)[j]:>+9.4f} "
              f"{bias[j]:>+9.4f} {std[j]:>9.4f} {rmse[j]:>9.4f}")
    print(f"\n  gap@truth: mean={gaps.mean():.3f}  "
          f"min={gaps.min():.3f}  max={gaps.max():.3f}")
    print(f"  total runtime: {total_s:.1f}s "
          f"({total_s / max(1, len(per_rep)):.1f}s/rep)")

    summary = {
        "scenario": "network_game",
        "estimator": "combest_n_slack",
        "selection": selection,
        "theta_true": theta_true.tolist(),
        "param_names": PARAM_NAMES,
        "T": T, "n_reps": len(per_rep), "avg_degree": avg_deg,
        "config": {"dgp": dgp, "estimation": est, "seeds": cfg["seeds"]},
        "summary": {
            "mean": thetas.mean(axis=0).tolist(),
            "bias": bias.tolist(),
            "std":  std.tolist(),
            "rmse": rmse.tolist(),
            "gap_at_truth_mean": float(gaps.mean()),
            "gap_at_truth_min":  float(gaps.min()),
            "gap_at_truth_max":  float(gaps.max()),
        },
        "runtime_total_s": round(total_s, 2),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  wrote {(out_dir / 'summary.json').relative_to(BASE)}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", choices=["min", "max", "argmax"],
                        default="min",
                        help="equilibrium-selection rule for the DGP "
                             "(default: min, matching Graham and Gonzalez)")
    parser.add_argument("--n-reps", type=int, default=5)
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--S", type=int, default=None,
                        help="override estimation.n_simulations")
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--tolerance", type=float, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()
    run(selection=args.selection, n_reps=args.n_reps, T=args.T,
        S=args.S, max_iters=args.max_iters, tolerance=args.tolerance,
        run_name=args.run_name)
