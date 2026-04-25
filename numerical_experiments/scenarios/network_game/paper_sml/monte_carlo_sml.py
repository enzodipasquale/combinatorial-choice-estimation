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
from oracle import naive_probit
from paper_sml.sml import (
    draw_scenarios,
    draw_uniforms,
    fit_sml,
    fit_sml_constrained_delta,
    fit_sml_crn,
    fit_sml_crn_constrained_delta,
    numerical_hessian,
    simulated_loglik,
    simulated_loglik_crn,
)

PARAM_NAMES = ["beta_1", "beta_2", "beta_3", "beta_4", "delta"]


def _fit_one(design, theta_true, shock_seed, S, eval_seed, selection,
             init="truth", anchor="same", hessian_h=1e-4,
             protocol="one-step", s_round1=1):
    beta_true = theta_true[:4]
    delta_true = float(theta_true[4])

    # Y generated with the requested DGP selection rule. The SML likelihood
    # is always evaluated in paper's "min NE" form regardless; feeding
    # argmax-V Y is an explicit misspecification test (the mirror of the
    # paper's Panel B).
    rep = generate_one_rep(design, beta_true, delta_true,
                           shock_seed=shock_seed, selection=selection)
    Y, U = rep["Y"], rep["U"]

    # ---- CRN branch (paper Appendix D first paragraph) -------------
    # Fix S × T uniforms, regenerate scenarios at each θ via inverse-CDF.
    # No anchor, no recycling, no two-step.
    if protocol == "crn":
        T = design["X"].shape[0]
        uniforms_S = draw_uniforms(S, T, seed=shock_seed + 10_000)

        # Optimizer init. Paper doesn't specify; we honor --init.
        if init == "truth":
            theta_init_vec = theta_true.copy()
        elif init == "truth_perturb":
            rng_init = np.random.default_rng(shock_seed + 99_999)
            theta_init_vec = theta_true + 0.1 * rng_init.standard_normal(5)
            theta_init_vec[4] = max(float(theta_init_vec[4]), 0.01)
        elif init == "naive":
            theta_init_vec = np.asarray(
                naive_probit(Y, design["X"], design["D"]), dtype=float)
        elif init == "zeros":
            theta_init_vec = np.zeros(5)
        else:
            raise ValueError(f"init={init!r} not supported with protocol='crn'")

        t0 = time.perf_counter()
        theta_hat, res = fit_sml_crn(
            design["X"], design["D"], Y, uniforms_S,
            theta_init=theta_init_vec, verbose=False)
        runtime_fit = time.perf_counter() - t0

        t0 = time.perf_counter()
        ll_at_hat  = simulated_loglik_crn(
            design["X"], design["D"], Y,
            theta_hat[:4], float(theta_hat[4]), uniforms_S)
        ll_at_true = simulated_loglik_crn(
            design["X"], design["D"], Y,
            beta_true, delta_true, uniforms_S)

        beta_H0, res_H0 = fit_sml_crn_constrained_delta(
            design["X"], design["D"], Y, uniforms_S,
            delta_fixed=delta_true,
            beta_init=theta_hat[:4].copy(), verbose=False)
        ll_at_H0 = simulated_loglik_crn(
            design["X"], design["D"], Y,
            beta_H0, delta_true, uniforms_S)

        def ll_at(th):
            return simulated_loglik_crn(
                design["X"], design["D"], Y,
                th[:4], float(max(th[4], 0.0)), uniforms_S)
        hessian = numerical_hessian(ll_at, theta_hat, h=hessian_h)
        runtime_extra = time.perf_counter() - t0

        return {
            "theta_hat":         np.asarray(theta_hat, dtype=float),
            "theta_hat_round1":  None,
            "theta_init":        theta_init_vec.copy(),
            "beta_H0":           np.asarray(beta_H0, dtype=float),
            "Y_obs":       Y.astype(bool),
            "U_obs":       U.astype(float),
            "bundle_size": int(Y.sum()),
            "loglik_at_hat":  float(ll_at_hat),
            "loglik_at_true": float(ll_at_true),
            "loglik_at_H0":   float(ll_at_H0),
            "hessian_at_hat": hessian.astype(float),
            "n_fev":         int(res.nfev),
            "n_fev_H0":      int(res_H0.nfev),
            "optimizer_success":    bool(res.success),
            "optimizer_success_H0": bool(res_H0.success),
            "runtime_fit_s":   runtime_fit,
            "runtime_extra_s": runtime_extra,
            "runtime_total_s": runtime_fit + runtime_extra,
        }

    # ---- protocol-aware setup of optimizer init and IS anchor ----
    # "two-step" (paper Appendix D): round 1 uses naive-probit init + anchor
    # with s_round1 draws to produce a preliminary θ̃; round 2 sets
    # init = anchor = θ̃ and uses S draws. "one-step" uses --init/--anchor
    # directly (for diagnostics).
    theta_hat_round1 = None
    if protocol == "two-step":
        theta_probit = np.asarray(
            naive_probit(Y, design["X"], design["D"]), dtype=float)
        theta_r1, _res_r1, _ = fit_sml(
            design["X"], design["D"], Y,
            S=s_round1, seed=shock_seed + 5_000, selection="min",
            theta_init=theta_probit,
            beta0=theta_probit[:4], delta0=float(theta_probit[4]),
            verbose=False)
        theta_init_vec   = theta_r1.copy()
        anchor_vec       = theta_r1.copy()
        theta_hat_round1 = theta_r1.copy()
    else:
        if init == "truth":
            theta_init_vec = theta_true.copy()
        elif init == "naive":
            theta_init_vec = np.asarray(
                naive_probit(Y, design["X"], design["D"]), dtype=float)
        elif init == "zeros":
            theta_init_vec = np.zeros(5)
        else:
            raise ValueError(f"init={init!r}; expected 'truth', 'naive', or 'zeros'")

        if anchor == "same":
            anchor_vec = theta_init_vec
        elif anchor == "truth":
            anchor_vec = theta_true.copy()
        elif anchor == "naive":
            anchor_vec = np.asarray(
                naive_probit(Y, design["X"], design["D"]), dtype=float)
        else:
            raise ValueError(f"anchor={anchor!r}; expected 'same', 'truth', or 'naive'")
    beta0_vec  = anchor_vec[:4].copy()
    delta0_val = float(anchor_vec[4])

    # -------- unconstrained fit --------
    t0 = time.perf_counter()
    theta_hat, res, fit_scenarios = fit_sml(
        design["X"], design["D"], Y,
        S=S, seed=shock_seed + 10_000, selection="min",
        theta_init=theta_init_vec,
        beta0=beta0_vec, delta0=delta0_val, verbose=False)
    runtime_fit = time.perf_counter() - t0

    # -------- evaluate all log-likelihoods on one common scenario set --
    # Fresh scenarios drawn once for the LR test + Hessian so every
    # comparison is on the same simulated rectangles. Anchored at the
    # same point the optimizer uses, so importance weights are coherent.
    t0 = time.perf_counter()
    eval_scenarios = draw_scenarios(
        design["X"], design["D"], Y,
        beta0=beta0_vec, delta0=delta0_val,
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
        beta0=beta0_vec, delta0=delta0_val, verbose=False)
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
        "theta_hat":         np.asarray(theta_hat, dtype=float),
        "theta_hat_round1":  (theta_hat_round1.astype(float)
                               if theta_hat_round1 is not None else None),
        "theta_init":        theta_init_vec.copy(),
        "beta_H0":           np.asarray(beta_H0, dtype=float),
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

def run(n_reps, S, T=None, run_name=None, cfg_path=None, selection="min",
        init="truth", anchor="same", protocol="one-step", s_round1=1):
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
          f"n_reps={n_reps}  S={S}  selection={selection!r}  "
          f"protocol={protocol!r}  s_round1={s_round1}  "
          f"init={init!r}  anchor={anchor!r}")
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
        out = _fit_one(design, theta_true, shock_seed, S, eval_seed,
                       selection, init=init, anchor=anchor,
                       protocol=protocol, s_round1=s_round1)
        out.update({"rep_index": i,
                    "shock_seed": shock_seed,
                    "eval_seed":  eval_seed,
                    "selection":  selection,
                    "init":       init,
                    "anchor":     anchor,
                    "protocol":   protocol,
                    "s_round1":   s_round1,
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
        "selection": selection,
        "protocol": protocol,
        "s_round1": s_round1,
        "init": init,
        "anchor": anchor,
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
    parser.add_argument("--selection", choices=["min", "max", "argmax"],
                        default="min",
                        help="DGP equilibrium-selection rule (default: min, "
                             "matching Graham and Gonzalez; 'argmax' is the "
                             "mirror misspecification experiment)")
    parser.add_argument("--init",
                        choices=["truth", "truth_perturb", "naive", "zeros"],
                        default="truth",
                        help="optimizer init. truth uses theta_true; "
                             "truth_perturb uses theta_true + N(0, 0.1) "
                             "(seeded per rep); naive uses single-agent "
                             "probit on Y; zeros is a blind cold start.")
    parser.add_argument("--anchor", choices=["same", "truth", "naive"],
                        default="same",
                        help="importance-sampler anchor θ^(0). 'same' "
                             "(default) uses the init value; 'truth' forces "
                             "theta_true; 'naive' forces the naive-probit "
                             "estimate. Decoupling init from anchor lets "
                             "us isolate the two 'cheats'.")
    parser.add_argument("--protocol",
                        choices=["one-step", "two-step", "crn"],
                        default="one-step",
                        help="estimation protocol. crn (paper Appendix D, "
                             "first paragraph): fix S × T uniforms, "
                             "regenerate scenarios at each θ via inverse-CDF, "
                             "no anchor / no recycling. two-step: round 1 "
                             "naive-probit, round 2 re-anchors at θ̃. "
                             "one-step: --init/--anchor directly (diagnostic).")
    parser.add_argument("--s-round1", type=int, default=1,
                        help="S for round 1 of two-step protocol "
                             "(paper: 'just a few'; default 1).")
    parser.add_argument("--run-name", type=str, default=None,
                        help="subdirectory under results/ "
                             "(default: panel_b_T<T>_S<S>_<n>reps)")
    args = parser.parse_args()
    run(n_reps=args.n_reps, S=args.S, T=args.T,
        selection=args.selection, init=args.init, anchor=args.anchor,
        protocol=args.protocol, s_round1=args.s_round1,
        run_name=args.run_name)
