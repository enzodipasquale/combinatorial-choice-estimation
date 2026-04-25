#!/usr/bin/env python3
"""Write network_game MC results to slides/artifacts/.

Outputs:
  * tables/tab_network_game_panelb.tex   MLE vs CB comparison on Panel B
  * tables/network_game_runtimes.tex     \\newcommand macros for the slides

Reads the per-rep JSONs under
    results/<run_name>/{sml,combest}/rep_*.json
(see monte_carlo.py / paper_sml/monte_carlo_sml.py).

Usage:
    python sync_to_slides.py [--run-name panel_b_T500_S10_250reps]
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import chi2

warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(all="ignore")

SCRIPT_DIR = Path(__file__).parent
PROJ_ROOT = SCRIPT_DIR.parent.parent.parent
SLIDES = PROJ_ROOT / "slides" / "artifacts"
TABLES = SLIDES / "tables"

DEFAULT_RUN_NAME = "panel_b_T500_S10_50reps"
DELTA_TRUE = 0.20


def _load(run_dir, subdir):
    files = sorted((run_dir / subdir).glob("rep_*.json"))
    return [json.load(open(f)) for f in files]


def _sml_stats(reps):
    """Mean, std, LR-test size (H0: δ = 0.2), Wald CI coverage."""
    theta = np.array([r["theta_hat"] for r in reps])
    ll_hat = np.array([r["loglik_at_hat"] for r in reps])
    ll_H0  = np.array([r["loglik_at_H0"]  for r in reps])
    hessians = np.array([r["hessian_at_hat"] for r in reps])

    delta = theta[:, 4]
    LR = np.maximum(2.0 * (ll_hat - ll_H0), 0.0)
    size = float((LR > chi2.ppf(0.95, 1)).mean())

    se = np.full(len(reps), np.nan)
    for i, H in enumerate(hessians):
        try:
            v = -np.linalg.inv(H)[4, 4]
            if v > 0:
                se[i] = np.sqrt(v)
        except np.linalg.LinAlgError:
            pass
    valid = ~np.isnan(se)
    lo, hi = delta - 1.96 * se, delta + 1.96 * se
    cov = float(((lo <= DELTA_TRUE) & (DELTA_TRUE <= hi))[valid].mean())

    return {
        "mean": float(delta.mean()),
        "std":  float(delta.std(ddof=1)),
        "LR":   size,
        "CI":   cov,
    }


def _combest_stats(reps):
    """Mean, std for δ̂. LR / CI do not apply (moment-inequality estimator)."""
    delta = np.array([r["theta_hat"][4] for r in reps])
    return {
        "mean": float(delta.mean()),
        "std":  float(delta.std(ddof=1)),
        "LR":   None,
        "CI":   None,
    }


def _misspec_stats(combest_reps):
    """Per-rep combest misspecification diagnostics.

    For each rep computes the argmax-of-potential bundle Y* from the saved
    (U_obs) and compares to Y_obs = min NE (both at theta_true).

      * Hamming(Y_obs, Y*):           integer flip count, agents out of T.
      * bundle sizes |Y_obs|, |Y*|.
      * gap_at_truth = V(Y*; theta_true, U) - V(Y_obs; theta_true, U)
        (already cached per rep).

    argmax V componentwise dominates min NE in the supermodular lattice, so
    every flip is 0 -> 1 (no 1 -> 0 flips); we still report it to confirm.
    """
    # Lazy import so sync_to_slides.py stays fast in the no-results path.
    import sys
    sys.path.insert(0, str(SCRIPT_DIR))
    from generate_data import argmax_potential, build_fixed_design
    design = build_fixed_design(T=combest_reps[0]["T"],
                                avg_degree=10,
                                graph_seed=20260423)
    beta_true = np.array([-1.0, -0.5, -1.0, 0.5])

    hamming = np.empty(len(combest_reps), dtype=int)
    ymin_size = np.empty(len(combest_reps), dtype=int)
    yarg_size = np.empty(len(combest_reps), dtype=int)
    flip_01 = np.empty(len(combest_reps), dtype=int)
    flip_10 = np.empty(len(combest_reps), dtype=int)

    for i, r in enumerate(combest_reps):
        Y = np.asarray(r["Y_obs"], dtype=bool)
        U = np.asarray(r["U_obs"], dtype=float)
        Y_star = argmax_potential(design["X"], design["D"], U,
                                   beta_true, DELTA_TRUE)
        hamming[i]  = int((Y ^ Y_star).sum())
        ymin_size[i] = int(Y.sum())
        yarg_size[i] = int(Y_star.sum())
        flip_01[i]  = int((~Y & Y_star).sum())
        flip_10[i]  = int((Y & ~Y_star).sum())

    gap = np.array([r["gap_at_truth"] for r in combest_reps])
    return {
        "n_reps":         int(len(combest_reps)),
        "n_misspec":      int((hamming > 0).sum()),
        "hamming_mean":   float(hamming.mean()),
        "hamming_median": float(np.median(hamming)),
        "hamming_min":    int(hamming.min()),
        "hamming_max":    int(hamming.max()),
        "ymin_size_mean": float(ymin_size.mean()),
        "yarg_size_mean": float(yarg_size.mean()),
        "flip_01_mean":   float(flip_01.mean()),
        "flip_10_mean":   float(flip_10.mean()),
        "gap_mean":       float(gap.mean()),
        "gap_median":     float(np.median(gap)),
        "gap_max":        float(gap.max()),
    }


def _fmt(x, d=3):
    return "---" if x is None else f"{x:.{d}f}"


def _panelb_table(sml, combest, sml_n, combest_n):
    def row(label, key, d=3):
        return (f"  {label} & {_fmt(sml[key], d)} "
                f"& {_fmt(combest[key], d)} \\\\")

    return "\n".join([
        "% Auto-generated by numerical_experiments/scenarios/network_game/sync_to_slides.py",
        "% Panel B: T=500, N=1, S=10.",
        r"\begin{tabular}{@{}l cc@{}}",
        r"\toprule",
        r"            & MLE             & CB              \\",
        rf"            & ({sml_n} reps) & ({combest_n} reps) \\",
        r"\midrule",
        row(r"Mean of $\hat\delta$",   "mean"),
        row(r"Std.\ dev.\ of $\hat\delta$", "std"),
        row(r"LR test size",            "LR"),
        row(r"95\% CI coverage",        "CI"),
        r"\bottomrule",
        r"\end{tabular}",
    ]) + "\n"


def _runtime_macros(sml_reps, combest_reps, misspec):
    sml_per = float(np.median([r["runtime_total_s"] for r in sml_reps]))
    cb_per  = float(np.median([r["runtime_s"]       for r in combest_reps]))
    T       = int(sml_reps[0]["T"])
    S_sml   = int(sml_reps[0]["S"])
    S_cb    = int(combest_reps[0]["S"])
    avg_deg = float(sml_reps[0]["avg_degree"])
    lines = [
        "% Auto-generated: network_game runtime + dimension macros.",
        rf"\newcommand{{\netgameT}}{{{T}}}",
        rf"\newcommand{{\netgameAvgDeg}}{{{avg_deg:.1f}}}",
        rf"\newcommand{{\netgameSsml}}{{{S_sml}}}",
        rf"\newcommand{{\netgameScombest}}{{{S_cb}}}",
        rf"\newcommand{{\netgameNReps}}{{{len(sml_reps)}}}",
        rf"\newcommand{{\netgameDeltaTrue}}{{{DELTA_TRUE:.2f}}}",
        rf"\newcommand{{\netgameSmlPerRep}}{{{sml_per:.0f}}}",
        rf"\newcommand{{\netgameCombestPerRep}}{{{cb_per:.0f}}}",
        "% Misspecification diagnostics (min NE vs argmax V, at theta_true):",
        rf"\newcommand{{\netgameMisspecNReps}}{{{misspec['n_misspec']}}}",
        rf"\newcommand{{\netgameHammingMean}}{{{misspec['hamming_mean']:.1f}}}",
        rf"\newcommand{{\netgameHammingMedian}}{{{misspec['hamming_median']:.0f}}}",
        rf"\newcommand{{\netgameHammingMin}}{{{misspec['hamming_min']}}}",
        rf"\newcommand{{\netgameHammingMax}}{{{misspec['hamming_max']}}}",
        rf"\newcommand{{\netgameYminSize}}{{{misspec['ymin_size_mean']:.0f}}}",
        rf"\newcommand{{\netgameYargSize}}{{{misspec['yarg_size_mean']:.0f}}}",
        rf"\newcommand{{\netgameFlipZeroToOne}}{{{misspec['flip_01_mean']:.1f}}}",
        rf"\newcommand{{\netgameFlipOneToZero}}{{{misspec['flip_10_mean']:.1f}}}",
        rf"\newcommand{{\netgameGapMean}}{{{misspec['gap_mean']:.2f}}}",
        rf"\newcommand{{\netgameGapMedian}}{{{misspec['gap_median']:.2f}}}",
        rf"\newcommand{{\netgameGapMax}}{{{misspec['gap_max']:.2f}}}",
    ]
    return "\n".join(lines) + "\n"


def main(run_name=DEFAULT_RUN_NAME):
    run_dir = SCRIPT_DIR / "results" / run_name
    TABLES.mkdir(parents=True, exist_ok=True)

    sml_reps     = _load(run_dir, "sml")
    combest_reps = _load(run_dir, "combest")
    if not sml_reps or not combest_reps:
        print(f"[sync_to_slides] no per-rep JSONs under {run_dir}; "
              f"run the MC first.")
        return

    sml_stats = _sml_stats(sml_reps)
    cb_stats  = _combest_stats(combest_reps)
    misspec   = _misspec_stats(combest_reps)

    (TABLES / "tab_network_game_panelb.tex").write_text(
        _panelb_table(sml_stats, cb_stats, len(sml_reps), len(combest_reps)))
    (TABLES / "network_game_runtimes.tex").write_text(
        _runtime_macros(sml_reps, combest_reps, misspec))

    for name in ("tab_network_game_panelb.tex", "network_game_runtimes.tex"):
        print(f"wrote {(TABLES / name).relative_to(PROJ_ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME,
                        help=f"results/<run-name>/{{sml,combest}}/ "
                             f"(default: {DEFAULT_RUN_NAME})")
    args = parser.parse_args()
    main(run_name=args.run_name)
