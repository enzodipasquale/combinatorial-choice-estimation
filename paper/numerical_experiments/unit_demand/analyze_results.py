#!/usr/bin/env python3
"""
Load raw .npz from all three S-mode subdirectories, compute statistics,
generate a consolidated LaTeX table and efficiency figure.

Run locally:  python analyze_results.py
"""
import sys
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).parent
RAW_BASE = SCRIPT_DIR / "results" / "raw"
OUTPUT_DIR = SCRIPT_DIR / "results"

S_MODES = ["one", "sqrt_n", "match_n"]
S_LABELS = {
    "one": r"$S{=}1$",
    "sqrt_n": r"$S{=}\sqrt{N}$",
    "match_n": r"$S{=}N$",
}


def load_cell(s_mode, J, N):
    path = RAW_BASE / s_mode / f"probit_J{J}_N{N}.npz"
    if not path.exists():
        return None
    return dict(np.load(path, allow_pickle=True))


def compute_cell_stats(data):
    beta = data["beta_star"]
    bm = data["beta_mle"]
    bc = data["beta_cb"]
    N = int(data["N"])
    R = len(bm)

    bias_mle = np.mean(bm - beta, axis=0)
    bias_cb = np.mean(bc - beta, axis=0)
    sd_mle = np.std(bm, axis=0, ddof=1)
    sd_cb = np.std(bc, axis=0, ddof=1)

    avg_var_mle = np.var(bm, axis=0, ddof=1).mean()
    avg_var_cb = np.var(bc, axis=0, ddof=1).mean()
    are = avg_var_cb / avg_var_mle if avg_var_mle > 0 else np.inf

    return {
        "R": R, "N": N,
        "bias_mle": bias_mle, "bias_cb": bias_cb,
        "avg_sd_mle": sd_mle.mean(), "avg_sd_cb": sd_cb.mean(),
        "are": are,
        "time_mle": np.mean(data["time_mle"]),
        "time_cb": np.mean(data["time_cb"]),
        "n_simulations": int(data.get("n_simulations", 0)),
    }


def generate_consolidated_table(all_stats, J_values, N_values, beta,
                                 output_path):
    K = len(beta)
    n_cols = len(N_values)
    beta_str = ",".join(f"{b:.1f}" for b in beta)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Monte Carlo comparison: Probit MLE vs.\ Combest with "
        r"$S \in \{1, \sqrt{N}, N\}$ simulations",
        rf"($K={K}$, $\beta^*=({beta_str})$, "
        r"$\sigma=1.0$, $\rho=0.5$, $R=200$).}",
        r"\label{tab:unit_demand_efficiency}",
        r"\small",
        rf"\begin{{tabular}}{{ll{'r' * n_cols}}}",
        r"\toprule",
        " & & " + " & ".join(f"$N={N}$" for N in N_values) + r" \\",
        r"\midrule",
    ]

    for J in J_values:
        label = "Binary choice ($J=2$)" if J == 2 else f"$J={J}$"
        lines.append(rf"\multicolumn{{{2 + n_cols}}}{{l}}"
                      rf"{{\textbf{{{label}}}}} \\[2pt]")

        # SD rows: MLE (same across S), then CB per S
        s0 = next((all_stats.get((sm, J, N)) for sm in S_MODES
                    for N in N_values if (sm, J, N) in all_stats), None)
        if s0:
            vals = []
            for N in N_values:
                s = next((all_stats[(sm, J, N)] for sm in S_MODES
                          if (sm, J, N) in all_stats), None)
                vals.append(f"{s['avg_sd_mle']:.4f}" if s else "--")
            lines.append("  & SD (MLE) & " + " & ".join(vals) + r" \\")

        for sm in S_MODES:
            sl = S_LABELS[sm]
            vals = []
            for N in N_values:
                s = all_stats.get((sm, J, N))
                vals.append(f"{s['avg_sd_cb']:.4f}" if s else "--")
            lines.append(f"  & SD (CB, {sl}) & " + " & ".join(vals)
                          + r" \\")

        lines.append(r"  \\[-4pt]")

        # ARE rows
        for sm in S_MODES:
            sl = S_LABELS[sm]
            vals = []
            for N in N_values:
                s = all_stats.get((sm, J, N))
                vals.append(f"{s['are']:.2f}" if s else "--")
            lines.append(f"  & ARE ({sl}) & " + " & ".join(vals) + r" \\")

        lines.append(r"  \\[-4pt]")

        # SE ratio rows: sqrt(ARE)
        for sm in S_MODES:
            sl = S_LABELS[sm]
            vals = []
            for N in N_values:
                s = all_stats.get((sm, J, N))
                vals.append(f"{np.sqrt(s['are']):.2f}" if s else "--")
            lines.append(r"  & s.e.\ ratio (" + sl + ") & "
                          + " & ".join(vals) + r" \\")

        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"LaTeX table: {output_path}")


def generate_figure(all_stats, J_values, N_values, output_path):
    """Two panels (one per J). sqrt(N)*SD vs N for MLE and each S mode."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {
        "one":     ("v:", "C2", r"CB ($S{=}1$)"),
        "sqrt_n":  ("s--", "C1", r"CB ($S{=}\sqrt{N}$)"),
        "match_n": ("D-.", "C3", r"CB ($S{=}N$)"),
    }

    fig, axes = plt.subplots(1, len(J_values),
                              figsize=(5 * len(J_values), 4),
                              sharey=False)
    if len(J_values) == 1:
        axes = [axes]

    for ax, J in zip(axes, J_values):
        # MLE curve (from any S mode — identical)
        Ns, mle_curve = [], []
        for N in N_values:
            s = next((all_stats[(sm, J, N)] for sm in S_MODES
                      if (sm, J, N) in all_stats), None)
            if s is None:
                continue
            Ns.append(N)
            mle_curve.append(np.sqrt(N) * s["avg_sd_mle"])

        ax.plot(Ns, mle_curve, "o-", label="Probit MLE",
                color="C0", linewidth=1.5, markersize=6)

        # CB curves per S mode
        for sm in S_MODES:
            marker, color, lbl = styles[sm]
            cb_Ns, cb_curve = [], []
            for N in N_values:
                s = all_stats.get((sm, J, N))
                if s is None:
                    continue
                cb_Ns.append(N)
                cb_curve.append(np.sqrt(N) * s["avg_sd_cb"])
            if cb_curve:
                ax.plot(cb_Ns, cb_curve, marker, label=lbl,
                        color=color, linewidth=1.5, markersize=6)

        label = "Binary choice ($J=2$)" if J == 2 else f"$J = {J}$"
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("$N$", fontsize=11)
        ax.set_ylabel(r"$\sqrt{N} \cdot \mathrm{SD}$", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Figure: {output_path}")
    plt.close(fig)


def main():
    import yaml
    with open(SCRIPT_DIR / "config.yaml") as f:
        config = yaml.safe_load(f)

    J_values = config["grid"]["J"]
    N_values = config["grid"]["N"]
    beta = np.array(config["experiment"]["beta_star"])

    print("Loading results...\n")
    all_stats = {}
    for sm in S_MODES:
        for J in J_values:
            for N in N_values:
                data = load_cell(sm, J, N)
                if data is None:
                    continue
                s = compute_cell_stats(data)
                all_stats[(sm, J, N)] = s
                print(f"  {sm:>7s}  J={J:>2} N={N:>5}: R={s['R']:>3}  "
                      f"S={s['n_simulations']:>4}  "
                      f"SD_mle={s['avg_sd_mle']:.4f}  "
                      f"SD_cb={s['avg_sd_cb']:.4f}  "
                      f"ARE={s['are']:.2f}  "
                      f"t_cb={s['time_cb']:.3f}s")

    if not all_stats:
        print("No results found.")
        return

    print()
    generate_consolidated_table(all_stats, J_values, N_values, beta,
                                 OUTPUT_DIR / "table.tex")
    generate_figure(all_stats, J_values, N_values,
                     OUTPUT_DIR / "efficiency.pdf")

    # Terminal summaries
    print(f"\n{'='*70}")
    print("ARE (Var_CB / Var_MLE)")
    print(f"{'='*70}")
    header = f"{'S mode':>8} {'J':>3} |" + "".join(
        f"  N={N:>5}  |" for N in N_values)
    print(header)
    print("-" * len(header))
    for sm in S_MODES:
        for J in J_values:
            row = f"{sm:>8} {J:>3} |"
            for N in N_values:
                s = all_stats.get((sm, J, N))
                row += f"  {s['are']:>6.2f}  |" if s else f"  {'--':>6}  |"
            print(row)

    print(f"\n{'='*70}")
    print("s.e. ratio = sqrt(ARE)")
    print(f"{'='*70}")
    print(header)
    print("-" * len(header))
    for sm in S_MODES:
        for J in J_values:
            row = f"{sm:>8} {J:>3} |"
            for N in N_values:
                s = all_stats.get((sm, J, N))
                row += f"  {np.sqrt(s['are']):>6.2f}  |" if s else f"  {'--':>6}  |"
            print(row)

    print(f"\n{'='*70}")
    print("Speedup (Time_MLE / Time_CB)")
    print(f"{'='*70}")
    print(header)
    print("-" * len(header))
    for sm in S_MODES:
        for J in J_values:
            row = f"{sm:>8} {J:>3} |"
            for N in N_values:
                s = all_stats.get((sm, J, N))
                if s and s["time_cb"] > 0:
                    row += f"  {s['time_mle']/s['time_cb']:>6.1f}x |"
                else:
                    row += f"  {'--':>6}  |"
            print(row)


if __name__ == "__main__":
    main()
