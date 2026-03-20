#!/usr/bin/env python3
"""
Load raw .npz from HPC, compute statistics, generate figure + LaTeX table.
Run locally: python analyze_results.py
"""
import sys
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results" / "raw"
OUTPUT_DIR = SCRIPT_DIR / "results"


def load_cell(J, N):
    path = RESULTS_DIR / f"probit_J{J}_N{N}.npz"
    if not path.exists():
        return None
    return dict(np.load(path, allow_pickle=True))


def compute_cell_stats(data):
    beta = data["beta_star"]
    bm = data["beta_mle"]
    bc = data["beta_cb"]
    N = int(data["N"])
    K = len(beta)
    R = len(bm)

    # Per-component statistics
    bias_mle = np.mean(bm - beta, axis=0)       # (K,)
    bias_cb = np.mean(bc - beta, axis=0)         # (K,)
    sd_mle = np.std(bm, axis=0, ddof=1)          # (K,)
    sd_cb = np.std(bc, axis=0, ddof=1)           # (K,)
    rmse_mle = np.sqrt(np.mean((bm - beta)**2, axis=0))  # (K,)
    rmse_cb = np.sqrt(np.mean((bc - beta)**2, axis=0))   # (K,)

    # Scalar summaries (averaged across components)
    avg_var_mle = np.var(bm, axis=0, ddof=1).mean()
    avg_var_cb = np.var(bc, axis=0, ddof=1).mean()
    are = avg_var_cb / avg_var_mle if avg_var_mle > 0 else np.inf

    return {
        "R": R, "N": N, "K": K,
        "bias_mle": bias_mle, "bias_cb": bias_cb,
        "sd_mle": sd_mle, "sd_cb": sd_cb,
        "rmse_mle": rmse_mle, "rmse_cb": rmse_cb,
        "avg_sd_mle": sd_mle.mean(), "avg_sd_cb": sd_cb.mean(),
        "avg_rmse_mle": rmse_mle.mean(), "avg_rmse_cb": rmse_cb.mean(),
        "n_var_mle": N * avg_var_mle, "n_var_cb": N * avg_var_cb,
        "are": are,
        "time_mle": np.mean(data["time_mle"]),
        "time_cb": np.mean(data["time_cb"]),
    }


def generate_latex_table(all_stats, J_values, N_values, beta, output_path):
    """Standard Monte Carlo table: per-component Bias, SD for each estimator."""
    K = len(beta)
    n_cols = len(N_values)
    col_spec = " ".join("r" for _ in N_values)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Monte Carlo results: Combest vs.\ Probit MLE "
        rf"($K={K}$, $\beta^*=({','.join(f'{b:.1f}' for b in beta)})$, "
        rf"$\sigma={1.0}$, $\rho={0.5}$, $R=200$)}}",
        r"\label{tab:efficiency}",
        r"\small",
        rf"\begin{{tabular}}{{ll{'r' * n_cols}}}",
        r"\toprule",
        " & & " + " & ".join(f"$N={N}$" for N in N_values) + r" \\",
        r"\midrule",
    ]

    for J in J_values:
        label = "Binary choice" if J == 2 else f"$J={J}$"
        lines.append(rf"\multicolumn{{{2 + n_cols}}}{{l}}"
                      rf"{{\textbf{{{label}}}}} \\[2pt]")

        # Bias (norm across components)
        for est, key in [("MLE", "bias_mle"), ("CB", "bias_cb")]:
            vals = []
            for N in N_values:
                s = all_stats.get((J, N))
                if s is None:
                    vals.append("--")
                else:
                    vals.append(f"{np.linalg.norm(s[key]):.4f}")
            lines.append(f"  & $\\|\\text{{Bias}}\\|$ ({est}) & "
                          + " & ".join(vals) + r" \\")

        # SD (average across components)
        for est, key in [("MLE", "avg_sd_mle"), ("CB", "avg_sd_cb")]:
            vals = []
            for N in N_values:
                s = all_stats.get((J, N))
                if s is None:
                    vals.append("--")
                else:
                    vals.append(f"{s[key]:.4f}")
            lines.append(f"  & SD ({est}) & " + " & ".join(vals) + r" \\")

        # RMSE (average across components)
        for est, key in [("MLE", "avg_rmse_mle"), ("CB", "avg_rmse_cb")]:
            vals = []
            for N in N_values:
                s = all_stats.get((J, N))
                if s is None:
                    vals.append("--")
                else:
                    vals.append(f"{s[key]:.4f}")
            lines.append(f"  & RMSE ({est}) & " + " & ".join(vals) + r" \\")

        # ARE
        vals = []
        for N in N_values:
            s = all_stats.get((J, N))
            if s is None:
                vals.append("--")
            else:
                vals.append(f"{s['are']:.2f}")
        lines.append(r"  & ARE & " + " & ".join(vals) + r" \\")

        # Runtime
        for est, key in [("MLE", "time_mle"), ("CB", "time_cb")]:
            vals = []
            for N in N_values:
                s = all_stats.get((J, N))
                if s is None:
                    vals.append("--")
                else:
                    vals.append(f"{s[key]:.2f}")
            lines.append(f"  & Time ({est}) & " + " & ".join(vals) + r" \\")

        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"LaTeX table: {output_path}")


def generate_figure(all_stats, J_values, N_values, output_path):
    """
    Figure with two panels (one per J).
    Each panel: sqrt(N) * SD vs N for MLE and Combest.
    Curves should flatten in the asymptotic regime.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(J_values),
                              figsize=(5 * len(J_values), 4),
                              sharey=False)
    if len(J_values) == 1:
        axes = [axes]

    for ax, J in zip(axes, J_values):
        Ns = []
        sqrt_n_sd_mle, sqrt_n_sd_cb = [], []
        for N in N_values:
            s = all_stats.get((J, N))
            if s is None:
                continue
            Ns.append(N)
            sqrt_n_sd_mle.append(np.sqrt(N) * s["avg_sd_mle"])
            sqrt_n_sd_cb.append(np.sqrt(N) * s["avg_sd_cb"])

        ax.plot(Ns, sqrt_n_sd_mle, "o-", label="Probit MLE",
                color="C0", linewidth=1.5, markersize=6)
        ax.plot(Ns, sqrt_n_sd_cb, "s--", label="Combest",
                color="C1", linewidth=1.5, markersize=6)

        label = "Binary choice ($J=2$)" if J == 2 else f"$J = {J}$"
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("$N$", fontsize=11)
        ax.set_ylabel(r"$\sqrt{N} \cdot \mathrm{SD}$", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Annotate ARE at largest N
        if Ns:
            s = all_stats[(J, Ns[-1])]
            ax.annotate(f"ARE = {s['are']:.2f}",
                        xy=(0.95, 0.05), xycoords="axes fraction",
                        ha="right", va="bottom", fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="wheat", alpha=0.8))

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
    for J in J_values:
        for N in N_values:
            data = load_cell(J, N)
            if data is None:
                print(f"  MISSING: J={J}, N={N}")
                continue
            s = compute_cell_stats(data)
            all_stats[(J, N)] = s
            print(f"  J={J:>2} N={N:>5}: R={s['R']:>3}  "
                  f"Bias_mle={np.linalg.norm(s['bias_mle']):.4f}  "
                  f"Bias_cb={np.linalg.norm(s['bias_cb']):.4f}  "
                  f"SD_mle={s['avg_sd_mle']:.4f}  "
                  f"SD_cb={s['avg_sd_cb']:.4f}  "
                  f"ARE={s['are']:.2f}  "
                  f"t_mle={s['time_mle']:.1f}s  "
                  f"t_cb={s['time_cb']:.3f}s")

    print()
    generate_latex_table(all_stats, J_values, N_values, beta,
                          OUTPUT_DIR / "table.tex")
    generate_figure(all_stats, J_values, N_values,
                     OUTPUT_DIR / "efficiency.pdf")

    # Terminal summary
    print(f"\n{'='*70}")
    print("ARE (Var_CB / Var_MLE) — should stabilize as N grows")
    print(f"{'='*70}")
    header = f"{'J':>4} |" + "".join(f"  N={N:>5}  |" for N in N_values)
    print(header)
    print("-" * len(header))
    for J in J_values:
        row = f"{J:>4} |"
        for N in N_values:
            s = all_stats.get((J, N))
            row += f"  {s['are']:>6.2f}  |" if s else f"  {'--':>6}  |"
        print(row)

    print(f"\n{'='*70}")
    print("Speedup (Time_MLE / Time_CB)")
    print(f"{'='*70}")
    print(header)
    print("-" * len(header))
    for J in J_values:
        row = f"{J:>4} |"
        for N in N_values:
            s = all_stats.get((J, N))
            if s:
                speedup = s["time_mle"] / s["time_cb"] if s["time_cb"] > 0 else np.inf
                row += f"  {speedup:>6.0f}x |"
            else:
                row += f"  {'--':>6}  |"
        print(row)


if __name__ == "__main__":
    main()
