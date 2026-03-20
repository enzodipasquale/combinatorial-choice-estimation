#!/usr/bin/env python3
"""
Load raw .npz results from HPC, compute statistics, generate figure + table.
Run locally after HPC job completes.
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
    R = len(bm)

    bias_mle = np.mean(bm - beta, axis=0)
    bias_cb = np.mean(bc - beta, axis=0)
    mse_mle = np.mean((bm - beta)**2, axis=0)
    mse_cb = np.mean((bc - beta)**2, axis=0)
    var_mle = np.var(bm, axis=0)
    var_cb = np.var(bc, axis=0)
    N = int(data["N"])

    return {
        "R": R,
        "bias_mle": bias_mle, "bias_cb": bias_cb,
        "rmse_mle": np.sqrt(mse_mle.mean()),
        "rmse_cb": np.sqrt(mse_cb.mean()),
        "mse_mle": mse_mle.mean(), "mse_cb": mse_cb.mean(),
        "n_var_mle": N * var_mle.mean(), "n_var_cb": N * var_cb.mean(),
        "are": mse_cb.mean() / mse_mle.mean(),
        "time_mle": np.mean(data["time_mle"]),
        "time_cb": np.mean(data["time_cb"]),
    }


def generate_latex_table(all_stats, J_values, N_values, output_path):
    """Generate LaTeX table for the paper."""
    cols = " ".join(f"S[table-format=1.3]" for _ in N_values)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Asymptotic efficiency: Combest vs Probit MLE}",
        r"\label{tab:efficiency}",
        rf"\begin{{tabular}}{{ll {cols}}}",
        r"\toprule",
        " & & " + " & ".join(f"{{$N={N}$}}" for N in N_values) + r" \\",
        r"\midrule",
    ]

    for J in J_values:
        label = "Binary" if J == 2 else f"$J={J}$"
        lines.append(rf"\multicolumn{{{2 + len(N_values)}}}{{l}}{{\textbf{{{label}}}}} \\")

        for row_label, key in [
            ("Bias (MLE)", "bias_mle_norm"),
            ("Bias (CB)", "bias_cb_norm"),
            ("RMSE (MLE)", "rmse_mle"),
            ("RMSE (CB)", "rmse_cb"),
            ("ARE", "are"),
            ("Time MLE (s)", "time_mle"),
            ("Time CB (s)", "time_cb"),
        ]:
            vals = []
            for N in N_values:
                s = all_stats.get((J, N))
                if s is None:
                    vals.append("--")
                elif key == "bias_mle_norm":
                    vals.append(f"{np.linalg.norm(s['bias_mle']):.4f}")
                elif key == "bias_cb_norm":
                    vals.append(f"{np.linalg.norm(s['bias_cb']):.4f}")
                elif key == "are":
                    vals.append(f"{s[key]:.2f}")
                elif key.startswith("time"):
                    vals.append(f"{s[key]:.2f}")
                else:
                    vals.append(f"{s[key]:.4f}")
            lines.append(f"  & {row_label} & " + " & ".join(vals) + r" \\")
        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"LaTeX table saved to {output_path}")


def generate_figure(all_stats, J_values, N_values, output_path):
    """Generate N*MSE vs N figure, one panel per J."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(J_values), figsize=(5 * len(J_values), 4),
                              sharey=False)
    if len(J_values) == 1:
        axes = [axes]

    for ax, J in zip(axes, J_values):
        Ns, nvar_mle, nvar_cb = [], [], []
        for N in N_values:
            s = all_stats.get((J, N))
            if s is None:
                continue
            Ns.append(N)
            nvar_mle.append(s["n_var_mle"])
            nvar_cb.append(s["n_var_cb"])

        ax.plot(Ns, nvar_mle, "o-", label="Probit MLE", color="C0")
        ax.plot(Ns, nvar_cb, "s--", label="Combest", color="C1")
        label = "Binary choice" if J == 2 else f"$J = {J}$"
        ax.set_title(label)
        ax.set_xlabel("$N$")
        ax.set_ylabel("$N \\cdot \\mathrm{MSE}$")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {output_path}")
    plt.close(fig)


def main():
    import yaml
    with open(SCRIPT_DIR / "config.yaml") as f:
        config = yaml.safe_load(f)

    J_values = config["grid"]["J"]
    N_values = config["grid"]["N"]

    all_stats = {}
    for J in J_values:
        for N in N_values:
            data = load_cell(J, N)
            if data is None:
                print(f"  Missing: J={J}, N={N}")
                continue
            s = compute_cell_stats(data)
            all_stats[(J, N)] = s
            print(f"  J={J} N={N}: R={s['R']}  "
                  f"RMSE_mle={s['rmse_mle']:.4f}  RMSE_cb={s['rmse_cb']:.4f}  "
                  f"ARE={s['are']:.2f}  "
                  f"N*Var_mle={s['n_var_mle']:.2f}  N*Var_cb={s['n_var_cb']:.2f}")

    print()
    generate_latex_table(all_stats, J_values, N_values, OUTPUT_DIR / "table.tex")
    generate_figure(all_stats, J_values, N_values, OUTPUT_DIR / "efficiency.pdf")

    # Print summary to terminal
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    header = f"{'J':>4} |" + "".join(f"  N={N:>5}  |" for N in N_values)
    print(header)
    print("-" * len(header))
    for J in J_values:
        row = f"{J:>4} |"
        for N in N_values:
            s = all_stats.get((J, N))
            row += f"  {s['are']:>6.2f}  |" if s else f"  {'--':>6}  |"
        print(row)


if __name__ == "__main__":
    main()
