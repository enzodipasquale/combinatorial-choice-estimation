#!/usr/bin/env python3
"""Load raw .npz for {probit, logit}, generate consolidated LaTeX table + figure.

Usage:
  python analyze_results.py --model probit
  python analyze_results.py --model logit
  python analyze_results.py           # both
"""
import argparse
from pathlib import Path
import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).parent

S_MODES = ["one", "sqrt_n", "match_n"]
S_LABELS = {
    "one": r"$S{=}1$",
    "sqrt_n": r"$S{=}\sqrt{N}$",
    "match_n": r"$S{=}N$",
}
MLE_LABEL = {"probit": "Probit MLE", "logit": "Logit MLE (closed form)"}


def results_root(model):
    return SCRIPT_DIR / "results" / model


def load_cell(model, s_mode, J, N):
    path = results_root(model) / "raw" / s_mode / f"{model}_J{J}_N{N}.npz"
    if not path.exists():
        return None
    return dict(np.load(path, allow_pickle=True))


def compute_stats(data):
    beta = data["beta_star"]
    bm = data["beta_mle"]
    bc = data["beta_cb"]
    N = int(data["N"])
    R = len(bm)
    sd_mle = np.std(bm, axis=0, ddof=1).mean()
    sd_cb = np.std(bc, axis=0, ddof=1).mean()
    var_mle = np.var(bm, axis=0, ddof=1).mean()
    var_cb = np.var(bc, axis=0, ddof=1).mean()
    are = var_cb / var_mle if var_mle > 0 else np.inf
    return {
        "R": R, "N": N,
        "avg_sd_mle": sd_mle, "avg_sd_cb": sd_cb,
        "are": are,
        "time_mle": np.mean(data["time_mle"]),
        "time_cb": np.mean(data["time_cb"]),
        "n_simulations": int(data.get("n_simulations", 0)),
    }


def generate_table(all_stats, J_values, N_values, beta, model, output_path):
    K = len(beta)
    n_cols = len(N_values)
    beta_str = ",".join(f"{b:.1f}" for b in beta)
    mle_name = {"probit": "Probit MLE", "logit": "Logit MLE"}[model]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Monte Carlo comparison: {mle_name} vs.\ Combest with "
        r"$S \in \{1, \sqrt{N}, N\}$ simulations",
        rf"($K={K}$, $\beta^*=({beta_str})$, "
        r"$\sigma=1.0$, $\rho=0.5$, $R=200$).}",
        rf"\label{{tab:unit_demand_{model}_efficiency}}",
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

        # SD (MLE)
        vals = []
        for N in N_values:
            s = next((all_stats[(sm, J, N)] for sm in S_MODES
                      if (sm, J, N) in all_stats), None)
            vals.append(f"{s['avg_sd_mle']:.4f}" if s else "--")
        lines.append("  & SD (MLE) & " + " & ".join(vals) + r" \\")

        # SD (CB) per S mode
        for sm in S_MODES:
            vals = [f"{all_stats[(sm, J, N)]['avg_sd_cb']:.4f}"
                    if (sm, J, N) in all_stats else "--"
                    for N in N_values]
            lines.append(f"  & SD (CB, {S_LABELS[sm]}) & "
                          + " & ".join(vals) + r" \\")

        lines.append(r"  \\[-4pt]")

        # ARE
        for sm in S_MODES:
            vals = [f"{all_stats[(sm, J, N)]['are']:.2f}"
                    if (sm, J, N) in all_stats else "--"
                    for N in N_values]
            lines.append(f"  & ARE ({S_LABELS[sm]}) & "
                          + " & ".join(vals) + r" \\")

        lines.append(r"  \\[-4pt]")

        # s.e. ratio = sqrt(ARE)
        for sm in S_MODES:
            vals = [f"{np.sqrt(all_stats[(sm, J, N)]['are']):.2f}"
                    if (sm, J, N) in all_stats else "--"
                    for N in N_values]
            lines.append(r"  & s.e.\ ratio (" + S_LABELS[sm] + ") & "
                          + " & ".join(vals) + r" \\")

        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"LaTeX table: {output_path}")


def generate_figure(all_stats, J_values, N_values, model, output_path):
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
        Ns, mle_curve = [], []
        for N in N_values:
            s = next((all_stats[(sm, J, N)] for sm in S_MODES
                      if (sm, J, N) in all_stats), None)
            if s is None:
                continue
            Ns.append(N)
            mle_curve.append(np.sqrt(N) * s["avg_sd_mle"])

        ax.plot(Ns, mle_curve, "o-", label=MLE_LABEL[model],
                color="C0", linewidth=1.5, markersize=6)

        for sm in S_MODES:
            marker, color, lbl = styles[sm]
            xs, ys = [], []
            for N in N_values:
                s = all_stats.get((sm, J, N))
                if s is None:
                    continue
                xs.append(N)
                ys.append(np.sqrt(N) * s["avg_sd_cb"])
            if ys:
                ax.plot(xs, ys, marker, label=lbl,
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


def analyze_one(model, config):
    J_values = config["grid"]["J"]
    N_values = config["grid"]["N"]
    beta = np.array(config["experiment"]["beta_star"])

    print(f"\n### {model.upper()} ###\n")
    all_stats = {}
    for sm in S_MODES:
        for J in J_values:
            for N in N_values:
                data = load_cell(model, sm, J, N)
                if data is None:
                    continue
                s = compute_stats(data)
                all_stats[(sm, J, N)] = s
                print(f"  {sm:>7s}  J={J:>2} N={N:>5}: R={s['R']:>3}  "
                      f"S={s['n_simulations']:>4}  "
                      f"SD_mle={s['avg_sd_mle']:.4f}  "
                      f"SD_cb={s['avg_sd_cb']:.4f}  "
                      f"ARE={s['are']:.2f}  "
                      f"t_cb={s['time_cb']:.3f}s")

    if not all_stats:
        print(f"  no results for {model}")
        return

    out_dir = results_root(model)
    generate_table(all_stats, J_values, N_values, beta, model,
                   out_dir / "table.tex")
    generate_figure(all_stats, J_values, N_values, model,
                    out_dir / "efficiency.pdf")

    header = f"{'S mode':>8} {'J':>3} |" + "".join(
        f"  N={N:>5}  |" for N in N_values)

    def _print_matrix(title, key_fn):
        print(f"\n{'='*len(header)}\n{title}\n{'='*len(header)}")
        print(header)
        print("-" * len(header))
        for sm in S_MODES:
            for J in J_values:
                row = f"{sm:>8} {J:>3} |"
                for N in N_values:
                    s = all_stats.get((sm, J, N))
                    row += f"  {key_fn(s):>6.2f}  |" if s else f"  {'--':>6}  |"
                print(row)

    _print_matrix("ARE (Var_CB / Var_MLE)", lambda s: s['are'])
    _print_matrix("s.e. ratio = sqrt(ARE)", lambda s: np.sqrt(s['are']))
    _print_matrix("Speedup (Time_MLE / Time_CB)",
                  lambda s: (s['time_mle'] / s['time_cb']
                              if s['time_cb'] > 0 else np.inf))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["probit", "logit", "both"],
                        default="both")
    args = parser.parse_args()

    with open(SCRIPT_DIR / "config.yaml") as f:
        config = yaml.safe_load(f)

    models = ["probit", "logit"] if args.model == "both" else [args.model]
    for m in models:
        analyze_one(m, config)


if __name__ == "__main__":
    main()
