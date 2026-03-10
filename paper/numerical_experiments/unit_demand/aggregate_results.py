#!/usr/bin/env python3
import json
import csv
import yaml
import argparse
import numpy as np
from pathlib import Path


DGP_LABELS = {"logit": "Logit", "probit": "Probit"}


def load_all_statistics(results_dir, config):
    dgps = list(config["dgps"].keys())
    grid_J, grid_N = config["grid"]["J"], config["grid"]["N"]
    stats = {}
    for dgp in dgps:
        stats[dgp] = {}
        for J in grid_J:
            stats[dgp][J] = {}
            for N in grid_N:
                f = results_dir / f"stats_{dgp}_N{N}_J{J}.json"
                if f.exists():
                    with open(f) as fp:
                        stats[dgp][J][N] = json.load(fp)
                else:
                    stats[dgp][J][N] = None
    return stats


def fmt(val, decimals=3):
    return "---" if val is None or np.isnan(val) else f"{val:.{decimals}f}"


def generate_table_csv(stats, config, output_path):
    dgps = list(config["dgps"].keys())
    grid_J, grid_N = config["grid"]["J"], config["grid"]["N"]

    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["DGP", "J", "N",
                     "Bias MLE", "RMSE MLE", "Runtime MLE",
                     "Bias Combest", "RMSE Combest", "Runtime Combest"])
        for dgp in dgps:
            for J in grid_J:
                for N in grid_N:
                    s = stats[dgp][J][N]
                    if s is None:
                        w.writerow([dgp, J, N] + ["---"] * 6)
                    else:
                        w.writerow([dgp, J, N,
                                    fmt(s["bias_mle"]), fmt(s["rmse_mle"]),
                                    fmt(s["runtime_mle"], 2),
                                    fmt(s["bias_combest"]), fmt(s["rmse_combest"]),
                                    fmt(s["runtime_combest"], 2)])


def generate_table_latex(stats, config, output_path):
    dgps = list(config["dgps"].keys())
    grid_J, grid_N = config["grid"]["J"], config["grid"]["N"]
    n_N = len(grid_N)
    exp = config["experiment"]

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\footnotesize",
        "\\caption{Comparison of MLE and combinatorial estimator: unit demand}",
        "\\label{tab:unit_demand}",
        "\\begin{threeparttable}",
        "\\begin{tabular}{l" + " c" * (len(grid_J) * n_N) + "}",
        "\\toprule",
        " & " + " & ".join(
            f"\\multicolumn{{{n_N}}}{{c}}{{$J = {J}$}}" for J in grid_J) + " \\\\",
    ]

    cmidrules = []
    for i, J in enumerate(grid_J):
        start = 2 + i * n_N
        cmidrules.append(f"\\cmidrule(lr){{{start}-{start + n_N - 1}}}")
    lines.append(" ".join(cmidrules))
    lines.append("$N$ & " + " & ".join([str(N) for N in grid_N] * len(grid_J)) + " \\\\")
    lines.append("\\midrule")

    for dgp in dgps:
        label = DGP_LABELS.get(dgp, dgp)
        ncols = len(grid_J) * n_N + 1
        lines.append(f"\\multicolumn{{{ncols}}}{{l}}{{\\textit{{{label}}}}} \\\\[2pt]")

        rows = [
            ("Bias (MLE)", "bias_mle", 3),
            ("Bias (Combest)", "bias_combest", 3),
            ("RMSE (MLE)", "rmse_mle", 3),
            ("RMSE (Combest)", "rmse_combest", 3),
            ("Runtime (MLE)", "runtime_mle", 2),
            ("Runtime (Combest)", "runtime_combest", 2),
        ]

        for row_label, key, dec in rows:
            cells = []
            for J in grid_J:
                for N in grid_N:
                    s = stats[dgp][J][N]
                    val = s.get(key, np.nan) if s else np.nan
                    cells.append(fmt(val, dec))
            lines.append(f"{row_label} & " + " & ".join(cells) + " \\\\")

        if lines[-1].endswith("\\\\"):
            lines[-1] = lines[-1][:-2] + "\\\\[4pt]"

    beta_str = ", ".join(str(b) for b in exp["beta_star"])
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\begin{tablenotes}",
        "\\footnotesize",
        f"\\item \\textit{{Notes:}} Averages over replications. "
        f"$K = {exp['K']}$, $\\beta^* = ({beta_str})$. "
        f"Runtime in seconds.",
        "\\end{tablenotes}",
        "\\end{threeparttable}",
        "\\end{table}",
    ]

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    base = Path(__file__).parent
    with open(base / args.config) as f:
        config = yaml.safe_load(f)

    results_dir = base / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    stats = load_all_statistics(results_dir, config)

    generate_table_csv(stats, config, results_dir / "table.csv")
    print(f"Generated CSV: {results_dir / 'table.csv'}")
    generate_table_latex(stats, config, results_dir / "table.tex")
    print(f"Generated LaTeX: {results_dir / 'table.tex'}")


if __name__ == "__main__":
    main()
