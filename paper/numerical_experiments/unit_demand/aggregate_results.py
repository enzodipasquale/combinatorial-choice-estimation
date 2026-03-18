#!/usr/bin/env python3
import json
import csv
import yaml
import argparse
import numpy as np
from pathlib import Path


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
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "---"
    return f"{val:.{decimals}f}"


def generate_table_csv(stats, config, output_path):
    dgps = list(config["dgps"].keys())
    grid_J, grid_N = config["grid"]["J"], config["grid"]["N"]

    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["DGP", "J", "N",
                     "RMSE MLE", "RMSE Combest",
                     "N*MSE MLE", "N*MSE Combest",
                     "Eff. Ratio",
                     "Runtime MLE", "Runtime Combest"])
        for dgp in dgps:
            for J in grid_J:
                for N in grid_N:
                    s = stats[dgp][J][N]
                    if s is None:
                        w.writerow([dgp, J, N] + ["---"] * 7)
                    else:
                        w.writerow([dgp, J, N,
                                    fmt(s.get("mle_rmse_mean"), 4),
                                    fmt(s.get("combest_rmse_mean"), 4),
                                    fmt(s.get("N_mse_mle"), 2),
                                    fmt(s.get("N_mse_combest"), 2),
                                    fmt(s.get("efficiency_ratio_total"), 3),
                                    fmt(s.get("runtime_mle"), 2),
                                    fmt(s.get("runtime_combest"), 2)])


def generate_table_latex(stats, config, output_path):
    dgps = list(config["dgps"].keys())
    grid_J, grid_N = config["grid"]["J"], config["grid"]["N"]
    exp = config["experiment"]

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\footnotesize",
        "\\caption{Asymptotic efficiency: combinatorial estimator vs.\\ simulated MLE (probit)}",
        "\\label{tab:efficiency}",
        "\\begin{threeparttable}",
        "\\begin{tabular}{l" + " c" * len(grid_N) + "}",
        "\\toprule",
        "$N$ & " + " & ".join(str(N) for N in grid_N) + " \\\\",
        "\\midrule",
    ]

    for dgp in dgps:
        for J in grid_J:
            rows = [
                ("RMSE (MLE)", "mle_rmse_mean", 4),
                ("RMSE (Combest)", "combest_rmse_mean", 4),
                ("$N \\cdot$ MSE (MLE)", "N_mse_mle", 2),
                ("$N \\cdot$ MSE (Combest)", "N_mse_combest", 2),
                ("Efficiency ratio", "efficiency_ratio_total", 3),
                ("Runtime (MLE)", "runtime_mle", 2),
                ("Runtime (Combest)", "runtime_combest", 2),
            ]

            for row_label, key, dec in rows:
                cells = []
                for N in grid_N:
                    s = stats[dgp][J][N]
                    val = s.get(key) if s else None
                    cells.append(fmt(val, dec))
                lines.append(f"{row_label} & " + " & ".join(cells) + " \\\\")

    beta_str = ", ".join(str(b) for b in exp["beta_star"])
    sigma_str = exp.get("sigma", 1.0)
    ghk_str = exp.get("ghk_draws", 200)
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\begin{tablenotes}",
        "\\footnotesize",
        f"\\item \\textit{{Notes:}} "
        f"$K = {exp['K']}$, $J = {grid_J[0]}$, "
        f"$\\beta^* = ({beta_str})$, "
        f"$\\sigma = {sigma_str}$, "
        f"GHK draws $= {ghk_str}$. "
        f"Efficiency ratio $= $ MSE(Combest) / MSE(MLE). "
        f"$N \\cdot$ MSE should stabilize under $\\sqrt{{N}}$-consistency. "
        f"Averages over {exp['n_replications']} replications.",
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
