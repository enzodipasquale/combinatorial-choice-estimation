#!/usr/bin/env python3
import json
import csv
import yaml
import argparse
import numpy as np
from pathlib import Path


SPEC_LABELS = {
    "gross_substitutes": "Gross substitutes",
    "supermodular": "Supermodular",
    "linear_knapsack": "Linear knapsack",
    "quadratic_knapsack": "Quadratic knapsack",
}


def load_all_statistics(results_dir, config):
    specs = list(config["specifications"].keys())
    grid_M, grid_N = config["grid"]["M"], config["grid"]["N"]
    stats = {}
    for spec in specs:
        stats[spec] = {}
        for M in grid_M:
            stats[spec][M] = {}
            for N in grid_N:
                f = results_dir / f"stats_{spec}_N{N}_M{M}.json"
                if f.exists():
                    with open(f) as fp:
                        stats[spec][M][N] = json.load(fp)
                else:
                    stats[spec][M][N] = None
    return stats


def fmt(val, decimals=3):
    return "---" if val is None or np.isnan(val) else f"{val:.{decimals}f}"


def get_val(local, hpc, key):
    if hpc is not None and key in hpc:
        return hpc[key]
    if local is not None and key in local:
        return local[key]
    return np.nan


def generate_table_csv(local_stats, hpc_stats, config, output_path):
    specs = list(config["specifications"].keys())
    grid_M, grid_N = config["grid"]["M"], config["grid"]["N"]

    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Specification", "M", "N",
                     "Bias α", "RMSE α", "Bias λ", "RMSE λ",
                     "Runtime (local)", "Runtime (HPC)"])
        for spec in specs:
            for M in grid_M:
                for N in grid_N:
                    loc = local_stats[spec][M][N]
                    hpc = hpc_stats[spec][M][N]
                    s = hpc or loc
                    if s is None:
                        w.writerow([spec, M, N] + ["---"] * 6)
                    else:
                        rt_local = loc.get("runtime", np.nan) if loc else np.nan
                        rt_hpc = hpc.get("runtime", np.nan) if hpc else np.nan
                        w.writerow([spec, M, N,
                                    fmt(s.get("bias_alpha", np.nan)),
                                    fmt(s.get("rmse_alpha", np.nan)),
                                    fmt(s.get("bias_lambda", np.nan)),
                                    fmt(s.get("rmse_lambda", np.nan)),
                                    fmt(rt_local, 2),
                                    fmt(rt_hpc, 2)])


def generate_table_latex(local_stats, hpc_stats, config, output_path):
    specs = list(config["specifications"].keys())
    grid_M, grid_N = config["grid"]["M"], config["grid"]["N"]
    n_N = len(grid_N)

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\footnotesize",
        "\\caption{Bias, RMSE, and runtime}",
        "\\label{tab:benchmarks}",
        "\\begin{threeparttable}",
        "\\begin{tabular}{l" + " c" * (len(grid_M) * n_N) + "}",
        "\\toprule",
        " & " + " & ".join(f"\\multicolumn{{{n_N}}}{{c}}{{$M = {M}$}}" for M in grid_M) + " \\\\",
    ]

    cmidrules = []
    for i, M in enumerate(grid_M):
        start = 2 + i * n_N
        cmidrules.append(f"\\cmidrule(lr){{{start}-{start + n_N - 1}}}")
    lines.append(" ".join(cmidrules))
    lines.append("$N$ & " + " & ".join([str(N) for N in grid_N] * len(grid_M)) + " \\\\")
    lines.append("\\midrule")

    for spec in specs:
        label = SPEC_LABELS.get(spec, spec)
        ncols = len(grid_M) * n_N + 1
        lines.append(f"\\multicolumn{{{ncols}}}{{l}}{{\\textit{{{label}}}}} \\\\[2pt]")

        has_lambda = config["specifications"][spec].get("lambda") is not None
        rows = [
            ("Bias $\\alpha$", "bias_alpha", 3),
            ("RMSE $\\alpha$", "rmse_alpha", 3),
        ]
        if has_lambda:
            rows += [
                ("Bias $\\lambda$", "bias_lambda", 3),
                ("RMSE $\\lambda$", "rmse_lambda", 3),
            ]

        for row_label, key, dec in rows:
            cells = []
            for M in grid_M:
                for N in grid_N:
                    val = get_val(local_stats[spec][M][N], hpc_stats[spec][M][N], key)
                    cells.append(fmt(val, dec))
            lines.append(f"{row_label} & " + " & ".join(cells) + " \\\\")

        for rt_label, src_stats in [("Runtime (local)", local_stats), ("Runtime (HPC)", hpc_stats)]:
            cells = []
            has_any = False
            for M in grid_M:
                for N in grid_N:
                    s = src_stats[spec][M][N]
                    val = s.get("runtime", np.nan) if s else np.nan
                    if not np.isnan(val):
                        has_any = True
                    cells.append(fmt(val, 2))
            if has_any:
                lines.append(f"{rt_label} & " + " & ".join(cells) + " \\\\")

        if lines[-1].endswith("\\\\"):
            lines[-1] = lines[-1][:-2] + "\\\\[4pt]"

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\begin{tablenotes}",
        "\\footnotesize",
        "\\item \\textit{Notes:} Averages over 50 replications. $S = 1$, $\\rho = 0.5$. "
        "$\\alpha$: modular agent feature coefficient; $\\lambda$: non-modular feature coefficient. "
        "Runtime in seconds (point estimation).",
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

    local_dir = base / "results" / "local"
    hpc_dir = base / "results" / "hpc"
    local_dir.mkdir(parents=True, exist_ok=True)
    hpc_dir.mkdir(parents=True, exist_ok=True)

    local_stats = load_all_statistics(local_dir, config)
    hpc_stats = load_all_statistics(hpc_dir, config)

    generate_table_csv(local_stats, hpc_stats, config, base / "results" / "table.csv")
    print(f"Generated CSV table: {base / 'results' / 'table.csv'}")
    generate_table_latex(local_stats, hpc_stats, config, base / "results" / "table.tex")
    print(f"Generated LaTeX table: {base / 'results' / 'table.tex'}")


if __name__ == "__main__":
    main()
