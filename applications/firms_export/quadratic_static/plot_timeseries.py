#!/bin/env python
"""
Plot observed vs predicted aggregate exports to top-10 destinations over time.

Produces two figures:
  1. Observed: number of firms exporting to each top-10 destination per year
  2. Predicted: same, using simulated bundles at estimated theta

Usage:
  mpirun -n 4 python plot_timeseries.py [--n_sample 10000]
"""
import sys, argparse
from pathlib import Path
import numpy as np
import yaml
import combest as ce
from combest.subproblems.registry.quadratic_obj.quadratic_supermodular.min_cut import (
    QuadraticSupermodularMinCutSolver,
)
from oracles import build_oracles
from prepare_data import main as load_data

BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / "config.yaml") as f:
    CFG = yaml.safe_load(f)

COUNTRY = CFG["data"]["country"]
KEEP_TOP = CFG["data"]["keep_top"]
START_BUFFER = CFG["data"]["start_buffer"]
SEED = CFG["estimation"]["seed"]
SIGMA_PERM = CFG["estimation"]["sigma_perm"]
SIGMA_TRANS = CFG["estimation"]["sigma_trans"]
N_SIMULATIONS = CFG["estimation"]["n_simulations"]
MAX_RG_ITERS = CFG["estimation"]["max_rg_iters"]

N_COV = 5
NAMES = ["rev", "fc", "dist", "syn", "syn_d"]
TOP_K = 10


def aggregate_by_year(bundles, year_idx, all_years, n_dest):
    """Number of firms exporting to each destination per year."""
    n_years = len(all_years)
    counts = np.zeros((n_years, n_dest))
    for i in range(len(bundles)):
        ti = year_idx[i]
        counts[ti] += bundles[i].astype(float)
    return counts


def aggregate_by_year_distributed(bundles, year_idx, all_years, n_dest, comm):
    """Aggregate with MPI reduction."""
    n_years = len(all_years)
    local_counts = np.zeros((n_years, n_dest))
    for i in range(len(bundles)):
        ti = year_idx[i]
        local_counts[ti] += bundles[i].astype(float)

    total_counts = comm.Reduce(local_counts)
    return total_counts


def plot_timeseries(counts, all_years, destinations, top_idx, title, filename):
    """Plot time series of aggregate exports for top destinations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    for j in top_idx:
        ax.plot(all_years, counts[:, j], marker="o", markersize=3,
                label=destinations[j])
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of firms exporting")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved {filename}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_sample", type=int, default=CFG["data"]["n_sample"])
    return p.parse_args()


def main():
    args = parse_args()

    model = ce.Model()
    is_root = model.is_root()
    comm = model.comm_manager

    # --- Load data ---
    if is_root:
        ctx = load_data(COUNTRY, KEEP_TOP, start_buffer=START_BUFFER,
                        n_sample=args.n_sample)
        n_obs = ctx["n_obs"]
        M = ctx["M"]
        firm_idx = ctx["firm_idx"]
        year_idx = ctx["year_idx"]
        all_years = ctx["all_years"]
        destinations = ctx["destinations"]
        dist_home = ctx["dist_home"]
        syn_chars = ctx["syn_chars"]

        modular = np.stack([
            ctx["rev_chars"],
            -np.ones((n_obs, M)),
            -np.broadcast_to(dist_home[None, :], (n_obs, M)),
        ], axis=-1)

        C = syn_chars
        C_d = C * (dist_home[:, None] + dist_home[None, :]) / 2
        quadratic = np.stack([C, C_d], axis=-1)

        input_data = {
            "id_data": {
                "obs_bundles": ctx["obs_bundles"],
                "modular": modular,
                "constraint_mask": None,
                "firm_idx": firm_idx,
                "year_idx": year_idx,
            },
            "item_data": {
                "quadratic": quadratic,
            },
        }
    else:
        input_data, n_obs, M, firm_idx = None, None, None, None
        all_years, destinations, year_idx = None, None, None

    n_obs = comm.bcast(n_obs)
    M = comm.bcast(M)
    all_years = comm.bcast(all_years)
    destinations = comm.bcast(destinations)

    cfg = {
        "dimensions": {"n_obs": n_obs, "n_items": M,
                       "n_covariates": N_COV, "n_simulations": N_SIMULATIONS},
        "row_generation": {"max_iters": MAX_RG_ITERS, "tolerance": 1e-6,
                          "theta_bounds": {"lb": 0}},
    }
    model.load_config(cfg)
    model.data.load_and_distribute_input_data(input_data)

    err_oracle = build_oracles(
        model, firm_idx, seed=SEED,
        sigma_perm=SIGMA_PERM, sigma_trans=SIGMA_TRANS)

    model.subproblems.load_solver(QuadraticSupermodularMinCutSolver)
    model.subproblems.initialize_solver()
    model.features.build_quadratic_covariates_from_data()
    model.features.set_error_oracle(err_oracle)

    # --- Estimate theta ---
    if is_root:
        print(f"Data: {COUNTRY}, top {KEEP_TOP}, N={n_obs}, M={M}")
        print("Estimating theta...")

    result = model.point_estimation.n_slack.solve(
        initialize_solver=False, verbose=is_root)
    theta = comm.bcast(result.theta_hat if is_root else None)

    if is_root:
        print(f"\ntheta_hat = {theta}")
        for j, name in enumerate(NAMES):
            print(f"  {name:>8} = {theta[j]:+.6f}")

    # --- Simulate bundles at estimated theta ---
    sim_bundles = model.subproblems.solve(theta)

    # --- Aggregate by year ---
    ld = model.data.local_data
    local_year_idx = ld.id_data["year_idx"]

    obs_counts = aggregate_by_year_distributed(
        ld.id_data["obs_bundles"], local_year_idx, all_years, M, comm)
    sim_counts = aggregate_by_year_distributed(
        sim_bundles, local_year_idx, all_years, M, comm)

    # --- Plot on root ---
    if is_root:
        # Top 10 destinations by total observed exports
        total_obs = obs_counts.sum(axis=0)
        top_idx = np.argsort(total_obs)[::-1][:TOP_K]

        print(f"\nTop {TOP_K} destinations by total exports:")
        for rank, j in enumerate(top_idx):
            print(f"  {rank+1}. {destinations[j]:>5}  "
                  f"total={int(total_obs[j]):>6}")

        out_dir = BASE_DIR / "figures"
        out_dir.mkdir(exist_ok=True)

        plot_timeseries(obs_counts, all_years, destinations, top_idx,
                        f"Observed: firms exporting to top {TOP_K} destinations ({COUNTRY})",
                        out_dir / "observed_timeseries.png")

        plot_timeseries(sim_counts, all_years, destinations, top_idx,
                        f"Predicted: firms exporting to top {TOP_K} destinations ({COUNTRY})",
                        out_dir / "predicted_timeseries.png")


if __name__ == "__main__":
    main()
