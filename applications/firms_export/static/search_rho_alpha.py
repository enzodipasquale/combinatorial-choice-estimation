#!/bin/env python
import os, sys, json, time, argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import yaml
import combest as ce
from combest.estimation.callbacks import adaptive_gurobi_timeout

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "data"))
from search_rho import (build_model, build_correlation_matrix, generate_base_draws,
                         build_errors, compute_moments, make_warmstart_callback,
                         install_constraint_store, filter_by_slack)

config = yaml.safe_load(open(BASE_DIR / "config.yaml"))
app = config["application"]


def save_results(args, rho_values, alpha_values, grid, comm):
    base = BASE_DIR / args.output_dir
    slurm_id = os.environ.get("SLURM_JOB_ID")
    tag = slurm_id if slurm_id else datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "timestamp": datetime.now().isoformat(),
        "slurm_job_id": slurm_id,
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
        "slurm_num_nodes": os.environ.get("SLURM_NNODES"),
        "n_ranks": comm.comm_size,
        "args": vars(args),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    l2 = np.array([[g["l2"] for g in row] for row in grid])
    thetas = np.array([[g["theta"] for g in row] for row in grid])
    times = np.array([[g["time"] for g in row] for row in grid])

    np.savez(out_dir / "results.npz",
             rhos=rho_values, alphas=alpha_values,
             l2=l2, thetas=thetas, times=times)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 6))
        R, A = np.meshgrid(rho_values, alpha_values)
        cs = ax.contourf(R, A, l2, levels=20, cmap="viridis")
        fig.colorbar(cs, ax=ax, label="L2 moment distance")
        imin = np.unravel_index(np.argmin(l2), l2.shape)
        ax.plot(rho_values[imin[1]], alpha_values[imin[0]], "r*", markersize=18,
                label=f"min at ρ={rho_values[imin[1]]:.2f}, α={alpha_values[imin[0]]:.2f}")
        ax.set_xlabel("ρ (spatial correlation)")
        ax.set_ylabel("α (persistence weight)")
        ax.set_title("Moment distance vs (ρ, α)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "rho_alpha_l2.png", dpi=150)
    except ImportError:
        pass

    print(f"\nResults saved to {out_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rho_min", type=float, default=0.0)
    p.add_argument("--rho_max", type=float, default=1.0)
    p.add_argument("--n_rho", type=int, default=11)
    p.add_argument("--alpha_min", type=float, default=0.1)
    p.add_argument("--alpha_max", type=float, default=0.9)
    p.add_argument("--n_alpha", type=int, default=9)
    p.add_argument("--n_sample", type=int, default=app.get("n_sample"))
    p.add_argument("--n_simulations", type=int, default=None)
    p.add_argument("--seed", type=int, default=999)
    p.add_argument("--no_warmstart", action="store_true")
    p.add_argument("--ws_keep_pct", type=float, default=30)
    p.add_argument("--output_dir", type=str, default="results")
    return p.parse_args()


def main():
    args = parse_args()
    if args.n_simulations is not None:
        config["dimensions"]["n_simulations"] = args.n_simulations

    model, input_data, n_dest, n_years = build_model(n_sample=args.n_sample)

    rho_values = np.linspace(args.rho_min, args.rho_max, args.n_rho)
    alpha_values = np.linspace(args.alpha_min, args.alpha_max, args.n_alpha)
    n_items = model.config.dimensions.n_items
    n_agents = model.config.dimensions.n_agents
    n_sims = model.config.dimensions.n_simulations
    agent_counts = model.comm_manager.agent_counts
    comm = model.comm_manager
    warmstart = not args.no_warmstart

    if model.is_root():
        obs_moments = compute_moments(input_data["id_data"]["obs_bundles"], n_dest, n_years)
        A = build_correlation_matrix(input_data)
        base_x, base_z, base_nu = generate_base_draws(n_agents, n_dest, n_items, args.seed)
        base_Ax = np.nan_to_num(base_x @ A.T)
        dw = np.tile(input_data["id_data"]["discount_weights"], (n_sims, 1))
    else:
        obs_moments = None
        base_Ax, base_z, base_nu, dw = None, None, None, None

    stored = []
    if warmstart:
        install_constraint_store(model, stored)

    pt_cb, _ = adaptive_gurobi_timeout(config["callbacks"]["row_gen"])

    grid = []
    total = len(alpha_values) * len(rho_values)
    count = 0

    for ai, alpha in enumerate(alpha_values):
        stored.clear()
        row = []

        for ri, rho in enumerate(rho_values):
            count += 1
            if model.is_root():
                print(f"\n{'='*80}\nα={alpha:.3f}  ρ={rho:.4f}  [{count}/{total}]\n{'='*80}")
            t0 = time.time()

            if model.is_root():
                all_errors = build_errors(rho, alpha, base_Ax, base_z, base_nu, dw, n_dest)
            else:
                all_errors = None
            model.features.local_modular_errors = comm.Scatterv_by_row(all_errors, agent_counts)

            init_cb = None
            if warmstart and ri > 0 and stored:
                init_cb = make_warmstart_callback(list(stored), all_errors)
            stored.clear()

            result = model.row_generation.solve(
                iteration_callback=pt_cb, initialization_callback=init_cb, verbose=True)
            theta = comm.bcast(result.theta_hat if model.is_root() else None)

            if warmstart:
                n_kept, n_total = filter_by_slack(model, stored, args.ws_keep_pct)
                if model.is_root() and n_total > 0:
                    print(f"Warm-start: keeping {n_kept}/{n_total} constraints "
                          f"(tightest {args.ws_keep_pct:.0f}%)")

            sim_bundles = model.subproblems.solve(theta)
            sim_moments = compute_moments(sim_bundles, n_dest, n_years, comm=comm, n_total=n_agents)
            elapsed = time.time() - t0

            if model.is_root():
                l2 = float(np.linalg.norm(sim_moments - obs_moments))
                row.append({"rho": rho, "alpha": alpha, "theta": theta.copy(),
                            "l2": l2, "time": elapsed})
                print(f"Moment L2 distance: {l2:.6f}  ({elapsed:.0f}s)")

        if model.is_root():
            grid.append(row)
            print(f"\n--- α={alpha:.3f} ---")
            print(f"{'rho':>6} | {'L2':>12} | {'time':>6}")
            print("-" * 35)
            for r in row:
                print(f"{r['rho']:6.3f} | {r['l2']:12.6f} | {r['time']:5.0f}s")

    if model.is_root():
        flat = [r for row in grid for r in row]
        best = min(flat, key=lambda r: r["l2"])
        names = model.config.dimensions.covariate_names
        print(f"\n{'='*80}\nBest ρ = {best['rho']:.4f}, α = {best['alpha']:.4f}  "
              f"(L2 = {best['l2']:.6f})")
        for k in sorted(names):
            print(f"  {names[k]:>14s}:  {best['theta'][k]:.6f}")

        save_results(args, rho_values, alpha_values, grid, comm)


if __name__ == "__main__":
    main()
