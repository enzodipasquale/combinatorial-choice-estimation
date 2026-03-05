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
from prepare_data import main as load_data, build_input_data
from solver import (DiscountedJointQuadKnapsackSolver, discounted_covariates_oracle,
                    count_covariates)
from run_estimation import build_covariate_meta

config = yaml.safe_load(open(BASE_DIR / "config.yaml"))
app = config["application"]


def build_model(n_sample=None):
    model = ce.Model()

    if model.is_root():
        ctx = load_data(app["country"], app["keep_top"], app["discount"], n_sample=n_sample)
        input_data = build_input_data(ctx)
        covariate_names, lbs = build_covariate_meta(input_data)

        config["dimensions"].update(
            n_obs=ctx["n_obs"], n_items=ctx["n_items"],
            n_covariates=count_covariates(input_data),
            covariate_names=covariate_names)
        config["row_generation"]["theta_bounds"]["lbs"] = lbs
        n_dest, n_years = ctx["n_dest"], ctx["n_years"]
    else:
        input_data, n_dest, n_years = None, None, None

    n_dest = model.comm_manager.bcast(n_dest)
    n_years = model.comm_manager.bcast(n_years)
    model.load_config(config)
    model.data.load_and_distribute_input_data(input_data)

    model.features.set_covariates_oracle(discounted_covariates_oracle)
    model.features.build_local_modular_error_oracle(seed=app["seed"])
    model.subproblems.load_solver(DiscountedJointQuadKnapsackSolver)
    model.subproblems.initialize_solver()

    return model, input_data, n_dest, n_years


def build_correlation_matrix(input_data):
    A = input_data["item_data"]["quadratic_2d"][:, :, 0].copy()
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    return A / np.where(norms > 1e-10, norms, 1.0)


def generate_base_draws(n_agents, n_dest, n_items, seed=999):
    x = np.zeros((n_agents, n_dest))
    z = np.zeros((n_agents, n_dest))
    nu = np.zeros((n_agents, n_items))
    for i in range(n_agents):
        rng = np.random.default_rng((seed, i))
        x[i] = rng.normal(0, 1, n_dest)
        z[i] = rng.normal(0, 1, n_dest)
        nu[i] = rng.normal(0, 1, n_items)
    return x, z, nu


def build_errors(rho, alpha, base_Ax, base_z, base_nu, dw, n_dest):
    n_years = dw.shape[1]
    eps = np.sqrt(rho) * base_Ax + np.sqrt(1 - rho) * base_z
    eps = np.tile(eps[:, None, :], (1, n_years, 1)).reshape(eps.shape[0], -1)
    total = np.sqrt(alpha) * eps + np.sqrt(1 - alpha) * base_nu
    total *= np.repeat(dw, n_dest, axis=1)
    return total


def make_warmstart_callback(stored, all_errors):
    def cb(rg):
        for idx, bun, cov in stored:
            err = (all_errors[idx] * bun).sum(-1)
            rg.add_master_constraints(idx, bun, cov, err)
    return cb


def install_constraint_store(model, stored):
    solver = model.row_generation
    original = solver.add_master_constraints
    def storing(indices, bundles, covariates, errors):
        stored.append((indices.copy(), bundles.copy(), covariates.copy()))
        return original(indices, bundles, covariates, errors)
    if model.is_root():
        solver.add_master_constraints = storing


def filter_by_slack(model, stored, keep_pct):
    if not model.is_root() or not stored:
        return 0, 0
    cc = model.row_generation.all_concatenated_constraints
    if cc is None:
        return 0, 0
    slacks = cc.Slack
    total = len(slacks)
    threshold = np.percentile(slacks, keep_pct)
    offset, filtered = 0, []
    for idx, bun, cov in stored:
        n = len(idx)
        keep = slacks[offset:offset + n] <= threshold
        if keep.any():
            filtered.append((idx[keep], bun[keep], cov[keep]))
        offset += n
    n_kept = sum(len(f[0]) for f in filtered)
    stored.clear()
    stored.extend(filtered)
    return n_kept, total


def compute_moments(bundles, n_dest, n_years, comm=None, n_total=None):
    B = bundles.reshape(-1, n_years, n_dest).astype(float)
    pair = np.einsum('itj,itl->jl', B, B)
    pers = np.einsum('itj,itj->j', B[:, 1:], B[:, :-1])
    if comm is not None:
        pair = comm.Reduce(pair)
        pers = comm.Reduce(pers)
        if not comm.is_root():
            return None
    n = n_total if n_total is not None else bundles.shape[0]
    return np.concatenate([pair[np.triu_indices(n_dest, k=1)] / (n * n_years),
                           pers / (n * (n_years - 1))])


def save_results(args, results, comm):
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

    np.savez(out_dir / "results.npz",
             rhos=np.array([r["rho"] for r in results]),
             l2=np.array([r["l2"] for r in results]),
             thetas=np.array([r["theta"] for r in results]),
             times=np.array([r["time"] for r in results]),
             alpha=args.alpha)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rhos = [r["rho"] for r in results]
        dists = [r["l2"] for r in results]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(rhos, dists, "o-", linewidth=2, markersize=8)
        imin = int(np.argmin(dists))
        ax.plot(rhos[imin], dists[imin], "r*", markersize=18,
                label=f"min at ρ={rhos[imin]:.2f}")
        ax.set_xlabel("ρ (error correlation)")
        ax.set_ylabel("L2 moment distance")
        ax.set_title("Moment distance vs ρ")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "rho_vs_l2.png", dpi=150)
    except ImportError:
        pass

    print(f"\nResults saved to {out_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rho_min", type=float, default=0.0)
    p.add_argument("--rho_max", type=float, default=1.0)
    p.add_argument("--n_rho", type=int, default=21)
    p.add_argument("--alpha", type=float, default=0.5)
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

    results = []
    for i, rho in enumerate(rho_values):
        if model.is_root():
            print(f"\n{'='*80}\nRHO = {rho:.4f}  [{i+1}/{len(rho_values)}]\n{'='*80}")
        t0 = time.time()

        if model.is_root():
            all_errors = build_errors(rho, args.alpha, base_Ax, base_z, base_nu, dw, n_dest)
        else:
            all_errors = None
        model.features.local_modular_errors = comm.Scatterv_by_row(all_errors, agent_counts)

        init_cb = None
        if warmstart and i > 0 and stored:
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
            results.append({"rho": rho, "theta": theta.copy(), "l2": l2, "time": elapsed})
            print(f"Moment L2 distance: {l2:.6f}  ({elapsed:.0f}s)")

    if model.is_root():
        print(f"\n{'='*80}\nSUMMARY\n{'='*80}")
        print(f"{'rho':>6} | {'L2':>12} | {'time':>6} | theta")
        print("-" * 80)
        for r in results:
            theta_str = "  ".join(f"{v:8.4f}" for v in r["theta"])
            print(f"{r['rho']:6.3f} | {r['l2']:12.6f} | {r['time']:5.0f}s | {theta_str}")

        best = min(results, key=lambda r: r["l2"])
        names = model.config.dimensions.covariate_names
        print(f"\nBest ρ = {best['rho']:.4f}  (L2 = {best['l2']:.6f})")
        for k in sorted(names):
            print(f"  {names[k]:>14s}:  {best['theta'][k]:.6f}")

        save_results(args, results, comm)


if __name__ == "__main__":
    main()
