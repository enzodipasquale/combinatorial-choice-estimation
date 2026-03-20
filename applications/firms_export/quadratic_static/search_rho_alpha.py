#!/bin/env python
"""
Grid search over (rho, alpha) for the quadratic static model.

rho:   spatial correlation of persistent error component
alpha: weight on persistent vs transitory error
"""
import os, sys, json, time, argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import yaml
import combest as ce
from combest.subproblems.registry.quadratic_obj.quadratic_supermodular.min_cut import (
    QuadraticSupermodularMinCutSolver,
)
from prepare_data import main as load_data

BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / "config.yaml") as f:
    CFG = yaml.safe_load(f)

COUNTRY = CFG["data"]["country"]
KEEP_TOP = CFG["data"]["keep_top"]
START_BUFFER = CFG["data"]["start_buffer"]
N_SIMULATIONS = CFG["estimation"]["n_simulations"]
MAX_RG_ITERS = CFG["estimation"]["max_rg_iters"]

N_COV = 5
NAMES = ["rev", "fc", "dist", "syn", "syn_d"]


# ---------------------------------------------------------------------------
# Error construction
# ---------------------------------------------------------------------------

def build_correlation_matrix(syn_chars):
    """Normalize rows of proximity matrix to unit norm."""
    A = syn_chars.copy()
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    return A / np.where(norms > 1e-10, norms, 1.0)


def generate_base_draws(n_obs, M, firm_idx, seed=999):
    """One persistent draw per firm, one transitory draw per observation."""
    unique_firms = np.unique(firm_idx)
    firm_x = {}
    firm_z = {}
    for f in unique_firms:
        rng = np.random.default_rng((seed, int(f), 0))
        firm_x[f] = rng.normal(0, 1, M)
        firm_z[f] = rng.normal(0, 1, M)

    base_x = np.zeros((n_obs, M))
    base_z = np.zeros((n_obs, M))
    base_nu = np.zeros((n_obs, M))
    for i in range(n_obs):
        fi = firm_idx[i]
        base_x[i] = firm_x[fi]
        base_z[i] = firm_z[fi]
        base_nu[i] = np.random.default_rng((seed, i, 1)).normal(0, 1, M)

    return base_x, base_z, base_nu


def build_errors(rho, alpha, base_Ax, base_z, base_nu):
    """Build error matrix for given (rho, alpha)."""
    eps = np.sqrt(rho) * base_Ax + np.sqrt(1 - rho) * base_z
    return np.sqrt(alpha) * eps + np.sqrt(1 - alpha) * base_nu


# ---------------------------------------------------------------------------
# Moments
# ---------------------------------------------------------------------------

def compute_moments(bundles, firm_idx, year_idx):
    """Pairwise co-export + marginal + within-firm persistence."""
    B = bundles.astype(float)
    n_obs, M = B.shape

    # Pairwise co-export frequencies
    pair = (B.T @ B) / n_obs
    pairwise = pair[np.triu_indices(M, k=1)]

    # Marginal export frequencies
    marginal = B.mean(axis=0)

    # Within-firm persistence
    persistence = np.zeros(M)
    n_pairs = 0
    order = np.lexsort((year_idx, firm_idx))
    for k in range(1, len(order)):
        i_prev, i_curr = order[k - 1], order[k]
        if firm_idx[i_prev] == firm_idx[i_curr] and year_idx[i_curr] == year_idx[i_prev] + 1:
            persistence += B[i_prev] * B[i_curr]
            n_pairs += 1
    if n_pairs > 0:
        persistence /= n_pairs

    return np.concatenate([pairwise, marginal, persistence])


def compute_moments_distributed(bundles, firm_idx, year_idx, comm, n_total):
    """Compute moments with MPI reduction."""
    B = bundles.astype(float)
    n_local, M = B.shape

    pair = B.T @ B
    marginal_sum = B.sum(axis=0)

    persistence_sum = np.zeros(M)
    n_pairs_local = 0
    order = np.lexsort((year_idx, firm_idx))
    for k in range(1, len(order)):
        i_prev, i_curr = order[k - 1], order[k]
        if firm_idx[i_prev] == firm_idx[i_curr] and year_idx[i_curr] == year_idx[i_prev] + 1:
            persistence_sum += B[i_prev] * B[i_curr]
            n_pairs_local += 1

    pair = comm.Reduce(pair)
    marginal_sum = comm.Reduce(marginal_sum)
    persistence_sum = comm.Reduce(persistence_sum)
    n_pairs_total = comm.Reduce(np.array(n_pairs_local, dtype=float))

    if not comm.is_root():
        return None

    pairwise = pair[np.triu_indices(M, k=1)] / n_total
    marginal = marginal_sum / n_total
    persistence = persistence_sum / max(n_pairs_total, 1)

    return np.concatenate([pairwise, marginal, persistence])


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(rho_values, alpha_values, grid, comm, output_dir):
    slurm_id = os.environ.get("SLURM_JOB_ID")
    tag = slurm_id if slurm_id else datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = output_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    l2 = np.array([[g["l2"] for g in row] for row in grid])
    thetas = np.array([[g["theta"] for g in row] for row in grid])
    times = np.array([[g["time"] for g in row] for row in grid])

    np.savez(out_dir / "results.npz",
             rhos=rho_values, alphas=alpha_values,
             l2=l2, thetas=thetas, times=times)

    meta = {
        "timestamp": datetime.now().isoformat(),
        "slurm_job_id": slurm_id,
        "n_ranks": comm.comm_size,
        "country": COUNTRY,
        "keep_top": KEEP_TOP,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

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
                label=f"min at rho={rho_values[imin[1]]:.2f}, alpha={alpha_values[imin[0]]:.2f}")
        ax.set_xlabel("rho (spatial correlation)")
        ax.set_ylabel("alpha (persistence weight)")
        ax.set_title("Moment distance vs (rho, alpha)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "rho_alpha_l2.png", dpi=150)
    except ImportError:
        pass

    print(f"\nResults saved to {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    s = CFG["search"]
    p = argparse.ArgumentParser()
    p.add_argument("--rho_min", type=float, default=s["rho_min"])
    p.add_argument("--rho_max", type=float, default=s["rho_max"])
    p.add_argument("--n_rho", type=int, default=s["n_rho"])
    p.add_argument("--alpha_min", type=float, default=s["alpha_min"])
    p.add_argument("--alpha_max", type=float, default=s["alpha_max"])
    p.add_argument("--n_alpha", type=int, default=s["n_alpha"])
    p.add_argument("--n_sample", type=int, default=CFG["data"]["n_sample"])
    p.add_argument("--seed", type=int, default=s["seed"])
    p.add_argument("--output_dir", type=str, default="results")
    return p.parse_args()


def main():
    args = parse_args()

    model = ce.Model()
    is_root = model.is_root()
    comm = model.comm_manager

    # --- Load data and build model ---
    if is_root:
        ctx = load_data(COUNTRY, KEEP_TOP, start_buffer=START_BUFFER,
                        n_sample=args.n_sample)
        n_obs = ctx["n_obs"]
        M = ctx["M"]
        firm_idx = ctx["firm_idx"]
        year_idx = ctx["year_idx"]
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
        input_data, n_obs, M, firm_idx, year_idx, syn_chars = (
            None, None, None, None, None, None)

    n_obs = comm.bcast(n_obs)
    M = comm.bcast(M)

    cfg = {
        "dimensions": {"n_obs": n_obs, "n_items": M,
                       "n_covariates": N_COV, "n_simulations": N_SIMULATIONS},
        "row_generation": {"max_iters": MAX_RG_ITERS, "tolerance": 1e-6,
                          "theta_bounds": {"lb": 0}},
    }
    model.load_config(cfg)
    model.data.load_and_distribute_input_data(input_data)

    model.subproblems.load_solver(QuadraticSupermodularMinCutSolver)
    model.subproblems.initialize_solver()
    model.features.build_quadratic_covariates_from_data()

    # --- Prepare base draws and observed moments ---
    if is_root:
        A = build_correlation_matrix(syn_chars)
        base_x, base_z, base_nu = generate_base_draws(
            n_obs, M, firm_idx, seed=args.seed)
        base_Ax = np.nan_to_num(base_x @ A.T)
        obs_moments = compute_moments(ctx["obs_bundles"], firm_idx, year_idx)

        print(f"Data: {COUNTRY}, top {KEEP_TOP} destinations")
        print(f"  N={n_obs}, M={M}")
        print(f"  moments vector length: {len(obs_moments)}")
    else:
        base_Ax, base_z, base_nu = None, None, None

    # --- Get local data for moment computation ---
    ld = model.data.local_data
    local_firm_idx = ld.id_data["firm_idx"]
    local_year_idx = ld.id_data["year_idx"]
    agent_counts = comm.agent_counts
    n_agents = model.config.dimensions.n_agents

    # --- Grid search ---
    rho_values = np.linspace(args.rho_min, args.rho_max, args.n_rho)
    alpha_values = np.linspace(args.alpha_min, args.alpha_max, args.n_alpha)

    grid = []
    total = len(alpha_values) * len(rho_values)
    count = 0

    for ai, alpha in enumerate(alpha_values):
        row = []
        for ri, rho in enumerate(rho_values):
            count += 1
            if is_root:
                print(f"\n{'='*70}")
                print(f"alpha={alpha:.3f}  rho={rho:.3f}  [{count}/{total}]")
                print(f"{'='*70}")
            t0 = time.time()

            # Build and distribute errors
            if is_root:
                all_errors = build_errors(rho, alpha, base_Ax, base_z, base_nu)
            else:
                all_errors = None
            local_errors = comm.Scatterv_by_row(all_errors, agent_counts)
            model.features.local_modular_errors = local_errors

            # Error oracle (consistent with local_modular_errors)
            errors_ref = local_errors

            def _make_err_oracle(err):
                def error_oracle(bundles, ids):
                    return (err[ids] * bundles.astype(float)).sum(-1)
                return error_oracle
            model.features.set_error_oracle(_make_err_oracle(errors_ref))

            # Estimate theta
            result = model.point_estimation.n_slack.solve(
                initialize_solver=False, verbose=is_root)
            theta = comm.bcast(result.theta_hat if is_root else None)

            # Simulate bundles and compute moments
            sim_bundles = model.subproblems.solve(theta)
            sim_moments = compute_moments_distributed(
                sim_bundles, local_firm_idx, local_year_idx,
                comm, n_agents)

            elapsed = time.time() - t0

            if is_root:
                l2 = float(np.linalg.norm(sim_moments - obs_moments))
                row.append({"rho": rho, "alpha": alpha,
                            "theta": theta.copy(), "l2": l2, "time": elapsed})
                print(f"theta = {theta}")
                print(f"L2 = {l2:.6f}  ({elapsed:.1f}s)")

        if is_root:
            grid.append(row)
            print(f"\n--- alpha={alpha:.3f} ---")
            print(f"{'rho':>6} | {'L2':>12} | {'time':>6}")
            print("-" * 35)
            for r in row:
                print(f"{r['rho']:6.3f} | {r['l2']:12.6f} | {r['time']:5.0f}s")

    if is_root:
        flat = [r for row in grid for r in row]
        best = min(flat, key=lambda r: r["l2"])
        print(f"\n{'='*70}")
        print(f"BEST: rho={best['rho']:.4f}, alpha={best['alpha']:.4f}  "
              f"(L2={best['l2']:.6f})")
        for j, name in enumerate(NAMES):
            print(f"  {name:>8} = {best['theta'][j]:+.6f}")

        save_results(rho_values, alpha_values, grid, comm,
                     BASE_DIR / args.output_dir)


if __name__ == "__main__":
    main()
