#!/usr/bin/env python3
"""
Extract u_hat from bootstrap checkpoint .sol files on HPC.
No Gurobi needed — parses .sol text files directly.

Usage (on HPC):
  python3 extract_bootstrap_uhat.py /path/to/master_config_boot/checkpoints/bootstrap

Outputs: bootstrap_uhat.npz in the same directory as this script.
"""
import os, sys, json, argparse
import numpy as np
from pathlib import Path


def parse_sol(sol_path, prefix="utility["):
    """Parse variable values from a Gurobi .sol file."""
    vals = {}
    with open(sol_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 2:
                continue
            name, val = parts[0], float(parts[1])
            if name.startswith(prefix):
                idx = int(name.split("[")[1].rstrip("]"))
                vals[idx] = val
    return vals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir",
                        help="Path to .../checkpoints/bootstrap folder")
    parser.add_argument("--output", default=None,
                        help="Output path (default: bootstrap_uhat.npz next to this script)")
    args = parser.parse_args()

    boot_dir = Path(args.checkpoint_dir)
    if not boot_dir.exists():
        print(f"ERROR: {boot_dir} does not exist")
        sys.exit(1)

    # Find all boot_NNNN folders, sorted
    folders = sorted([d for d in boot_dir.iterdir()
                      if d.is_dir() and d.name.startswith("boot_")])
    print(f"Found {len(folders)} bootstrap checkpoint folders")

    if not folders:
        sys.exit(1)

    # Also extract theta from point estimate if available
    pt_dir = boot_dir.parent / "point_estimate"
    has_pt = pt_dir.exists() and (pt_dir / "master.sol").exists()

    all_u = []
    all_theta = []
    boot_ids = []
    converged_flags = []

    for folder in folders:
        sol_path = folder / "master.sol"
        meta_path = folder / "meta.json"

        if not sol_path.exists():
            print(f"  SKIP {folder.name}: no master.sol")
            continue

        # Parse u values
        u_vals = parse_sol(str(sol_path), "utility[")
        theta_vals = parse_sol(str(sol_path), "parameter[")

        if not u_vals:
            print(f"  SKIP {folder.name}: no utility variables in .sol")
            continue

        n_agents = max(u_vals.keys()) + 1
        u_hat = np.zeros(n_agents)
        for k, v in u_vals.items():
            u_hat[k] = v

        n_cov = max(theta_vals.keys()) + 1 if theta_vals else 0
        theta = np.zeros(n_cov)
        for k, v in theta_vals.items():
            theta[k] = v

        # Read convergence flag (ignore it per user instruction, but store it)
        converged = True
        if meta_path.exists():
            with open(meta_path) as f:
                converged = json.load(f).get("converged", True)

        boot_id = int(folder.name.split("_")[1])
        boot_ids.append(boot_id)
        all_u.append(u_hat)
        all_theta.append(theta)
        converged_flags.append(converged)

        print(f"  {folder.name}: n_agents={n_agents}, n_cov={n_cov}, "
              f"sum(u)={u_hat.sum():.2f}, converged={converged}")

    # Stack into arrays
    u_hat_array = np.stack(all_u)       # (n_boot, n_agents)
    theta_array = np.stack(all_theta)   # (n_boot, n_covariates)

    # Save
    out_path = args.output or str(Path(__file__).parent / "bootstrap_uhat.npz")
    np.savez_compressed(out_path,
                        u_hat=u_hat_array,
                        theta=theta_array,
                        boot_ids=np.array(boot_ids),
                        converged=np.array(converged_flags))

    print(f"\nSaved {len(all_u)} bootstrap samples to {out_path}")
    print(f"  u_hat shape: {u_hat_array.shape}")
    print(f"  theta shape: {theta_array.shape}")
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  File size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
