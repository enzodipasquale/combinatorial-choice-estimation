#!/usr/bin/env python3
import subprocess
import yaml
import argparse
from pathlib import Path
from itertools import product


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--spec", type=str, default=None)
    parser.add_argument("--replications", type=int, default=None)
    parser.add_argument("--mpi-procs", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--env", type=str, default="local", choices=["local", "hpc"])
    args = parser.parse_args()

    config_path = Path(__file__).parent / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    debug = args.debug or config.get("experiment", {}).get("debug_mode", False)
    grid_M = config["grid"].get("debug_M" if debug else "M", [10])
    grid_N = config["grid"].get("debug_N" if debug else "N", [10])
    specs = [args.spec] if args.spec else list(config["specifications"].keys())
    n_reps = args.replications or config.get("experiment", {}).get("n_replications", 50)
    n_bootstrap = min(config.get("experiment", {}).get("n_bootstrap", 200), args.mpi_procs)
    stats_dir = f"results/{args.env}"

    def get_timeout(M):
        if args.timeout:
            return args.timeout
        if M <= 50:
            return 600
        if M <= 100:
            return 1800
        return 3600

    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent

    for spec in specs:
        spec_cfg = config["specifications"].get(spec, {})
        max_M = spec_cfg.get("max_M", 200)
        alpha_cfg, lambda_cfg = spec_cfg.get("alpha"), spec_cfg.get("lambda")

        for M, N in product(grid_M, grid_N):
            if M > max_M:
                print(f"Skipping {spec} M={M} (max_M={max_M})")
                continue

            resolve = lambda c, M: c.get(M, c.get(str(M))) if isinstance(c, dict) else c
            alpha_val = resolve(alpha_cfg, M) if alpha_cfg is not None else None
            lambda_val = resolve(lambda_cfg, M) if lambda_cfg is not None else None

            print(f"\n{'='*60}\nRunning {spec}, N={N}, M={M}, α={alpha_val}, λ={lambda_val}\n{'='*60}")

            for rep in range(n_reps):
                print(f"  Replication {rep+1}/{n_reps}...", end=" ", flush=True)
                cmd = [
                    str(project_root / "run_with_timeout.sh"), str(get_timeout(M)),
                    "mpirun", "-n", str(args.mpi_procs),
                    "python", str(script_dir / "run_experiment.py"),
                    "--spec", spec, "--N", str(N), "--M", str(M),
                    "--replication", str(rep), "--config", config_path.name,
                    "--n-bootstrap", str(n_bootstrap),
                ]
                if alpha_val is not None:
                    cmd.extend(["--alpha", str(alpha_val)])
                if lambda_val is not None:
                    cmd.extend(["--lambda", str(lambda_val)])
                result = subprocess.run(cmd, cwd=str(project_root))
                print("✓" if result.returncode == 0 else f"✗ (exit {result.returncode})")

            print("  Computing statistics...", end=" ", flush=True)
            result = subprocess.run([
                "python", str(script_dir / "compute_statistics.py"),
                "--spec", spec, "--N", str(N), "--M", str(M),
                "--output-dir", stats_dir,
            ], cwd=str(project_root))
            print("✓" if result.returncode == 0 else "✗")

    print(f"\n{'='*60}\nAggregating results...\n{'='*60}")
    subprocess.run(["python", str(script_dir / "aggregate_results.py"),
                     "--config", str(config_path)], cwd=str(project_root))
    print("Done!")


if __name__ == "__main__":
    main()
