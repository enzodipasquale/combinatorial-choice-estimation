#!/usr/bin/env python3
"""
Run experiments for multiple problem sizes with timeout protection.
"""
import os
import sys
import yaml
import subprocess
from pathlib import Path
import argparse

BASE_DIR = Path(__file__).parent
EXPERIMENTS_DIR = BASE_DIR / "experiments"


def load_yaml_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Run experiments for all sizes')
    parser.add_argument('experiment_dir', type=str, help='Experiment directory (e.g., experiments/naive/greedy)')
    parser.add_argument('--sizes', type=str, default='sizes.yaml', 
                       help='Sizes config file (default: sizes.yaml)')
    parser.add_argument('--mpi', type=int, default=10, help='Number of MPI processes (default: 10)')
    parser.add_argument('--timeout', type=int, default=None,
                       help='Timeout in seconds for debugging (optional, uses timeout wrapper if provided)')
    
    args = parser.parse_args()
    
    base_dir = BASE_DIR
    exp_arg = Path(args.experiment_dir)
    exp_dir = exp_arg if exp_arg.is_absolute() else base_dir / exp_arg
    if not exp_dir.exists():
        print(f"Error: Experiment directory not found: {exp_dir}")
        return 1
    
    # Set working directory to project root for MPI (same as experiments_paper)
    project_root = base_dir.parent
    
    sizes_path = exp_dir / args.sizes
    if not sizes_path.exists():
        print(f"Error: Sizes config not found: {sizes_path}")
        print("Creating default sizes.yaml...")
        create_default_sizes(sizes_path)
    
    sizes_cfg = load_yaml_config(str(sizes_path))
    config_path = exp_dir / 'config.yaml'
    
    # Load config (optional - defaults if missing)
    if config_path.exists():
        cfg = load_yaml_config(str(config_path))
    else:
        cfg = {'results_csv': 'results.csv'}
    
    # Determine which sizes to run
    if 'sizes' in sizes_cfg:
        sizes_list = []
        for size_name, size_def in sizes_cfg['sizes'].items():
            sizes_list.append((size_name, size_def['num_agents'], size_def['num_items']))
        print(f"Running {len(sizes_list)} named sizes...")
    elif 'sizes_grid' in sizes_cfg:
        agents_list = sizes_cfg['sizes_grid']['agents']
        items_list = sizes_cfg['sizes_grid']['items']
        sizes_list = []
        for num_agents in agents_list:
            for num_items in items_list:
                sizes_list.append((f'I{num_agents}_J{num_items}', num_agents, num_items))
        print(f"Running {len(sizes_list)} size combinations ({len(agents_list)}x{len(items_list)})...")
    else:
        print("Error: No sizes definition found in sizes.yaml")
        return 1
    
    results_path = exp_dir / cfg.get('results_csv', 'results.csv')
    
    # Clear existing results if requested
    if results_path.exists() and '--clear' in sys.argv:
        results_path.unlink()
        print(f"Cleared existing results file")
    
    # Run for each size
    timeout_wrapper = base_dir / 'run_with_timeout.py'
    for size_name, num_agents, num_items in sizes_list:
        print(f"\n{'='*60}")
        print(f"Running size: {size_name} (I={num_agents}, J={num_items})")
        print(f"{'='*60}")
        
        run_script = exp_dir / 'run.py'
        env = os.environ.copy()
        env['NUM_AGENTS'] = str(num_agents)
        env['NUM_ITEMS'] = str(num_items)
        
        # Adjust MPI processes: use min(num_agents, requested_mpi) to avoid deadlocks
        # with small agent counts
        effective_mpi = min(num_agents, args.mpi)
        if effective_mpi != args.mpi:
            print(f"Note: Using {effective_mpi} MPI processes (limited by {num_agents} agents)")
        
        # Check if running under SLURM and launched via srun
        in_slurm = 'SLURM_JOB_ID' in os.environ
        launched_via_srun = 'SLURM_PROCID' in os.environ
        
        # Use timeout wrapper only if timeout is specified (for debugging)
        if args.timeout is not None:
            cmd = [sys.executable, str(timeout_wrapper), 
                   '--timeout', str(args.timeout),
                   '--mpi', str(effective_mpi),
                   str(run_script)]
        else:
            # Direct execution (no timeout)
            if launched_via_srun:
                # Launched via srun - MPI processes are already set up by SLURM
                # Just run the Python script directly, mpi4py will use the existing MPI processes
                cmd = [sys.executable, str(run_script)]
            elif in_slurm:
                # Under SLURM but not via srun - use mpirun
                # SLURM sets OMPI_COMM_WORLD_SIZE automatically, so we don't need to specify -n
                cmd = ['mpirun', sys.executable, str(run_script)]
            else:
                # Local execution - specify number of processes
                cmd = ['mpirun', '-n', str(effective_mpi), sys.executable, str(run_script)]
        
        try:
            result = subprocess.run(cmd, cwd=str(project_root), env=env, check=True)
            print(f"✓ Completed {size_name}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed {size_name}: {e}")
            if args.timeout and e.returncode == 124:
                print(f"  (Timed out after {args.timeout}s)")
            continue
    
    print(f"\n{'='*60}")
    print(f"All sizes completed. Results saved to: {results_path}")
    print(f"{'='*60}")
    
    return 0


def create_default_sizes(path: Path):
    """Create a default sizes.yaml file."""
    default = {
        'sizes': {
            'tiny': {'num_agents': 10, 'num_items': 10},
            'small': {'num_agents': 20, 'num_items': 20}
        }
    }
    with open(path, 'w') as f:
        yaml.dump(default, f, default_flow_style=False)
    print(f"Created default {path}")


if __name__ == '__main__':
    exit(main())


