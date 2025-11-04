#!/usr/bin/env python3
"""
Run experiments for multiple problem sizes.
Iterates over sizes defined in sizes.yaml and runs the experiment for each.
"""
import os
import sys
import yaml
import subprocess
from pathlib import Path
import argparse


def load_yaml_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Run experiments for all sizes')
    parser.add_argument('experiment_dir', type=str, help='Experiment directory (e.g., greedy)')
    parser.add_argument('--sizes', type=str, default='sizes.yaml', 
                       help='Sizes config file (default: sizes.yaml)')
    parser.add_argument('--mpi', type=int, default=10, help='Number of MPI processes (default: 10)')
    parser.add_argument('--timeout', type=int, default=None,
                       help='Timeout in seconds for debugging (optional, uses timeout wrapper if provided)')
    
    args = parser.parse_args()
    
    exp_dir = Path(__file__).parent / args.experiment_dir
    if not exp_dir.exists():
        print(f"Error: Experiment directory not found: {exp_dir}")
        return 1
    
    sizes_path = exp_dir / args.sizes
    if not sizes_path.exists():
        print(f"Error: Sizes config not found: {sizes_path}")
        print("Creating default sizes.yaml...")
        create_default_sizes(sizes_path)
    
    sizes_cfg = load_yaml_config(str(sizes_path))
    config_path = exp_dir / 'config.yaml'
    cfg = load_yaml_config(str(config_path))
    
    # Determine which sizes to run
    if 'sizes' in sizes_cfg:
        # Named sizes (small, medium, large)
        sizes_list = []
        for size_name, size_def in sizes_cfg['sizes'].items():
            sizes_list.append((size_name, size_def['num_agents'], size_def['num_items']))
        print(f"Running {len(sizes_list)} named sizes...")
    elif 'sizes_grid' in sizes_cfg:
        # Grid of sizes
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
    base_dir = Path(__file__).parent
    project_root = base_dir.parent
    timeout_wrapper = base_dir / 'run_with_timeout.py'
    
    for size_name, num_agents, num_items in sizes_list:
        print(f"\n{'='*60}")
        print(f"Running size: {size_name} (I={num_agents}, J={num_items})")
        print(f"{'='*60}")
        
        # Run experiment with environment variables
        run_script = exp_dir / 'run.py'
        env = os.environ.copy()
        env['NUM_AGENTS'] = str(num_agents)
        env['NUM_ITEMS'] = str(num_items)
        
        # Use timeout wrapper only if timeout is specified (for debugging)
        if args.timeout is not None:
            cmd = ['python', str(timeout_wrapper), 
                   '--timeout', str(args.timeout),
                   '--mpi', str(args.mpi),
                   str(run_script)]
        else:
            # Direct MPI command (no timeout)
            cmd = ['mpirun', '-n', str(args.mpi), 'python', str(run_script)]
        
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
            'small': {'num_agents': 20, 'num_items': 20},
            'medium': {'num_agents': 50, 'num_items': 50},
            'large': {'num_agents': 100, 'num_items': 100}
        }
    }
    with open(path, 'w') as f:
        yaml.dump(default, f, default_flow_style=False)
    print(f"Created default {path}")


if __name__ == '__main__':
    exit(main())

