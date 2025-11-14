#!/usr/bin/env python3
"""
Test script to run all small experiments without MPI and verify table generation.
"""
import os
import sys
import subprocess
import pandas as pd
import shutil
import yaml
from pathlib import Path

BASE_DIR = Path(__file__).parent
EXPERIMENTS_DIR = BASE_DIR / "experiments"
EXPERIMENTS = ['greedy', 'knapsack', 'plain_single_item', 'quadknapsack', 'supermod']


def check_results_csv(csv_path, experiment_name):
    """Check that results CSV has all required fields for table generation."""
    if not csv_path.exists():
        print(f"  ✗ Results CSV not found: {csv_path}")
        return False
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  ✗ Failed to read CSV: {e}")
        return False
    
    if len(df) == 0:
        print(f"  ✗ CSV is empty")
        return False
    
    # Filter out ERROR rows
    df = df[df['method'] != 'ERROR'].copy()
    if len(df) == 0:
        print(f"  ✗ No valid results (only ERROR rows)")
        return False
    
    # Required base columns
    required_base = [
        'replication', 'seed', 'method', 'time_s', 'obj_value',
        'num_agents', 'num_items', 'num_features', 'num_simuls', 'sigma', 'subproblem'
    ]
    
    # Required timing columns
    required_timing = [
        'timing_compute', 'timing_solve', 'timing_comm',
        'timing_compute_pct', 'timing_solve_pct', 'timing_comm_pct'
    ]
    
    # Required objective consistency columns
    required_obj = [
        'obj_diff_rg_1slack', 'obj_diff_rg_ellipsoid', 
        'obj_diff_1slack_ellipsoid', 'obj_close_all'
    ]
    
    # Check num_features exists and is valid
    if 'num_features' not in df.columns:
        print(f"  ✗ Missing 'num_features' column")
        return False
    
    num_features = int(df['num_features'].iloc[0])
    if pd.isna(num_features):
        print(f"  ✗ num_features is NaN")
        return False
    
    # Required theta columns
    required_theta_true = [f'theta_true_{k}' for k in range(num_features)]
    required_theta_est = [f'theta_{k}' for k in range(num_features)]
    
    all_required = required_base + required_timing + required_obj + required_theta_true + required_theta_est
    
    missing = [col for col in all_required if col not in df.columns]
    if missing:
        print(f"  ✗ Missing columns: {missing}")
        return False
    
    # Check that we have data for both methods
    methods = df['method'].unique()
    if 'row_generation' not in methods:
        print(f"  ✗ Missing 'row_generation' method")
        return False
    if 'row_generation_1slack' not in methods:
        print(f"  ✗ Missing 'row_generation_1slack' method")
        return False
    
    # Check that theta columns have valid data
    for k in range(num_features):
        theta_col = f'theta_{k}'
        theta_true_col = f'theta_true_{k}'
        if df[theta_col].isna().all():
            print(f"  ✗ All values are NaN in {theta_col}")
            return False
        if df[theta_true_col].isna().all():
            print(f"  ✗ All values are NaN in {theta_true_col}")
            return False
    
    print(f"  ✓ CSV has all required fields ({len(df)} rows, {len(df.columns)} columns)")
    print(f"    Methods: {sorted(methods)}")
    print(f"    Features: {num_features}")
    return True


def run_experiment(experiment_name):
    """Run a single experiment with small sizes."""
    print(f"\n{'='*70}")
    print(f"Testing: {experiment_name}")
    print(f"{'='*70}")
    
    exp_dir = EXPERIMENTS_DIR / experiment_name
    if not exp_dir.exists():
        print(f"✗ Experiment directory not found: {exp_dir}")
        return False
    
    # Clear existing results
    results_path = exp_dir / 'results.csv'
    if results_path.exists():
        results_path.unlink()
        print(f"  Cleared existing results.csv")
    
    # Set environment variables for small test
    sizes_cfg = BASE_DIR / 'test_small_sizes.yaml'
    with open(sizes_cfg, 'r') as f:
        sizes = yaml.safe_load(f)
    
    size_def = sizes['sizes']['small']
    num_agents = size_def['num_agents']
    num_items = size_def['num_items']
    
    env = os.environ.copy()
    env['NUM_AGENTS'] = str(num_agents)
    env['NUM_ITEMS'] = str(num_items)
    
    # Temporarily modify config to set num_replications to 1 and adjust timeout
    config_path = exp_dir / 'config.yaml'
    backup_path = config_path.with_suffix('.yaml.backup')
    
    if backup_path.exists():
        shutil.copy(backup_path, config_path)
    else:
        shutil.copy(config_path, backup_path)
    
    cfg = yaml.safe_load(open(config_path))
    original_replications = cfg.get('num_replications', 1)
    original_timeout = cfg.get('timeout_seconds', 600)
    
    cfg['num_replications'] = 1  # Set to 1 for small tests
    
    # Increase timeout for quadknapsack (it's computationally expensive)
    if experiment_name == 'quadknapsack':
        cfg['timeout_seconds'] = 1200  # 20 minutes for small test
    
    # Only write if we actually changed something
    if cfg.get('num_replications', 1) != original_replications or cfg.get('timeout_seconds', 600) != original_timeout:
        with open(config_path, 'w') as f:
            yaml.dump(cfg, f)
    
    # Run with mpirun -n 1 (single process, effectively no MPI)
    run_script = exp_dir / 'run.py'
    project_root = BASE_DIR.parent
    
    cmd = ['mpirun', '-n', '1', 'python', str(run_script)]
    
    print(f"  Running: {' '.join(cmd)}")
    print(f"  Environment: NUM_AGENTS={num_agents}, NUM_ITEMS={num_items}")
    
    try:
        result = subprocess.run(
            cmd, 
            cwd=str(project_root), 
            env=env, 
            check=True,
            capture_output=True,
            text=True
        )
        print(f"  ✓ Experiment completed")
        if result.stdout:
            # Print last few lines of output
            lines = result.stdout.strip().split('\n')
            for line in lines[-5:]:
                print(f"    {line}")
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Experiment failed: {e}")
        if e.stdout:
            print(f"  stdout: {e.stdout[-500:]}")
        if e.stderr:
            print(f"  stderr: {e.stderr[-500:]}")
        return False
    
    # Restore config if we modified it
    config_path = exp_dir / 'config.yaml'
    backup_path = config_path.with_suffix('.yaml.backup')
    if backup_path.exists():
        shutil.copy(backup_path, config_path)
        backup_path.unlink()
    
    # Check results
    if not check_results_csv(results_path, experiment_name):
        return False
    
    return True


def test_table_generation(experiment_name):
    """Test that table generation works for this experiment."""
    print(f"\n  Testing table generation...")
    
    exp_dir = EXPERIMENTS_DIR / experiment_name
    results_path = exp_dir / 'results.csv'
    
    if not results_path.exists():
        print(f"  ✗ Results CSV not found for table generation")
        return False
    
    # Create a temporary output directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # Copy results to output directory
        import shutil
        shutil.copy(results_path, output_dir / 'results_raw.csv')
        
        # Run table generation
        cmd = [
            sys.executable,
            str(BASE_DIR / 'generate_latex_tables.py'),
            str(output_dir),
            '--experiment', experiment_name,
            '--output', str(output_dir / 'tables.tex')
        ]
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(BASE_DIR),
                check=True,
                capture_output=True,
                text=True
            )
            
            tables_path = output_dir / 'tables.tex'
            if tables_path.exists():
                content = tables_path.read_text()
                if len(content) > 100:  # Should have substantial content
                    print(f"  ✓ Table generation successful ({len(content)} chars)")
                    return True
                else:
                    print(f"  ✗ Table file too short")
                    return False
            else:
                print(f"  ✗ Table file not created")
                return False
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Table generation failed: {e}")
            if e.stderr:
                print(f"    stderr: {e.stderr[-300:]}")
            return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test small experiments without MPI')
    parser.add_argument('experiment', type=str, nargs='?', choices=EXPERIMENTS + ['all'],
                       default='all', help='Experiment to test (default: all)')
    args = parser.parse_args()
    
    experiments_to_run = EXPERIMENTS if args.experiment == 'all' else [args.experiment]
    
    print("Testing small experiments without MPI")
    print("=" * 70)
    
    results = {}
    
    for exp in experiments_to_run:
        success = run_experiment(exp)
        if success:
            table_success = test_table_generation(exp)
            results[exp] = {'run': success, 'table': table_success}
        else:
            results[exp] = {'run': False, 'table': False}
    
    # Summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    
    all_passed = True
    for exp, res in results.items():
        status = "✓" if (res['run'] and res['table']) else "✗"
        print(f"{status} {exp:20s} - Run: {res['run']}, Table: {res['table']}")
        if not (res['run'] and res['table']):
            all_passed = False
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())

