#!/usr/bin/env python3
"""
Main pipeline script to run inversion experiments and generate outputs.
Handles data generation, running multiple sizes, and output organization.
"""
import os
import sys
import yaml
import subprocess
from pathlib import Path
from datetime import datetime
import argparse

BASE_DIR = Path(__file__).parent
EXPERIMENTS_DIR = BASE_DIR / "experiments"
ARTIFACTS_DIR = BASE_DIR / "__results"
LOGS_DIR = BASE_DIR / "__logs_slurm"


def load_yaml_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def create_output_directory(experiment_name: str) -> Path:
    """Create timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ARTIFACTS_DIR / f"{experiment_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create README with experiment info
    readme_path = output_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(f"# Inversion Experiment Output: {experiment_name}\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Contains:\n")
        f.write("- `results_raw.csv`: Raw results from all replications and sizes\n")
        f.write("- `results_summary.csv`: Aggregated statistics\n")
        f.write("- `tables.tex`: LaTeX tables for paper (supports multiple sizes)\n")
        f.write("- `config.yaml`: Experiment configuration used\n")
        f.write("- `sizes.yaml`: Size definitions used\n")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description='Run complete inversion experiment pipeline (both naive and IV)')
    parser.add_argument('experiment', type=str, 
                       choices=['greedy', 'supermod', 'knapsack', 'quadknapsack'],
                       help='Experiment type to run (runs both naive and IV methods)')
    parser.add_argument('--mpi', type=int, default=10, help='Number of MPI processes (default: 10)')
    parser.add_argument('--output-dir', type=str, default=None, \
                       help='Output directory (default: experiments_paper_inversion/__results/<experiment>_<timestamp>)')
    parser.add_argument('--skip-run', action='store_true', help='Skip experiment run, only generate tables')
    parser.add_argument('--config', type=str, default=None, help='Override config file path')
    parser.add_argument('--timeout', type=int, default=None,
                       help='Timeout per size in seconds for debugging (optional, passed to run_all_sizes.py)')
    parser.add_argument('--sizes', type=str, default='sizes.yaml',
                       help='Sizes config file name (default: sizes.yaml, e.g., sizes_large.yaml)')
    
    args = parser.parse_args()
    
    # Setup paths - use combined experiment directory
    base_dir = BASE_DIR
    artifacts_dir = ARTIFACTS_DIR
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    exp_dir_combined = EXPERIMENTS_DIR / args.experiment
    
    # Collect possible naive/iv directories (legacy and grouped structure)
    naive_candidates = [
        EXPERIMENTS_DIR / "naive" / args.experiment,
        base_dir / f"{args.experiment}_naive",
        base_dir / "02_estimations" / "naive" / args.experiment  # legacy path fallback
    ]
    iv_candidates = [
        EXPERIMENTS_DIR / "iv" / args.experiment,
        base_dir / f"{args.experiment}_iv",
        base_dir / "02_estimations" / "iv" / args.experiment  # legacy path fallback
    ]
    
    existing_naive = next((path for path in naive_candidates if path.exists()), None)
    existing_iv = next((path for path in iv_candidates if path.exists()), None)
    
    if exp_dir_combined.exists():
        exp_dirs = [exp_dir_combined]
    elif existing_naive and existing_iv:
        print("Note: Using grouped naive/IV directories.")
        exp_dirs = [existing_naive, existing_iv]
    elif existing_naive or existing_iv:
        exp_dirs = [existing_naive or existing_iv]
    else:
        print(f"Error: Could not locate experiment directories for '{args.experiment}'.")
        print(f"  Checked combined directory: {exp_dir_combined}")
        for candidate in naive_candidates + iv_candidates:
            print(f"  Checked: {candidate}")
        return 1

    exp_dir = exp_dirs[0]
    
    # Create or use output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = create_output_directory(args.experiment)
    
    print(f"Output directory: {output_dir}")
    
    primary_exp_dir = exp_dirs[0]
    
    # Load config - use first available directory (optional if skipping run)
    cfg = {}
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = base_dir / config_path
    else:
        config_path = primary_exp_dir / 'config.yaml'
    
    if config_path.exists():
        cfg = load_yaml_config(str(config_path))
    elif not args.skip_run:
        print(f"Error: Config file not found: {config_path}")
        return 1
    else:
        # Skip run - config not needed
        cfg = {'results_csv': 'results.csv'}
    
    # Step 1: Run experiments (unless skipped)
    if not args.skip_run:
        print(f"\n{'='*70}")
        print(f"Running experiments: {args.experiment} (both naive and IV)")
        print(f"{'='*70}")
        
        # Run both naive and IV if separate directories exist
        python_cmd = sys.executable
        if len(exp_dirs) == 2:
            for exp_dir_to_run in exp_dirs:
                path_parts = {part.lower() for part in exp_dir_to_run.parts}
                if any('naive' == part or part.endswith('_naive') for part in path_parts) or '_naive' in str(exp_dir_to_run):
                    method_name = "naive"
                elif any('iv' == part or part.endswith('_iv') for part in path_parts) or '_iv' in str(exp_dir_to_run):
                    method_name = "iv"
                else:
                    method_name = exp_dir_to_run.name
                print(f"\n--- Running {method_name.upper()} method ---")
                
                sizes_path = exp_dir_to_run / args.sizes
                if not sizes_path.exists():
                    print(f"Warning: Sizes config not found: {sizes_path}, skipping")
                    continue
                
                exp_rel_path = exp_dir_to_run.relative_to(base_dir)
                cmd = [python_cmd, str(base_dir / 'run_all_sizes.py'), str(exp_rel_path), 
                       '--sizes', args.sizes,
                       '--mpi', str(args.mpi)]
                if args.timeout is not None:
                    cmd.extend(['--timeout', str(args.timeout)])
                try:
                    result = subprocess.run(cmd, cwd=base_dir, check=True)
                    print(f"✓ {method_name.upper()} experiments completed")
                except subprocess.CalledProcessError as e:
                    print(f"✗ {method_name.upper()} experiments failed: {e}")
                    if e.returncode == 124:
                        print(f"  (Timed out after {args.timeout}s)")
        else:
            # Single combined directory
            exp_dir_single = primary_exp_dir
            sizes_path = exp_dir_single / args.sizes
            if not sizes_path.exists():
                print(f"Error: Sizes config not found: {sizes_path}")
                return 1
            
            single_rel_path = exp_dir_single.relative_to(base_dir)
            cmd = [python_cmd, str(base_dir / 'run_all_sizes.py'), str(single_rel_path), 
                   '--sizes', args.sizes,
                   '--mpi', str(args.mpi)]
            if args.timeout is not None:
                cmd.extend(['--timeout', str(args.timeout)])
            try:
                result = subprocess.run(cmd, cwd=base_dir, check=True)
                print("✓ Experiments completed")
            except subprocess.CalledProcessError as e:
                print(f"✗ Experiments failed: {e}")
                return 1
    else:
        print("Skipping experiment run (using existing results)")
    
    # Step 2: Copy and combine raw results from both methods
    import shutil
    import pandas as pd
    
    all_results = []
    if len(exp_dirs) == 2:
        # Combine results from both naive and IV
        for exp_dir_source in exp_dirs:
            path_parts = {part.lower() for part in exp_dir_source.parts}
            if any('naive' == part or part.endswith('_naive') for part in path_parts) or '_naive' in str(exp_dir_source):
                method_name = "naive"
            elif any('iv' == part or part.endswith('_iv') for part in path_parts) or '_iv' in str(exp_dir_source):
                method_name = "iv"
            else:
                method_name = exp_dir_source.name
            config_source_path = exp_dir_source / 'config.yaml'
            if config_source_path.exists():
                cfg_source = load_yaml_config(str(config_source_path))
                results_csv_name = cfg_source.get('results_csv', 'results.csv')
            else:
                results_csv_name = 'results.csv'
            results_path_source = exp_dir_source / results_csv_name
            if results_path_source.exists():
                df = pd.read_csv(results_path_source)
                all_results.append(df)
                print(f"✓ Loaded {method_name} results: {len(df)} rows")
        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
            combined_results_path = output_dir / 'results_raw.csv'
            combined_df.to_csv(combined_results_path, index=False)
            print(f"✓ Combined results: {len(combined_df)} total rows")
            results_path = combined_results_path  # Use combined for summary/table generation
        else:
            print("⚠ No raw results found")
            results_path = None
    else:
        # Single directory
        results_path = primary_exp_dir / cfg.get('results_csv', 'results.csv')
        if results_path.exists():
            shutil.copy(results_path, output_dir / 'results_raw.csv')
            print(f"✓ Copied raw results: {len(list(results_path.read_text().splitlines()))} lines")
        else:
            print("⚠ No raw results found")
            results_path = None
    
    # Step 3: Generate summary
    if results_path and results_path.exists():
        print(f"\n{'='*70}")
        print("Generating summary statistics")
        print(f"{'='*70}")
        
        cmd = [python_cmd, str(base_dir / 'summarize_results.py'), 
               str(results_path), '--output', str(output_dir / 'results_summary.csv')]
        try:
            subprocess.run(cmd, cwd=base_dir, check=True)
            print("✓ Summary generated")
        except subprocess.CalledProcessError as e:
            print(f"✗ Summary generation failed: {e}")
            return 1
    
    # Step 4: Generate LaTeX tables (supports multiple sizes and both methods)
    if results_path and results_path.exists():
        print(f"\n{'='*70}")
        print("Generating LaTeX tables")
        print(f"{'='*70}")
        
        # Generate table from combined results using the base experiment name
        # The table generator will detect both methods in the CSV
        cmd = [python_cmd, str(base_dir / 'generate_latex_tables.py'),
               str(results_path), '--experiment', args.experiment,
               '--output', str(output_dir / 'tables.tex')]
        try:
            subprocess.run(cmd, cwd=base_dir, check=True)
            print("✓ LaTeX tables generated (both naive and IV methods)")
        except subprocess.CalledProcessError as e:
            print(f"✗ LaTeX generation failed: {e}")
            # Try alternative: use the results file directly
            print("Trying alternative table generation...")
            return 1
    
    # Step 5: Copy config files
    import shutil
    if config_path.exists():
        shutil.copy(config_path, output_dir / 'config.yaml')
    # Copy sizes.yaml from first available directory
    if len(exp_dirs) == 2:
        sizes_source = exp_dirs[0] / args.sizes
    else:
        sizes_source = primary_exp_dir / args.sizes
    if sizes_source.exists():
        shutil.copy(sizes_source, output_dir / args.sizes)
    
    # Summary
    print(f"\n{'='*70}")
    print("Pipeline completed successfully!")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    print(f"\nGenerated files:")
    for f in sorted(output_dir.glob('*')):
        if f.is_file():
            print(f"  - {f.name}")
    
    return 0


if __name__ == '__main__':
    exit(main())


