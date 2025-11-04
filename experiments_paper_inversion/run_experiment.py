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


def load_yaml_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def create_output_directory(base_dir: Path, experiment_name: str) -> Path:
    """Create timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / "outputs" / f"{experiment_name}_{timestamp}"
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
    parser.add_argument('--output-dir', type=str, default=None, 
                       help='Output directory (default: experiments_paper_inversion/outputs/experiment_timestamp)')
    parser.add_argument('--skip-run', action='store_true', help='Skip experiment run, only generate tables')
    parser.add_argument('--config', type=str, default=None, help='Override config file path')
    parser.add_argument('--timeout', type=int, default=None,
                       help='Timeout per size in seconds for debugging (optional, passed to run_all_sizes.py)')
    
    args = parser.parse_args()
    
    # Setup paths - use combined experiment directory
    base_dir = Path(__file__).parent
    exp_dir = base_dir / args.experiment
    
    # For backward compatibility, check if _naive or _iv directories exist
    naive_dir = base_dir / f"{args.experiment}_naive"
    iv_dir = base_dir / f"{args.experiment}_iv"
    
    if not exp_dir.exists() and not naive_dir.exists():
        print(f"Error: Experiment directory not found: {exp_dir}")
        print(f"  Also checked: {naive_dir}")
        return 1
    
    # If combined directory doesn't exist but separate ones do, use those
    if not exp_dir.exists():
        if naive_dir.exists() and iv_dir.exists():
            print(f"Note: Using separate naive/IV directories. Running both methods...")
            # We'll handle both directories
            exp_dirs = [naive_dir, iv_dir]
        else:
            print(f"Error: Neither combined nor separate directories found")
            return 1
    else:
        exp_dirs = [exp_dir]
    
    # Create or use output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = create_output_directory(base_dir, args.experiment)
    
    print(f"Output directory: {output_dir}")
    
    # Load config - use first available directory (optional if skipping run)
    cfg = {}
    if len(exp_dirs) == 2:
        config_path = args.config or (exp_dirs[0] / 'config.yaml')  # Use naive directory config
    else:
        config_path = args.config or (exp_dir / 'config.yaml')
    
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
        if len(exp_dirs) == 2:
            for exp_dir_to_run in exp_dirs:
                method_name = "naive" if "_naive" in str(exp_dir_to_run) else "iv"
                print(f"\n--- Running {method_name.upper()} method ---")
                
                sizes_path = exp_dir_to_run / 'sizes.yaml'
                if not sizes_path.exists():
                    print(f"Warning: Sizes config not found: {sizes_path}, skipping")
                    continue
                
                exp_name = exp_dir_to_run.name
                cmd = ['python', str(base_dir / 'run_all_sizes.py'), exp_name, 
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
            sizes_path = exp_dir / 'sizes.yaml'
            if not sizes_path.exists():
                print(f"Error: Sizes config not found: {sizes_path}")
                return 1
            
            cmd = ['python', str(base_dir / 'run_all_sizes.py'), args.experiment, 
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
            method_name = "naive" if "_naive" in str(exp_dir_source) else "iv"
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
        results_path = exp_dir / cfg.get('results_csv', 'results.csv')
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
        
        cmd = ['python', str(base_dir / 'summarize_results.py'), 
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
        
        # Determine experiment type for table generation
        # If we have combined results, use the base experiment name
        exp_type_for_table = args.experiment
        # Check if we should use _naive or _iv suffix based on available methods
        if len(exp_dirs) == 2:
            # Both methods exist - use base name, table generator will handle both
            exp_type_for_table = f"{args.experiment}_combined"
        elif naive_dir.exists():
            exp_type_for_table = f"{args.experiment}_naive"
        elif iv_dir.exists():
            exp_type_for_table = f"{args.experiment}_iv"
        
        # Generate table from combined results using the base experiment name
        # The table generator will detect both methods in the CSV
        cmd = ['python', str(base_dir / 'generate_latex_tables.py'),
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
    shutil.copy(config_path, output_dir / 'config.yaml')
    # Copy sizes.yaml from first available directory
    if len(exp_dirs) == 2:
        sizes_source = exp_dirs[0] / 'sizes.yaml'
    else:
        sizes_source = exp_dir / 'sizes.yaml'
    if sizes_source.exists():
        shutil.copy(sizes_source, output_dir / 'sizes.yaml')
    
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


