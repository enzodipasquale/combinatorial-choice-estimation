#!/usr/bin/env python3
"""
Main pipeline script to run experiments and generate outputs.
Handles data generation, running multiple sizes, and output organization.
"""
import os
import sys
import yaml
import shutil
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
        f.write(f"# Experiment Output: {experiment_name}\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Contains:\n")
        f.write("- `results_raw.csv`: Raw results from all replications and sizes\n")
        f.write("- `results_summary.csv`: Aggregated statistics\n")
        f.write("- `tables.tex`: LaTeX tables for paper\n")
        f.write("- `config.yaml`: Experiment configuration used\n")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description='Run complete experiment pipeline')
    parser.add_argument('experiment', type=str, choices=['greedy', 'supermod', 'knapsack', 'quadknapsack', 'plain_single_item'],
                       help='Experiment type to run')
    parser.add_argument('--mpi', type=int, default=10, help='Number of MPI processes (default: 10)')
    parser.add_argument('--output-dir', type=str, default=None, 
                       help='Output directory (default: experiments_paper/__results/... timestamp)')
    parser.add_argument('--skip-run', action='store_true', help='Skip experiment run, only generate tables')
    parser.add_argument('--config', type=str, default=None, help='Override config file path')
    parser.add_argument('--sizes', type=str, default=None,
                       help='Override sizes config path (relative or absolute)')
    
    args = parser.parse_args()
    
    # Setup paths
    base_dir = BASE_DIR
    artifacts_dir = ARTIFACTS_DIR
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    exp_dir = EXPERIMENTS_DIR / args.experiment
    
    if not exp_dir.exists():
        print(f"Error: Experiment directory not found: {exp_dir}")
        return 1
    
    # Create or use output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = create_output_directory(args.experiment)
    
    print(f"Output directory: {output_dir}")
    
    # Load config (with optional override)
    default_config_path = exp_dir / 'config.yaml'
    override_applied = False
    backup_config_path = None
    
    try:
        if args.config:
            override_path = Path(args.config)
            if not override_path.exists():
                print(f"Error: Override config file not found: {override_path}")
                return 1
            
            if not default_config_path.exists():
                print(f"Error: Default config file missing: {default_config_path}")
                return 1
            
            if override_path.resolve() != default_config_path.resolve():
                backup_config_path = default_config_path.with_suffix('.yaml.backup')
                shutil.copy(default_config_path, backup_config_path)
                shutil.copy(override_path, default_config_path)
                override_applied = True
        else:
            if not default_config_path.exists():
                print(f"Error: Config file not found: {default_config_path}")
                return 1
        
        config_path = default_config_path
        cfg = load_yaml_config(str(config_path))
    
        # Step 1: Run experiments (unless skipped)
        if not args.skip_run:
            print(f"\n{'='*70}")
            print(f"Running experiments: {args.experiment}")
            print(f"{'='*70}")
            
            if not args.sizes:
                sizes_path = exp_dir / 'sizes.yaml'
                if not sizes_path.exists():
                    print(f"Error: Sizes config not found: {sizes_path}")
                    return 1
            
            # Run all sizes
            cmd = [sys.executable, str(BASE_DIR / 'run_all_sizes.py'), args.experiment, '--mpi', str(args.mpi)]
            if args.sizes:
                cmd.extend(['--sizes', args.sizes])
            try:
                result = subprocess.run(cmd, cwd=base_dir, check=True)
                print("✓ Experiments completed")
            except subprocess.CalledProcessError as e:
                print(f"✗ Experiments failed: {e}")
                return 1
        else:
            print("Skipping experiment run (using existing results)")
        
        # Step 2: Copy raw results (with CSV cleanup if needed)
        results_path = exp_dir / cfg.get('results_csv', 'results.csv')
        if results_path.exists():
            # Try to fix CSV if it has parsing issues
            fix_script = base_dir / 'fix_csv_parsing.py'
            if fix_script.exists():
                try:
                    # Run fix script silently (it will only fix if needed)
                    subprocess.run(['python', str(fix_script), str(results_path), '--quiet'], 
                                cwd=str(base_dir), 
                                stdout=subprocess.DEVNULL, 
                                stderr=subprocess.DEVNULL,
                                check=False)
                except:
                    pass  # Continue even if fix fails
            
            shutil.copy(results_path, output_dir / 'results_raw.csv')
            print(f"✓ Copied raw results: {len(list(results_path.read_text().splitlines()))} lines")
        else:
            print("⚠ No raw results found")
        
        # Step 3: Generate summary
        print(f"\n{'='*70}")
        print("Generating summary statistics")
        print(f"{'='*70}")
        
        # Use same Python interpreter as this script
        python_cmd = sys.executable
        cmd = [python_cmd, str(BASE_DIR / 'summarize_results.py'), 
           str(results_path), '--output', str(output_dir / 'results_summary.csv')]
        try:
            subprocess.run(cmd, cwd=base_dir, check=True)
            print("✓ Summary generated")
        except subprocess.CalledProcessError as e:
            print(f"✗ Summary generation failed: {e}")
            return 1
        
        # Step 4: Generate LaTeX tables
        print(f"\n{'='*70}")
        print("Generating LaTeX tables")
        print(f"{'='*70}")
        
        # Use same Python interpreter as this script
        python_cmd = sys.executable
        # Pass the output directory (which contains results_raw.csv) instead of BASE_DIR
        cmd = [python_cmd, str(BASE_DIR / 'generate_latex_tables.py'),
               str(output_dir), '--experiment', args.experiment,
               '--output', str(output_dir / 'tables.tex')]
        try:
            subprocess.run(cmd, cwd=base_dir, check=True)
            print("✓ LaTeX tables generated")
        except subprocess.CalledProcessError as e:
            print(f"✗ LaTeX generation failed: {e}")
            return 1
        
        # Step 5: Copy config files
        shutil.copy(config_path, output_dir / 'config.yaml')
        if (exp_dir / 'sizes.yaml').exists():
            shutil.copy(exp_dir / 'sizes.yaml', output_dir / 'sizes.yaml')
        
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
    finally:
        if override_applied and backup_config_path and backup_config_path.exists():
            shutil.move(str(backup_config_path), str(default_config_path))


if __name__ == '__main__':
    exit(main())

