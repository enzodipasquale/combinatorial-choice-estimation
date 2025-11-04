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
    parser = argparse.ArgumentParser(description='Run complete inversion experiment pipeline')
    parser.add_argument('experiment', type=str, 
                       choices=['greedy_naive', 'greedy_iv', 'supermod_naive', 'supermod_iv',
                                'knapsack_naive', 'knapsack_iv', 'quadknapsack_naive', 'quadknapsack_iv'],
                       help='Experiment type to run')
    parser.add_argument('--mpi', type=int, default=10, help='Number of MPI processes (default: 10)')
    parser.add_argument('--output-dir', type=str, default=None, 
                       help='Output directory (default: experiments_paper_inversion/outputs/experiment_timestamp)')
    parser.add_argument('--skip-run', action='store_true', help='Skip experiment run, only generate tables')
    parser.add_argument('--config', type=str, default=None, help='Override config file path')
    parser.add_argument('--timeout', type=int, default=300,
                       help='Timeout in seconds (default: 300 = 5 minutes)')
    
    args = parser.parse_args()
    
    # Setup paths
    base_dir = Path(__file__).parent
    exp_dir = base_dir / args.experiment
    
    if not exp_dir.exists():
        print(f"Error: Experiment directory not found: {exp_dir}")
        return 1
    
    # Create or use output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = create_output_directory(base_dir, args.experiment)
    
    print(f"Output directory: {output_dir}")
    
    # Load config
    config_path = args.config or (exp_dir / 'config.yaml')
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1
    
    cfg = load_yaml_config(str(config_path))
    
    # Step 1: Run experiments (unless skipped)
    if not args.skip_run:
        print(f"\n{'='*70}")
        print(f"Running experiments: {args.experiment}")
        print(f"{'='*70}")
        
        sizes_path = exp_dir / 'sizes.yaml'
        if not sizes_path.exists():
            print(f"Error: Sizes config not found: {sizes_path}")
            return 1
        
        # Run all sizes with timeout
        cmd = ['python', str(base_dir / 'run_all_sizes.py'), args.experiment, 
               '--mpi', str(args.mpi), '--timeout', str(args.timeout)]
        try:
            result = subprocess.run(cmd, cwd=base_dir, check=True)
            print("✓ Experiments completed")
        except subprocess.CalledProcessError as e:
            print(f"✗ Experiments failed: {e}")
            return 1
    else:
        print("Skipping experiment run (using existing results)")
    
    # Step 2: Copy raw results
    results_path = exp_dir / cfg.get('results_csv', 'results.csv')
    if results_path.exists():
        import shutil
        shutil.copy(results_path, output_dir / 'results_raw.csv')
        print(f"✓ Copied raw results: {len(list(results_path.read_text().splitlines()))} lines")
    else:
        print("⚠ No raw results found")
    
    # Step 3: Generate summary
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
    
    # Step 4: Generate LaTeX tables (supports multiple sizes)
    print(f"\n{'='*70}")
    print("Generating LaTeX tables")
    print(f"{'='*70}")
    
    cmd = ['python', str(base_dir / 'generate_latex_tables.py'),
           str(base_dir), '--experiment', args.experiment,
           '--output', str(output_dir / 'tables.tex')]
    try:
        subprocess.run(cmd, cwd=base_dir, check=True)
        print("✓ LaTeX tables generated (flexible for multiple sizes)")
    except subprocess.CalledProcessError as e:
        print(f"✗ LaTeX generation failed: {e}")
        return 1
    
    # Step 5: Copy config files
    import shutil
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


if __name__ == '__main__':
    exit(main())


