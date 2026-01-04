#!/usr/bin/env python3
"""
Run all BLP inversion experiments with different seeds and collect results.
"""
import subprocess
import sys
import re
from collections import defaultdict

def extract_results(output):
    """Extract results from experiment output."""
    results = {}
    
    # Try to extract the results table
    if "RESULTS" in output:
        # Extract coefficient breakdown
        lines = output.split('\n')
        in_table = False
        for line in lines:
            if "Coefficient breakdown:" in line:
                in_table = True
                continue
            if in_table and "Summary arrays:" in line:
                break
            if in_table and line.strip() and not line.strip().startswith('-') and 'Parameter' not in line:
                # Parse table row
                parts = line.split()
                if len(parts) >= 4:
                    param_name = parts[0]
                    try:
                        true_val = float(parts[1])
                        naive_val = float(parts[2])
                        iv_val = float(parts[3])
                        results[param_name] = {
                            'true': true_val,
                            'naive': naive_val,
                            'iv': iv_val,
                            'naive_error': abs(naive_val - true_val),
                            'iv_error': abs(iv_val - true_val),
                        }
                    except (ValueError, IndexError):
                        continue
    
    # Also extract summary arrays as fallback
    if not results and "True theta:" in output:
        # Try to extract arrays
        true_match = re.search(r'True theta:\s+\[([^\]]+)\]', output)
        naive_match = re.search(r'Naive.*?:\s+\[([^\]]+)\]', output)
        iv_match = re.search(r'IV.*?:\s+\[([^\]]+)\]', output)
        
        if true_match and naive_match and iv_match:
            try:
                true_vals = [float(x.strip()) for x in true_match.group(1).split(',')]
                naive_vals = [float(x.strip()) for x in naive_match.group(1).split(',')]
                iv_vals = [float(x.strip()) for x in iv_match.group(1).split(',')]
                
                for i, (t, n, iv) in enumerate(zip(true_vals, naive_vals, iv_vals)):
                    results[f'Feature {i}'] = {
                        'true': t,
                        'naive': n,
                        'iv': iv,
                        'naive_error': abs(n - t),
                        'iv_error': abs(iv - t),
                    }
            except (ValueError, AttributeError):
                pass
    
    return results

def run_experiment(script_path, seed, experiment_name):
    """Run a single experiment with a given seed."""
    print(f"\n{'='*80}")
    print(f"Running {experiment_name} with seed {seed}")
    print(f"{'='*80}\n")
    
    # Modify the script temporarily to use the seed
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Find and replace seed parameter
    if 'seed=None' in content:
        modified_content = content.replace('seed=None', f'seed={seed}')
    elif 'seed=42' in content:
        modified_content = content.replace('seed=42', f'seed={seed}')
    else:
        # Try to find the prepare call
        modified_content = re.sub(
            r'prepared = scenario\.prepare\(comm=comm, seed=[^,)]+',
            f'prepared = scenario.prepare(comm=comm, seed={seed}',
            content
        )
    
    # Write temporary script
    temp_script = script_path.replace('.py', f'_temp_seed{seed}.py')
    with open(temp_script, 'w') as f:
        f.write(modified_content)
    
    try:
        # Run the experiment
        result = subprocess.run(
            ['mpirun', '-n', '10', 'python', temp_script],
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout
        )
        
        output = result.stdout + result.stderr
        
        # Extract results
        results = extract_results(output)
        
        # Print output for debugging
        if "RESULTS" in output:
            # Print just the results section
            results_start = output.find("RESULTS")
            results_end = output.find("="*100, results_start + 100)
            if results_end > results_start:
                print(output[results_start:results_end+100])
            else:
                print(output[results_start:results_start+2000])
        else:
            print("No RESULTS section found in output")
            print(output[-1000:])  # Print last 1000 chars
        
        return results, output
        
    except subprocess.TimeoutExpired:
        print(f"Experiment timed out after 30 minutes")
        return None, None
    except Exception as e:
        print(f"Error running experiment: {e}")
        return None, None
    finally:
        # Clean up temp script
        import os
        if os.path.exists(temp_script):
            os.remove(temp_script)

def main():
    """Run all experiments with different seeds."""
    experiments = [
        ('experiment_inversion_knapsack_factory.py', 'Knapsack'),
        ('experiment_inversion_greedy_factory.py', 'Greedy'),
        ('experiment_inversion_supermod_factory.py', 'Supermodular'),
        ('experiment_inversion_quadknapsack_factory.py', 'Quadratic Knapsack'),
    ]
    
    seeds = [0, 1, 2, 3, 4]
    
    all_results = defaultdict(lambda: defaultdict(dict))
    
    for script_name, exp_name in experiments:
        script_path = f'experiments/BLP/{script_name}'
        for seed in seeds:
            results, output = run_experiment(script_path, seed, f"{exp_name} (seed={seed})")
            if results:
                all_results[exp_name][seed] = results
    
    # Print summary
    print("\n\n" + "="*100)
    print("SUMMARY OF ALL EXPERIMENTS")
    print("="*100)
    
    for exp_name, seed_results in all_results.items():
        print(f"\n{exp_name}:")
        print("-" * 80)
        
        # Collect all parameter names
        all_params = set()
        for seed, results in seed_results.items():
            all_params.update(results.keys())
        
        for param in sorted(all_params):
            print(f"\n  {param}:")
            naive_errors = []
            iv_errors = []
            
            for seed in sorted(seed_results.keys()):
                if param in seed_results[seed]:
                    r = seed_results[seed][param]
                    naive_errors.append(r['naive_error'])
                    iv_errors.append(r['iv_error'])
                    print(f"    Seed {seed}: True={r['true']:.4f}, Naive={r['naive']:.4f} (err={r['naive_error']:.4f}), "
                          f"IV={r['iv']:.4f} (err={r['iv_error']:.4f})")
            
            if naive_errors:
                print(f"    Mean Naive Error: {sum(naive_errors)/len(naive_errors):.4f}")
                print(f"    Mean IV Error: {sum(iv_errors)/len(iv_errors):.4f}")
                print(f"    IV Improvement: {(sum(naive_errors) - sum(iv_errors))/sum(naive_errors)*100:.1f}%")

if __name__ == "__main__":
    main()




















