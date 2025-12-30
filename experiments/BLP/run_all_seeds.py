#!/usr/bin/env python3
"""Run all experiments with different seeds and collect results."""
import sys
import re
import subprocess
import os
from collections import defaultdict

def run_with_seed(script_path, seed):
    """Run experiment script with a specific seed."""
    script_abs = os.path.abspath(script_path)
    script_dir = os.path.dirname(script_abs)
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    with open(script_abs, 'r') as f:
        content = f.read()
    
    # Replace seed parameter
    content = content.replace('seed=None', f'seed={seed}')
    content = content.replace('seed=42', f'seed={seed}')
    
    # Also try regex replacement for other patterns
    content = re.sub(
        r'prepared = scenario\.prepare\(comm=comm, seed=[^,)]+',
        f'prepared = scenario.prepare(comm=comm, seed={seed}',
        content
    )
    
    temp_path = os.path.join(script_dir, os.path.basename(script_path).replace('.py', f'_temp{seed}.py'))
    with open(temp_path, 'w') as f:
        f.write(content)
    
    try:
        result = subprocess.run(
            ['mpirun', '-n', '10', 'python', temp_path],
            capture_output=True,
            text=True,
            timeout=1800,
            cwd=project_root
        )
        return result.stdout + result.stderr
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def extract_results_table(output):
    """Extract the results table from output."""
    if 'RESULTS' not in output:
        return None
    
    start = output.find('RESULTS')
    # Find the end (look for the closing ==== line)
    lines = output[start:].split('\n')
    result_lines = []
    for i, line in enumerate(lines):
        result_lines.append(line)
        if i > 5 and '='*100 in line:
            break
    
    return '\n'.join(result_lines)

def main():
    experiments = [
        ('experiments/BLP/experiment_inversion_knapsack_factory.py', 'Knapsack'),
        ('experiments/BLP/experiment_inversion_greedy_factory.py', 'Greedy'),
        ('experiments/BLP/experiment_inversion_supermod_factory.py', 'Supermodular'),
        ('experiments/BLP/experiment_inversion_quadknapsack_factory.py', 'QuadKnapsack'),
    ]
    
    seeds = [0, 1, 2, 3, 4]
    all_results = {}
    
    for script, name in experiments:
        print(f'\n\n{"="*80}')
        print(f'Running {name} experiments')
        print("="*80)
        all_results[name] = {}
        
        for seed in seeds:
            print(f'\n{name} seed {seed}...')
            output = run_with_seed(script, seed)
            
            if 'RESULTS' in output:
                result_table = extract_results_table(output)
                print(result_table)
                all_results[name][seed] = result_table
            else:
                print('No RESULTS found in output')
                print(output[-1000:])
    
    # Print final summary
    print('\n\n' + '='*100)
    print('FINAL SUMMARY - ALL EXPERIMENTS')
    print('='*100)
    
    for name, seed_results in all_results.items():
        print(f'\n\n{name}:')
        print('-'*80)
        for seed in sorted(seed_results.keys()):
            print(f'\nSeed {seed}:')
            print(seed_results[seed])

if __name__ == '__main__':
    main()

