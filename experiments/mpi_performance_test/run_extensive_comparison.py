"""
Extensive performance comparison: Main vs Feature branch
Tests multiple scales (Medium, Large, XL) with 4, 8, 12 MPI ranks
"""
import subprocess
import sys
import os
import time
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "experiments" / "mpi_performance_test" / "comparison_results"
RESULTS_DIR.mkdir(exist_ok=True)

# Test configurations
TEST_CONFIGS = [
    # (scale_name, num_agents, num_items, num_simulations, num_features, max_iters, seed)
    ("Medium", 128, 100, 5, 6, 50, 42),
    ("Large", 256, 150, 5, 6, 50, 42),
    ("XL", 512, 200, 5, 6, 50, 42),
]

MPI_RANKS = [4, 8, 12]

def run_mpi_test(branch, scale_name, num_agents, num_items, num_simulations, num_features, max_iters, seed, num_ranks):
    """Run a single MPI test and return results."""
    print(f"\n{'='*80}")
    print(f"Testing: {branch} branch | {scale_name} ({num_agents}x{num_items}) | {num_ranks} ranks")
    print(f"{'='*80}")
    
    # Switch to correct branch
    subprocess.run(["git", "checkout", branch], cwd=PROJECT_ROOT, check=True, 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["git", "pull", "origin", branch], cwd=PROJECT_ROOT, check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Run test
    cmd = [
        "mpirun", "-n", str(num_ranks),
        "python", "experiments/mpi_performance_test/run_single_test.py",
        "--agents", str(num_agents),
        "--items", str(num_items),
        "--simuls", str(num_simulations),
        "--features", str(num_features),
        "--max-iters", str(max_iters),
        "--seed", str(seed)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            print(f"ERROR: Test failed with return code {result.returncode}")
            print(result.stderr)
            return None
        
        # Parse output
        output = result.stdout
        results = {}
        
        for line in output.split('\n'):
            if 'Total time:' in line:
                results['total_time'] = float(line.split(':')[1].strip().replace('s', ''))
            elif 'Total iterations:' in line:
                results['iterations'] = int(line.split(':')[1].strip())
            elif ':' in line and 's' in line:
                parts = line.split(':')
                if len(parts) == 2:
                    key = parts[0].strip()
                    val_str = parts[1].strip().replace('s', '')
                    try:
                        results[key] = float(val_str)
                    except:
                        pass
        
        print(f"  Total time: {results.get('total_time', 0):.3f}s")
        print(f"  Iterations: {results.get('iterations', 0)}")
        print(f"  Pricing: {results.get('pricing', 0):.3f}s")
        print(f"  MPI gather: {results.get('mpi_gather', 0):.3f}s")
        print(f"  Compute features: {results.get('compute_features', 0):.3f}s")
        print(f"  Gather features: {results.get('gather_features', 0):.3f}s")
        
        return results
        
    except subprocess.TimeoutExpired:
        print(f"ERROR: Test timed out after 5 minutes")
        return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None

def main():
    """Run extensive comparison tests."""
    print("="*80)
    print("EXTENSIVE PERFORMANCE COMPARISON: MAIN vs FEATURE BRANCH")
    print("="*80)
    print(f"Testing {len(TEST_CONFIGS)} scales × {len(MPI_RANKS)} rank counts = {len(TEST_CONFIGS) * len(MPI_RANKS)} configurations per branch")
    print(f"Total tests: {len(TEST_CONFIGS) * len(MPI_RANKS) * 2} (2 branches)")
    print("="*80)
    
    all_results = {}
    
    for scale_name, num_agents, num_items, num_simulations, num_features, max_iters, seed in TEST_CONFIGS:
        all_results[scale_name] = {}
        
        for num_ranks in MPI_RANKS:
            rank_key = f"{num_ranks}ranks"
            all_results[scale_name][rank_key] = {}
            
            # Test main branch
            print(f"\n\n{'#'*80}")
            print(f"TESTING MAIN: {scale_name} with {num_ranks} ranks")
            print(f"{'#'*80}")
            main_result = run_mpi_test(
                "main", scale_name, num_agents, num_items, num_simulations, 
                num_features, max_iters, seed, num_ranks
            )
            all_results[scale_name][rank_key]['main'] = main_result
            time.sleep(2)  # Brief pause
            
            # Test feature branch
            print(f"\n\n{'#'*80}")
            print(f"TESTING FEATURE: {scale_name} with {num_ranks} ranks")
            print(f"{'#'*80}")
            feature_result = run_mpi_test(
                "feature/mpi-gather-optimization", scale_name, num_agents, num_items, num_simulations,
                num_features, max_iters, seed, num_ranks
            )
            all_results[scale_name][rank_key]['feature'] = feature_result
            time.sleep(2)  # Brief pause
            
            # Compare
            if main_result and feature_result:
                main_time = main_result.get('total_time', 0)
                feature_time = feature_result.get('total_time', 0)
                
                if main_time > 0:
                    diff_pct = ((feature_time - main_time) / main_time) * 100
                    print(f"\n{'='*80}")
                    print(f"COMPARISON: {scale_name} ({num_ranks} ranks)")
                    print(f"{'='*80}")
                    print(f"Main:    {main_time:.3f}s")
                    print(f"Feature: {feature_time:.3f}s")
                    print(f"Difference: {diff_pct:+.1f}%")
                    
                    if abs(diff_pct) < 2:
                        print("→ PERFORMANCE IS SIMILAR (<2% difference)")
                    elif diff_pct < 0:
                        print(f"→ FEATURE IS {abs(diff_pct):.1f}% FASTER")
                    else:
                        print(f"→ FEATURE IS {diff_pct:.1f}% SLOWER")
    
    # Save results to JSON
    results_file = RESULTS_DIR / f"comparison_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nResults saved to: {results_file}")
    
    # Print final summary
    print(f"\n\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}\n")
    print(f"{'Scale':<12} {'Ranks':<6} {'Main (s)':<10} {'Feature (s)':<12} {'Diff %':<10} {'Status':<15}")
    print("-"*80)
    
    for scale_name in TEST_CONFIGS:
        scale_name = scale_name[0]
        if scale_name in all_results:
            for rank_key in sorted(all_results[scale_name].keys()):
                main = all_results[scale_name][rank_key].get('main')
                feature = all_results[scale_name][rank_key].get('feature')
                
                if main and feature:
                    main_time = main.get('total_time', 0)
                    feature_time = feature.get('total_time', 0)
                    
                    if main_time > 0:
                        diff_pct = ((feature_time - main_time) / main_time) * 100
                        
                        if abs(diff_pct) < 2:
                            status = "SIMILAR"
                        elif diff_pct < 0:
                            status = f"FASTER {abs(diff_pct):.1f}%"
                        else:
                            status = f"SLOWER {diff_pct:.1f}%"
                        
                        print(f"{scale_name:<12} {rank_key:<6} {main_time:<10.3f} {feature_time:<12.3f} {diff_pct:+9.1f}% {status:<15}")

if __name__ == "__main__":
    main()



