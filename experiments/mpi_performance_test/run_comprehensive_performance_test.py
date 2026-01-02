"""
Comprehensive performance test comparing main vs feature branch.
Tests multiple scales and MPI rank counts.
"""
import sys
import time
from datetime import datetime
import numpy as np
from bundlechoice import BundleChoice
from bundlechoice.factory import ScenarioLibrary

def run_test(num_agents, num_items, num_simulations, num_features, max_iters, seed, branch_name):
    """Run a single test and return timing statistics."""
    print(f"\n{'='*80}")
    print(f"Testing: {branch_name} | Agents: {num_agents}, Items: {num_items}, Simuls: {num_simulations}")
    print(f"{'='*80}")
    
    # Generate data
    scenario = ScenarioLibrary.quadratic_supermodular(
        num_agents=num_agents,
        num_items=num_items,
        num_simulations=num_simulations,
        num_features=num_features,
        seed=seed
    )
    
    # Create BundleChoice instance
    bc = BundleChoice.from_scenario(scenario)
    
    # Run estimation
    start_time = time.time()
    result = bc.estimate(max_iters=max_iters)
    total_time = time.time() - start_time
    
    # Extract timing statistics
    timing_stats = result.diagnostics.get('timing_breakdown', {})
    
    # Print summary
    print(f"\nResults for {branch_name}:")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Iterations: {result.diagnostics.get('num_iterations', 'N/A')}")
    
    if timing_stats:
        print(f"  Key timing components:")
        print(f"    pricing: {timing_stats.get('pricing', 0):.3f}s")
        print(f"    mpi_gather: {timing_stats.get('mpi_gather', 0):.3f}s")
        print(f"    compute_features: {timing_stats.get('compute_features', 0):.3f}s")
        print(f"    gather_features: {timing_stats.get('gather_features', 0):.3f}s")
        print(f"    compute_errors: {timing_stats.get('compute_errors', 0):.3f}s")
        print(f"    master_update: {timing_stats.get('master_update', 0):.3f}s")
    
    return {
        'total_time': total_time,
        'iterations': result.diagnostics.get('num_iterations', 0),
        'timing_stats': timing_stats
    }

def main():
    """Run comprehensive performance tests."""
    print("="*80)
    print("COMPREHENSIVE PERFORMANCE TEST: MAIN vs FEATURE BRANCH")
    print("="*80)
    
    # Test configurations: (num_agents, num_items, num_simulations, num_features, max_iters, seed)
    test_configs = [
        # Medium scale
        (128, 100, 5, 6, 50, 42),
        # Large scale
        (256, 150, 5, 6, 50, 42),
        # XL scale
        (512, 200, 5, 6, 50, 42),
    ]
    
    # MPI rank counts to test
    mpi_ranks = [4, 8, 12]
    
    results = {}
    
    for num_agents, num_items, num_simulations, num_features, max_iters, seed in test_configs:
        scale_name = f"{num_agents}x{num_items}"
        results[scale_name] = {}
        
        for num_ranks in mpi_ranks:
            print(f"\n\n{'#'*80}")
            print(f"TESTING: {scale_name} with {num_ranks} MPI ranks")
            print(f"{'#'*80}\n")
            
            # Set MPI ranks (this will be handled by mpirun, but we note it)
            rank_key = f"{num_ranks}ranks"
            results[scale_name][rank_key] = {}
            
            # Test main branch
            try:
                main_result = run_test(
                    num_agents, num_items, num_simulations, num_features, max_iters, seed,
                    f"MAIN ({num_ranks} ranks)"
                )
                results[scale_name][rank_key]['main'] = main_result
            except Exception as e:
                print(f"ERROR testing main branch: {e}")
                results[scale_name][rank_key]['main'] = {'error': str(e)}
            
            # Test feature branch
            try:
                feature_result = run_test(
                    num_agents, num_items, num_simulations, num_features, max_iters, seed,
                    f"FEATURE ({num_ranks} ranks)"
                )
                results[scale_name][rank_key]['feature'] = feature_result
            except Exception as e:
                print(f"ERROR testing feature branch: {e}")
                results[scale_name][rank_key]['feature'] = {'error': str(e)}
            
            # Compare results
            if 'main' in results[scale_name][rank_key] and 'feature' in results[scale_name][rank_key]:
                main = results[scale_name][rank_key]['main']
                feature = results[scale_name][rank_key]['feature']
                
                if 'error' not in main and 'error' not in feature:
                    main_time = main['total_time']
                    feature_time = feature['total_time']
                    speedup = main_time / feature_time if feature_time > 0 else 0
                    slowdown = feature_time / main_time if main_time > 0 else 0
                    
                    print(f"\n{'='*80}")
                    print(f"COMPARISON: {scale_name} ({num_ranks} ranks)")
                    print(f"{'='*80}")
                    print(f"Main branch:    {main_time:.3f}s")
                    print(f"Feature branch: {feature_time:.3f}s")
                    if speedup > 1:
                        print(f"Feature is {speedup:.2f}x FASTER")
                    elif slowdown > 1:
                        print(f"Feature is {slowdown:.2f}x SLOWER")
                    else:
                        print(f"Performance is SIMILAR")
                    
                    # Compare key timing components
                    main_timing = main.get('timing_stats', {})
                    feature_timing = feature.get('timing_stats', {})
                    
                    print(f"\nTiming Component Comparison:")
                    components = ['pricing', 'mpi_gather', 'compute_features', 'gather_features', 
                                 'compute_errors', 'master_update']
                    for comp in components:
                        main_val = main_timing.get(comp, 0)
                        feature_val = feature_timing.get(comp, 0)
                        diff = feature_val - main_val
                        pct_diff = (diff / main_val * 100) if main_val > 0 else 0
                        print(f"  {comp:20s}: Main={main_val:7.3f}s, Feature={feature_val:7.3f}s, Diff={diff:+7.3f}s ({pct_diff:+6.1f}%)")
    
    # Print final summary
    print(f"\n\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}\n")
    
    for scale_name in results:
        print(f"\n{scale_name}:")
        for rank_key in results[scale_name]:
            if 'main' in results[scale_name][rank_key] and 'feature' in results[scale_name][rank_key]:
                main = results[scale_name][rank_key]['main']
                feature = results[scale_name][rank_key]['feature']
                
                if 'error' not in main and 'error' not in feature:
                    main_time = main['total_time']
                    feature_time = feature['total_time']
                    diff_pct = ((feature_time - main_time) / main_time * 100) if main_time > 0 else 0
                    
                    print(f"  {rank_key:12s}: Main={main_time:6.3f}s, Feature={feature_time:6.3f}s, Diff={diff_pct:+6.1f}%")

if __name__ == "__main__":
    main()



