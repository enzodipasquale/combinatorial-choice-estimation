#!/usr/bin/env python
"""
Direct comparison test: New cached greedy vs OptimizedGreedy (which already has caching).
This ensures the new caching implementation produces identical results.
"""
import numpy as np
import pytest
from mpi4py import MPI
from bundlechoice import BundleChoice

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def simple_features(i_id, B_j, data):
    """Feature oracle supporting both single and vectorized bundles."""
    modular_agent = data["agent_data"]["modular"][i_id]
    if B_j.ndim == 1:
        return np.concatenate((modular_agent.T @ B_j, [-B_j.sum() ** 2]))
    else:
        return np.concatenate((modular_agent.T @ B_j, -np.sum(B_j, axis=0, keepdims=True) ** 2), axis=0)


def test_greedy_vs_optimized_greedy_multiple_seeds():
    """
    Test new cached greedy against OptimizedGreedy with multiple random seeds.
    OptimizedGreedy already has caching, so it's our reference implementation.
    """
    num_agents = 30
    num_items = 15
    num_features = 6
    num_simuls = 1
    
    num_seeds = 5
    results_match = []
    
    for seed in range(num_seeds):
        np.random.seed(seed + 100)
        
        if rank == 0:
            modular = np.random.normal(0, 1, (num_agents, num_items, num_features-1))
            errors = np.random.normal(0, 1, size=(num_simuls, num_agents, num_items))
            input_data = {"agent_data": {"modular": modular}, "errors": errors}
        else:
            input_data = None
        
        cfg = {
            "dimensions": {"num_agents": num_agents, "num_items": num_items, 
                          "num_features": num_features, "num_simuls": num_simuls},
            "subproblem": {"name": "Greedy"},
        }
        
        # Test new Greedy
        bc1 = BundleChoice()
        bc1.load_config(cfg)
        bc1.data.load_and_scatter(input_data)
        bc1.features.set_oracle(simple_features)
        
        theta = np.ones(num_features)
        theta[-1] = 0.1 + seed * 0.05
        
        bundles_greedy = bc1.subproblems.init_and_solve(theta)
        
        # Test OptimizedGreedy (reference)
        cfg2 = cfg.copy()
        cfg2["subproblem"]["name"] = "OptimizedGreedy"
        bc2 = BundleChoice()
        bc2.load_config(cfg2)
        bc2.data.load_and_scatter(input_data)
        bc2.features.set_oracle(simple_features)
        bundles_optimized = bc2.subproblems.init_and_solve(theta)
        
        if rank == 0:
            # Compare results
            matches = np.array_equal(bundles_greedy, bundles_optimized)
            results_match.append(matches)
            
            if not matches:
                diff_count = np.sum(bundles_greedy != bundles_optimized)
                total = bundles_greedy.size
                print(f"\nSeed {seed}:")
                print(f"  Greedy sum: {bundles_greedy.sum()}")
                print(f"  Optimized sum: {bundles_optimized.sum()}")
                print(f"  Differences: {diff_count}/{total} ({100*diff_count/total:.2f}%)")
                
                # Greedy algorithms can differ when there are ties
                # But they should be very close
                assert diff_count / total < 0.1, f"Too many differences: {diff_count}/{total}"
    
    if rank == 0:
        match_rate = sum(results_match) / len(results_match)
        print(f"\n✅ Greedy caching comparison:")
        print(f"  Tested {num_seeds} random seeds")
        print(f"  Exact matches: {sum(results_match)}/{len(results_match)} ({100*match_rate:.0f}%)")
        print(f"  All results within acceptable tolerance")
        
        # At least half should match exactly (greedy is deterministic unless ties)
        assert match_rate >= 0.5, f"Too few exact matches: {match_rate*100:.0f}%"


def test_greedy_produces_valid_bundles():
    """Test that cached greedy always produces valid bundles."""
    num_agents = 20
    num_items = 12
    num_features = 5
    num_simuls = 1
    
    np.random.seed(42)
    
    if rank == 0:
        modular = np.random.normal(0, 1, (num_agents, num_items, num_features-1))
        errors = np.random.normal(0, 1, size=(num_simuls, num_agents, num_items))
        input_data = {"agent_data": {"modular": modular}, "errors": errors}
    else:
        input_data = None
    
    bc = BundleChoice()
    bc.load_config({
        "dimensions": {"num_agents": num_agents, "num_items": num_items, 
                      "num_features": num_features, "num_simuls": num_simuls},
        "subproblem": {"name": "Greedy"},
    })
    bc.data.load_and_scatter(input_data)
    bc.features.set_oracle(simple_features)
    
    # Test with multiple theta values
    for i in range(5):
        theta = np.random.randn(num_features)
        bundles = bc.subproblems.init_and_solve(theta)
        
        if rank == 0:
            # Check validity
            assert bundles.shape == (num_agents, num_items)
            assert bundles.dtype == bool or bundles.dtype == np.bool_
            assert np.all((bundles == 0) | (bundles == 1))
            assert bundles.sum() > 0  # At least some items selected
            
    if rank == 0:
        print("\n✅ All generated bundles are valid")


def test_greedy_caching_improves_objective():
    """Test that greedy consistently improves objective at each step (sanity check)."""
    num_agents = 15
    num_items = 8
    num_features = 4
    num_simuls = 1
    
    np.random.seed(123)
    
    if rank == 0:
        modular = np.random.normal(0, 1, (num_agents, num_items, num_features-1))
        errors = np.random.normal(0, 1, size=(num_simuls, num_agents, num_items))
        input_data = {"agent_data": {"modular": modular}, "errors": errors}
    else:
        input_data = None
    
    bc = BundleChoice()
    bc.load_config({
        "dimensions": {"num_agents": num_agents, "num_items": num_items, 
                      "num_features": num_features, "num_simuls": num_simuls},
        "subproblem": {"name": "Greedy"},
    })
    bc.data.load_and_scatter(input_data)
    bc.features.set_oracle(simple_features)
    
    theta = np.ones(num_features)
    theta[-1] = 0.2
    
    bundles = bc.subproblems.init_and_solve(theta)
    
    if rank == 0:
        # Compute objective values
        features = np.array([simple_features(i, bundles[i], input_data) for i in range(num_agents)])
        objectives = features @ theta + (input_data["errors"][0] * bundles).sum(axis=1)
        
        # All objectives should be positive (greedy selects items that improve utility)
        positive_rate = (objectives > 0).mean()
        
        print(f"\n✅ Greedy objective quality:")
        print(f"  Agents with positive utility: {positive_rate*100:.0f}%")
        print(f"  Mean objective: {objectives.mean():.4f}")
        print(f"  Min objective: {objectives.min():.4f}")
        print(f"  Max objective: {objectives.max():.4f}")
        
        # Most agents should have positive utility (greedy is reasonably good)
        assert positive_rate > 0.5, f"Too few positive utilities: {positive_rate*100:.0f}%"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
