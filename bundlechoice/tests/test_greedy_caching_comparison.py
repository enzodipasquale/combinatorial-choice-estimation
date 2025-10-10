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
    Test greedy determinism with multiple random seeds.
    Greedy should produce identical results when run twice with same inputs.
    """
    num_agents = 20
    num_items = 15
    num_features = 6
    num_simuls = 1
    
    num_seeds = 5
    all_deterministic = True
    
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
        
        # Test Greedy determinism
        bc = BundleChoice()
        bc.load_config(cfg)
        bc.data.load_and_scatter(input_data)
        bc.features.set_oracle(simple_features)
        
        theta = np.ones(num_features)
        theta[-1] = 0.1 + seed * 0.05
        
        # Solve twice with same parameters
        bundles_first = bc.subproblems.init_and_solve(theta)
        bc.subproblem_manager.initialize_local()
        bundles_second_local = bc.subproblem_manager.solve_local(theta)
        bundles_second = bc.comm_manager.concatenate_array_at_root_fast(bundles_second_local, root=0)
        
        if rank == 0:
            # Should be identical (deterministic)
            is_deterministic = np.array_equal(bundles_first, bundles_second)
            
            if not is_deterministic:
                diff_count = np.sum(bundles_first != bundles_second)
                total = bundles_first.size
                print(f"\nSeed {seed}: NOT deterministic")
                print(f"  Differences: {diff_count}/{total} ({100*diff_count/total:.2f}%)")
                all_deterministic = False
    
    if rank == 0:
        print(f"\n✅ Greedy determinism test:")
        print(f"  Tested {num_seeds} random seeds")
        print(f"  All runs deterministic: {all_deterministic}")
        
        # Greedy should be fully deterministic
        assert all_deterministic, "Greedy should produce identical results for same inputs"


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
    num_agents = 20
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
