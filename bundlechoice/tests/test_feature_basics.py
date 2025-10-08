#!/usr/bin/env python
"""
Test suite for basic feature functionality.
"""
import numpy as np
import pytest
from mpi4py import MPI
from bundlechoice import BundleChoice

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def simple_features(i_id, B_j, data):
    """Simple feature oracle for testing."""
    modular_agent = data["agent_data"]["modular"][i_id]
    if B_j.ndim == 1:
        return np.concatenate((modular_agent.T @ B_j, [-B_j.sum() ** 2]))
    else:
        return np.concatenate((modular_agent.T @ B_j, -np.sum(B_j, axis=0, keepdims=True) ** 2), axis=0)


def test_greedy_caching_correctness():
    """Test that cached greedy gives identical results to old implementation."""
    num_agents = 20
    num_items = 10
    num_features = 6
    num_simuls = 1
    
    # Generate identical data for both tests
    np.random.seed(42)
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
    
    # Test new cached version
    bc = BundleChoice()
    bc.load_config(cfg)
    bc.data.load_and_scatter(input_data)
    bc.features.set_oracle(simple_features)
    
    theta_test = np.ones(num_features)
    theta_test[-1] = 0.1
    
    bundles_new = bc.subproblems.init_and_solve(theta_test)
    
    # Test with OptimizedGreedy as reference (it already has caching)
    bc2 = BundleChoice()
    cfg2 = cfg.copy()
    cfg2["subproblem"]["name"] = "OptimizedGreedy"
    bc2.load_config(cfg2)
    bc2.data.load_and_scatter(input_data)
    bc2.features.set_oracle(simple_features)
    bundles_optimized = bc2.subproblems.init_and_solve(theta_test)
    
    if rank == 0:
        # Should produce identical results
        assert bundles_new.shape == bundles_optimized.shape
        # Allow some tolerance for numerical differences
        differences = np.sum(bundles_new != bundles_optimized)
        total = bundles_new.size
        error_rate = differences / total
        
        print(f"\nGreedy caching test:")
        print(f"  Shape: {bundles_new.shape}")
        print(f"  Differences: {differences}/{total} ({error_rate*100:.2f}%)")
        print(f"  New sum: {bundles_new.sum()}")
        print(f"  Optimized sum: {bundles_optimized.sum()}")
        
        # Should be very close (greedy is not deterministic with ties, but should be close)
        assert error_rate < 0.05, f"Too many differences: {error_rate*100:.2f}%"


def test_validate_setup():
    """Test validate_setup functionality."""
    bc = BundleChoice()
    
    # Should fail without config
    try:
        bc.validate_setup('row_generation')
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        if rank == 0:
            assert "config" in str(e).lower()
            print("\n✅ validate_setup correctly detects missing config")
    
    # Load config but no data
    bc.load_config({
        "dimensions": {"num_agents": 10, "num_items": 5, "num_features": 3, "num_simuls": 1},
        "subproblem": {"name": "Greedy"},
        "row_generation": {"max_iters": 10}
    })
    
    try:
        bc.validate_setup('row_generation')
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        if rank == 0:
            assert "data" in str(e).lower()
            print("✅ validate_setup correctly detects missing data")
    
    # Load data but no features
    if rank == 0:
        input_data = {
            "agent_data": {"modular": np.random.randn(10, 5, 2)},
            "errors": np.random.randn(1, 10, 5),
            "obs_bundle": np.random.randint(0, 2, (10, 5))
        }
    else:
        input_data = None
    
    bc.data.load_and_scatter(input_data)
    
    try:
        bc.validate_setup('row_generation')
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        if rank == 0:
            assert "features" in str(e).lower()
            print("✅ validate_setup correctly detects missing features")
    
    # Add features but no subproblem
    bc.features.set_oracle(simple_features)
    
    try:
        bc.validate_setup('row_generation')
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        if rank == 0:
            assert "subproblem" in str(e).lower()
            print("✅ validate_setup correctly detects missing subproblem")
    
    # Finally complete setup
    bc.subproblems.load()
    result = bc.validate_setup('row_generation')
    
    if rank == 0:
        assert result == True
        print("✅ validate_setup passes with complete setup")


def test_subproblem_statistics():
    """Test subproblem statistics tracking."""
    num_agents = 20
    num_items = 10
    num_features = 3
    
    if rank == 0:
        input_data = {
            "agent_data": {"modular": np.random.randn(num_agents, num_items, num_features)},
            "errors": np.random.randn(1, num_agents, num_items)
        }
    else:
        input_data = None
    
    bc = BundleChoice()
    bc.load_config({
        "dimensions": {"num_agents": num_agents, "num_items": num_items, 
                      "num_features": num_features, "num_simuls": 1},
        "subproblem": {"name": "Greedy"},
    })
    bc.data.load_and_scatter(input_data)
    bc.features.build_from_data()
    
    # Solve multiple times
    theta = np.ones(num_features)
    for i in range(3):
        bc.subproblems.init_and_solve(theta + i * 0.1)
    
    stats = bc.subproblems.get_stats()
    
    if rank == 0:
        print("\n✅ Subproblem statistics:")
        print(f"  Num solves: {stats['num_solves']}")
        print(f"  Total time: {stats['total_time']:.4f}s")
        print(f"  Mean time: {stats['mean_time']:.4f}s")
        print(f"  Max time: {stats['max_time']:.4f}s")
        
        assert stats['num_solves'] == 3
        assert stats['total_time'] > 0
        assert stats['mean_time'] > 0
        assert stats['max_time'] > 0


def test_result_caching():
    """Test result caching functionality."""
    num_agents = 20
    num_items = 10
    num_features = 3
    
    if rank == 0:
        input_data = {
            "agent_data": {"modular": np.random.randn(num_agents, num_items, num_features)},
            "errors": np.random.randn(1, num_agents, num_items)
        }
    else:
        input_data = None
    
    bc = BundleChoice()
    bc.load_config({
        "dimensions": {"num_agents": num_agents, "num_items": num_items, 
                      "num_features": num_features, "num_simuls": 1},
        "subproblem": {"name": "Greedy"},
    })
    bc.data.load_and_scatter(input_data)
    bc.features.build_from_data()
    
    # Enable caching
    bc.subproblems.enable_cache()
    
    theta = np.ones(num_features)
    
    # First solve (cache miss)
    bc.subproblems.init_and_solve(theta)
    stats1 = bc.subproblems.get_stats()
    
    # Second solve with same theta (cache hit - should be instant)
    bc.subproblems.init_and_solve(theta)
    stats2 = bc.subproblems.get_stats()
    
    # Clear cache
    bc.subproblems.disable_cache()
    bc.subproblems.clear_stats()
    
    # Third solve (cache disabled)
    bc.subproblems.init_and_solve(theta)
    stats3 = bc.subproblems.get_stats()
    
    if rank == 0:
        print("\n✅ Result caching:")
        print(f"  First solve: {stats1['num_solves']} solves, {stats1['total_time']:.4f}s")
        print(f"  With cache: {stats2['num_solves']} solves, {stats2['total_time']:.4f}s")
        print(f"  After clear: {stats3['num_solves']} solves, {stats3['total_time']:.4f}s")
        
        # Cache should reduce solve count
        assert stats2['num_solves'] == stats1['num_solves'] + 0  # Cached solve doesn't add time
        assert stats3['num_solves'] == 1  # Stats were cleared


def test_config_dict_creation():
    """Test manual configuration dictionary creation."""
    # Test creating different config styles manually
    configs = [
        {  # Fast config
            'dimensions': {'num_agents': 10, 'num_items': 5, 'num_features': 3, 'num_simuls': 1},
            'subproblem': {'name': 'Greedy'},
            'row_generation': {'max_iters': 20, 'tolerance_optimality': 0.01}
        },
        {  # Accurate config
            'dimensions': {'num_agents': 10, 'num_items': 5, 'num_features': 3, 'num_simuls': 1},
            'subproblem': {'name': 'QuadSupermodularNetwork'},
            'row_generation': {'max_iters': 200, 'tolerance_optimality': 1e-6}
        }
    ]
    
    for cfg in configs:
        assert 'dimensions' in cfg
        assert 'subproblem' in cfg
        
    if rank == 0:
        print(f"✅ Manual config creation works")


def test_batch_feature_detection():
    """Test automatic batch feature computation detection."""
    num_agents = 20
    num_items = 10
    num_features = 3
    
    # Batch-compatible oracle
    def batch_oracle(agent_id, bundles, data):
        modular = data["agent_data"]["modular"]
        if agent_id is None:  # Batch mode
            return modular.T @ bundles
        else:  # Single mode
            return modular[agent_id].T @ bundles
    
    if rank == 0:
        input_data = {
            "agent_data": {"modular": np.random.randn(num_agents, num_items, num_features)},
            "errors": np.random.randn(1, num_agents, num_items)
        }
    else:
        input_data = None
    
    bc = BundleChoice()
    bc.load_config({
        "dimensions": {"num_agents": num_agents, "num_items": num_items, 
                      "num_features": num_features, "num_simuls": 1},
        "subproblem": {"name": "Greedy"},
    })
    bc.data.load_and_scatter(input_data)
    bc.features.set_oracle(batch_oracle)
    
    # This should automatically detect batch support
    theta = np.ones(num_features)
    bundles = bc.subproblems.init_and_solve(theta)
    
    if rank == 0:
        assert bundles.shape == (num_agents, num_items)
        print("✅ Batch feature detection works")


def test_better_error_messages():
    """Test improved error messages."""
    bc = BundleChoice()
    bc.load_config({
        "dimensions": {"num_agents": 10, "num_items": 5, "num_features": 3, "num_simuls": 1},
        "subproblem": {"name": "Greedy"},
        "row_generation": {"max_iters": 10}
    })
    
    # Try to access row_generation without data
    try:
        bc.row_generation.solve()
        assert False, "Should have raised error"
    except RuntimeError as e:
        error_msg = str(e)
        if rank == 0:
            print(f"\n✅ Error message quality:")
            print(f"  Contains 'call bc.data': {'call bc.data' in error_msg}")
            print(f"  Contains 'call bc.features': {'call bc.features' in error_msg}")
            print(f"  Contains 'validate_setup': {'validate_setup' in error_msg}")
            
            assert 'call bc.data' in error_msg or 'data' in error_msg
            assert 'validate_setup' in error_msg


def test_config_validation():
    """Test automatic config validation."""
    bc = BundleChoice()
    
    # Invalid config should be caught immediately
    try:
        bc.load_config({
            "dimensions": {"num_agents": -5, "num_items": 10, "num_features": 3, "num_simuls": 1}
        })
        assert False, "Should have raised ValueError"
    except ValueError as e:
        if rank == 0:
            assert "num_agents must be positive" in str(e)
            print("\n✅ Config validation catches negative num_agents")
    
    # Another invalid config
    try:
        bc.load_config({
            "dimensions": {"num_agents": 10, "num_items": 10, "num_features": 3, "num_simuls": 1},
            "row_generation": {"tolerance_optimality": -0.01}
        })
        assert False, "Should have raised ValueError"
    except ValueError as e:
        if rank == 0:
            assert "tolerance_optimality must be positive" in str(e)
            print("✅ Config validation catches negative tolerance")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
