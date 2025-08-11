#!/usr/bin/env python3
"""
Deep analysis of BundleChoice user experience.
Exploring edge cases, error scenarios, and different user workflows.
"""

import numpy as np
import pytest
from mpi4py import MPI
from bundlechoice.core import BundleChoice
from bundlechoice.config import BundleChoiceConfig

def simple_features_oracle(i_id, B_j, data):
    """Simple feature oracle for testing."""
    agent_features = data["agent_data"]["features"][i_id]
    bundle_sum = np.sum(B_j, axis=0)
    features = agent_features * bundle_sum
    return features

def test_error_scenarios():
    """Test various error scenarios and user confusion points."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("\n" + "="*60)
        print("DEEP ANALYSIS: ERROR SCENARIOS")
        print("="*60)
        
        # Test 1: Missing subproblem initialization
        print("\n1. ERROR: Missing subproblem initialization")
        print("-" * 40)
        try:
            bc = BundleChoice()
            bc.load_config({
                "dimensions": {"num_agents": 10, "num_items": 5, "num_features": 2, "num_simuls": 1},
                "subproblem": {"name": "Greedy"},
                "ellipsoid": {"num_iters": 5, "verbose": False}
            })
            
            # Generate data
            agent_features = np.random.normal(0, 1, (10, 2))
            obs_bundles = np.random.choice([0, 1], size=(10, 5), p=[0.6, 0.4])
            errors = np.random.normal(0, 0.1, size=(10, 5))
            
            input_data = {
                "agent_data": {"features": agent_features},
                "obs_bundle": obs_bundles,
                "errors": errors
            }
            
            bc.data.load_and_scatter(input_data)
            bc.features.set_oracle(simple_features_oracle)
            
            # This will fail - missing bc.subproblems.load()
            theta_hat = bc.ellipsoid.solve()
            
        except Exception as e:
            print(f"❌ Error: {e}")
            print("💡 User confusion: No clear guidance on what's missing")
            print("💡 Solution needed: Better error message with actionable guidance")
        
        # Test 2: Wrong configuration structure
        print("\n2. ERROR: Wrong configuration structure")
        print("-" * 40)
        try:
            bc = BundleChoice()
            # Wrong config structure
            bc.load_config({
                "agents": 10,  # Wrong key
                "items": 5,    # Wrong key
                "features": 2,  # Wrong key
                "subproblem": "Greedy"  # Wrong structure
            })
            
        except Exception as e:
            print(f"❌ Error: {e}")
            print("💡 User confusion: No clear guidance on correct config structure")
            print("💡 Solution needed: Validation with examples")
        
        # Test 3: Missing data
        print("\n3. ERROR: Missing data")
        print("-" * 40)
        try:
            bc = BundleChoice()
            bc.load_config({
                "dimensions": {"num_agents": 10, "num_items": 5, "num_features": 2, "num_simuls": 1},
                "subproblem": {"name": "Greedy"},
                "ellipsoid": {"num_iters": 5, "verbose": False}
            })
            
            # Try to solve without loading data
            theta_hat = bc.ellipsoid.solve()
            
        except Exception as e:
            print(f"❌ Error: {e}")
            print("💡 User confusion: No clear guidance on required data")
            print("💡 Solution needed: Step-by-step validation")

def test_workflow_variations():
    """Test different user workflow variations."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("\n" + "="*60)
        print("DEEP ANALYSIS: WORKFLOW VARIATIONS")
        print("="*60)
        
        # Test 1: User who forgets the order
        print("\n1. WORKFLOW: User forgets initialization order")
        print("-" * 40)
        
        bc = BundleChoice()
        
        # Wrong order: try to set features before data
        try:
            bc.load_config({
                "dimensions": {"num_agents": 10, "num_items": 5, "num_features": 2, "num_simuls": 1},
                "subproblem": {"name": "Greedy"},
                "ellipsoid": {"num_iters": 5, "verbose": False}
            })
            
            bc.features.set_oracle(simple_features_oracle)  # Wrong order!
            bc.data.load_and_scatter({"agent_data": {"features": np.random.normal(0, 1, (10, 2))}})
            
        except Exception as e:
            print(f"❌ Error: {e}")
            print("💡 User confusion: No guidance on correct order")
            print("💡 Solution needed: Clear workflow documentation")
        
        # Test 2: User who wants to change configuration
        print("\n2. WORKFLOW: User wants to change configuration")
        print("-" * 40)
        
        bc = BundleChoice()
        
        # Initial setup
        bc.load_config({
            "dimensions": {"num_agents": 10, "num_items": 5, "num_features": 2, "num_simuls": 1},
            "subproblem": {"name": "Greedy"},
            "ellipsoid": {"num_iters": 5, "verbose": False}
        })
        
        # User wants to change to row generation
        try:
            bc.load_config({
                "row_generation": {"max_iters": 10, "tolerance_optimality": 0.001}
            })
            
            # Does the user know they need to reinitialize?
            print("💡 User confusion: What happens to existing setup?")
            print("💡 Solution needed: Clear guidance on config updates")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Test 3: User who wants to experiment
        print("\n3. WORKFLOW: User wants to experiment")
        print("-" * 40)
        
        # User wants to try different solvers
        solvers = ["ellipsoid", "row_generation"]
        results = {}
        
        for solver in solvers:
            try:
                bc = BundleChoice()
                bc.load_config({
                    "dimensions": {"num_agents": 10, "num_items": 5, "num_features": 2, "num_simuls": 1},
                    "subproblem": {"name": "Greedy"},
                    solver: {"num_iters": 5, "verbose": False}
                })
                
                # Generate data
                agent_features = np.random.normal(0, 1, (10, 2))
                obs_bundles = np.random.choice([0, 1], size=(10, 5), p=[0.6, 0.4])
                errors = np.random.normal(0, 0.1, size=(10, 5))
                
                input_data = {
                    "agent_data": {"features": agent_features},
                    "obs_bundle": obs_bundles,
                    "errors": errors
                }
                
                bc.data.load_and_scatter(input_data)
                bc.features.set_oracle(simple_features_oracle)
                bc.subproblems.load()
                
                if solver == "ellipsoid":
                    theta_hat = bc.ellipsoid.solve()
                else:
                    theta_hat = bc.row_generation.solve()
                
                results[solver] = theta_hat
                print(f"✅ {solver}: {theta_hat}")
                
            except Exception as e:
                print(f"❌ {solver}: {e}")
        
        print("💡 User confusion: Different APIs for different solvers")
        print("💡 Solution needed: Unified interface")

def test_configuration_complexity():
    """Test configuration complexity and user confusion."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("\n" + "="*60)
        print("DEEP ANALYSIS: CONFIGURATION COMPLEXITY")
        print("="*60)
        
        # Test 1: Complex configuration
        print("\n1. COMPLEXITY: Full configuration example")
        print("-" * 40)
        
        complex_config = {
            "dimensions": {
                "num_agents": 100,
                "num_items": 20,
                "num_features": 3,
                "num_simuls": 1
            },
            "subproblem": {
                "name": "Greedy",
                "settings": {
                    "tolerance": 1e-6,
                    "max_iterations": 1000
                }
            },
            "row_generation": {
                "max_iters": 100,
                "tolerance_optimality": 0.001,
                "min_iters": 1,
                "master_settings": {
                    "OutputFlag": 0,
                    "LogToConsole": 0
                }
            },
            "ellipsoid": {
                "num_iters": 50,
                "initial_radius": 10.0,
                "tolerance": 1e-6,
                "decay_factor": 0.95,
                "min_volume": 1e-10,
                "verbose": False
            }
        }
        
        print("Current config structure:")
        print(f"- {len(complex_config)} top-level keys")
        print(f"- {sum(len(v) if isinstance(v, dict) else 1 for v in complex_config.values())} total parameters")
        print("💡 User confusion: Too many parameters to remember")
        print("💡 Solution needed: Sensible defaults and helper methods")
        
        # Test 2: Configuration validation
        print("\n2. COMPLEXITY: Configuration validation")
        print("-" * 40)
        
        invalid_configs = [
            # Missing dimensions
            {"subproblem": {"name": "Greedy"}},
            
            # Wrong dimension types
            {"dimensions": {"num_agents": "abc", "num_items": 5, "num_features": 2, "num_simuls": 1}},
            
            # Invalid subproblem
            {"dimensions": {"num_agents": 10, "num_items": 5, "num_features": 2, "num_simuls": 1}, "subproblem": {"name": "Invalid"}},
            
            # Missing required parameters
            {"dimensions": {"num_agents": 10, "num_items": 5, "num_features": 2, "num_simuls": 1}, "subproblem": {"name": "Greedy"}, "ellipsoid": {}},
        ]
        
        for i, config in enumerate(invalid_configs, 1):
            try:
                bc = BundleChoice()
                bc.load_config(config)
                print(f"❌ Config {i} should have failed but didn't")
            except Exception as e:
                print(f"✅ Config {i} correctly failed: {e}")
        
        print("💡 User confusion: No clear validation feedback")
        print("💡 Solution needed: Better validation with suggestions")

def test_performance_implications():
    """Test performance implications of different patterns."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("\n" + "="*60)
        print("DEEP ANALYSIS: PERFORMANCE IMPLICATIONS")
        print("="*60)
        
        # Test 1: Repeated initialization
        print("\n1. PERFORMANCE: Repeated initialization")
        print("-" * 40)
        
        bc = BundleChoice()
        
        # Time multiple config loads
        import time
        
        configs = [
            {"dimensions": {"num_agents": 10, "num_items": 5, "num_features": 2, "num_simuls": 1}, "subproblem": {"name": "Greedy"}},
            {"dimensions": {"num_agents": 20, "num_items": 10, "num_features": 3, "num_simuls": 1}, "subproblem": {"name": "Greedy"}},
            {"dimensions": {"num_agents": 50, "num_items": 15, "num_features": 4, "num_simuls": 1}, "subproblem": {"name": "Greedy"}},
        ]
        
        for i, config in enumerate(configs):
            start_time = time.time()
            bc.load_config(config)
            end_time = time.time()
            print(f"Config {i+1}: {end_time - start_time:.4f}s")
        
        print("💡 User confusion: No guidance on efficient configuration")
        print("💡 Solution needed: Configuration best practices")
        
        # Test 2: Memory implications
        print("\n2. PERFORMANCE: Memory implications")
        print("-" * 40)
        
        # Test with different data sizes
        sizes = [(10, 5), (50, 10), (100, 20)]
        
        for agents, items in sizes:
            try:
                bc = BundleChoice()
                bc.load_config({
                    "dimensions": {"num_agents": agents, "num_items": items, "num_features": 2, "num_simuls": 1},
                    "subproblem": {"name": "Greedy"},
                    "ellipsoid": {"num_iters": 5, "verbose": False}
                })
                
                # Generate data
                agent_features = np.random.normal(0, 1, (agents, 2))
                obs_bundles = np.random.choice([0, 1], size=(agents, items), p=[0.6, 0.4])
                errors = np.random.normal(0, 0.1, size=(agents, items))
                
                input_data = {
                    "agent_data": {"features": agent_features},
                    "obs_bundle": obs_bundles,
                    "errors": errors
                }
                
                bc.data.load_and_scatter(input_data)
                bc.features.set_oracle(simple_features_oracle)
                bc.subproblems.load()
                
                theta_hat = bc.ellipsoid.solve()
                print(f"✅ {agents}x{items}: {theta_hat.shape}")
                
            except Exception as e:
                print(f"❌ {agents}x{items}: {e}")
        
        print("💡 User confusion: No guidance on scalability")
        print("💡 Solution needed: Performance guidelines")

def test_mpi_specific_issues():
    """Test MPI-specific user experience issues."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("\n" + "="*60)
        print("DEEP ANALYSIS: MPI-SPECIFIC ISSUES")
        print("="*60)
        
        # Test 1: MPI rank confusion
        print("\n1. MPI: Rank-specific behavior")
        print("-" * 40)
        
        print(f"Current rank: {rank}")
        print(f"Total ranks: {comm.Get_size()}")
        
        # Test 2: Data distribution confusion
        print("\n2. MPI: Data distribution")
        print("-" * 40)
        
        bc = BundleChoice()
        bc.load_config({
            "dimensions": {"num_agents": 20, "num_items": 5, "num_features": 2, "num_simuls": 1},
            "subproblem": {"name": "Greedy"},
            "ellipsoid": {"num_iters": 5, "verbose": False}
        })
        
        if rank == 0:
            # Only rank 0 has data
            agent_features = np.random.normal(0, 1, (20, 2))
            obs_bundles = np.random.choice([0, 1], size=(20, 5), p=[0.6, 0.4])
            errors = np.random.normal(0, 0.1, size=(20, 5))
            
            input_data = {
                "agent_data": {"features": agent_features},
                "obs_bundle": obs_bundles,
                "errors": errors
            }
        else:
            input_data = None
        
        bc.data.load_and_scatter(input_data)
        bc.features.set_oracle(simple_features_oracle)
        bc.subproblems.load()
        
        theta_hat = bc.ellipsoid.solve()
        
        if rank == 0:
            print(f"✅ Result: {theta_hat}")
            print("💡 User confusion: MPI behavior not obvious")
            print("💡 Solution needed: MPI documentation and examples")

def test_user_mental_model():
    """Test user mental model and expectations."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("\n" + "="*60)
        print("DEEP ANALYSIS: USER MENTAL MODEL")
        print("="*60)
        
        print("\n1. MENTAL MODEL: What users expect")
        print("-" * 40)
        
        expectations = [
            "I should be able to create a model and solve it",
            "I should get clear error messages when something is wrong",
            "I should be able to modify parameters easily",
            "I should be able to experiment with different solvers",
            "I should get feedback on what's happening",
            "I should be able to reuse components",
            "I should be able to validate my setup",
            "I should get performance guidance",
        ]
        
        for i, expectation in enumerate(expectations, 1):
            print(f"{i}. {expectation}")
        
        print("\n2. MENTAL MODEL: Current reality")
        print("-" * 40)
        
        reality = [
            "❌ Users must remember complex initialization order",
            "❌ Users get cryptic error messages",
            "❌ Users must know internal component names",
            "❌ Users must create complex config dictionaries",
            "❌ Users get no validation until solve time",
            "❌ Users must understand MPI behavior",
            "❌ Users get no performance guidance",
            "❌ Users must read source code to understand usage",
        ]
        
        for reality_item in reality:
            print(reality_item)
        
        print("\n3. MENTAL MODEL: Proposed improvements")
        print("-" * 40)
        
        improvements = [
            "✅ Clear step-by-step workflow",
            "✅ Self-documenting code",
            "✅ Validation at each step",
            "✅ Better error messages",
            "✅ Multiple ways to accomplish tasks",
            "✅ Sensible defaults",
            "✅ Performance guidance",
            "✅ Comprehensive documentation",
        ]
        
        for improvement in improvements:
            print(improvement)

if __name__ == "__main__":
    test_error_scenarios()
    test_workflow_variations()
    test_configuration_complexity()
    test_performance_implications()
    test_mpi_specific_issues()
    test_user_mental_model() 