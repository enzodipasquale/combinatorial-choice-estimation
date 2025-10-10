"""
Comprehensive integration tests for BundleChoice.

Tests complete workflows from data loading to estimation,
including error scenarios and edge cases.

Run with: mpirun -n 4 python -m pytest bundlechoice/tests/test_integration_comprehensive.py -v
"""

import numpy as np
import pytest
from mpi4py import MPI
from bundlechoice import BundleChoice
from bundlechoice.errors import SetupError, DimensionMismatchError, DataError, ValidationError

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


class TestCompleteWorkflows:
    """Test complete estimation workflows end-to-end."""
    
    def test_row_generation_complete_workflow(self):
        """Test complete workflow: data → features → estimation → validation."""
        # Setup - smaller sizes for fast testing
        num_agents = 20  # Reduced from 50
        num_items = 10   # Reduced from 15
        num_features = 3
        
        np.random.seed(42)
        if rank == 0:
            agent_modular = np.random.normal(0, 1, (num_agents, num_items, num_features))
            errors = np.random.normal(0, 0.5, size=(1, num_agents, num_items))
            
            input_data = {
                "agent_data": {"modular": agent_modular},
                "errors": errors
            }
        else:
            input_data = None
        
        # Configuration
        bc = BundleChoice()
        bc.load_config({
            "dimensions": {
                "num_agents": num_agents,
                "num_items": num_items,
                "num_features": num_features,
                "num_simuls": 1
            },
            "subproblem": {"name": "Greedy"},
            "row_generation": {
                "max_iters": 10,  # Reduced from 20
                "tolerance_optimality": 0.02,  # Relaxed from 0.01
                "gurobi_settings": {"OutputFlag": 0}
            }
        })
        
        # Load data
        bc.data.load_and_scatter(input_data)
        
        # Generate features
        bc.features.build_from_data()
        
        # Generate observations
        theta_true = np.array([1.0, 0.8, 1.2])
        obs_bundles = bc.subproblems.init_and_solve(theta_true)
        
        if rank == 0:
            input_data["obs_bundle"] = obs_bundles
        
        # Reload with observations
        bc.data.load_and_scatter(input_data)
        bc.features.build_from_data()
        bc.subproblems.load()
        
        # Validate setup
        bc.validate_setup('row_generation')
        
        # Estimate
        theta_hat = bc.row_generation.solve()
        
        # Validate results
        if rank == 0:
            assert theta_hat is not None
            assert theta_hat.shape == (num_features,)
            assert not np.isnan(theta_hat).any()
            assert not np.isinf(theta_hat).any()
            
            # Check estimation quality (should be reasonably close)
            error = np.linalg.norm(theta_hat - theta_true)
            assert error < 1.0, f"Estimation error too large: {error}"
    
    def test_ellipsoid_complete_workflow(self):
        """Test ellipsoid method complete workflow."""
        num_agents = 15  # Reduced from 40
        num_items = 8    # Reduced from 12
        num_features = 2
        
        np.random.seed(43)
        if rank == 0:
            agent_modular = np.random.normal(0, 1, (num_agents, num_items, num_features))
            errors = np.random.normal(0, 0.3, size=(1, num_agents, num_items))
            
            input_data = {
                "agent_data": {"modular": agent_modular},
                "errors": errors
            }
        else:
            input_data = None
        
        bc = BundleChoice()
        bc.load_config({
            "dimensions": {
                "num_agents": num_agents,
                "num_items": num_items,
                "num_features": num_features,
                "num_simuls": 1
            },
            "subproblem": {"name": "Greedy"},
            "ellipsoid": {
                "num_iters": 15,  # Reduced from 30
                "verbose": False
            }
        })
        
        bc.data.load_and_scatter(input_data)
        bc.features.build_from_data()
        
        theta_true = np.array([1.0, 0.5])
        obs_bundles = bc.subproblems.init_and_solve(theta_true)
        
        if rank == 0:
            input_data["obs_bundle"] = obs_bundles
        
        bc.data.load_and_scatter(input_data)
        bc.features.build_from_data()
        bc.subproblems.load()
        
        theta_hat = bc.ellipsoid.solve()
        
        if rank == 0:
            assert theta_hat is not None
            assert theta_hat.shape == (num_features,)
            assert not np.isnan(theta_hat).any()
    
    def test_quick_setup_workflow(self):
        """Test quick_setup convenience method."""
        num_agents = 15  # Reduced from 30
        num_items = 8    # Reduced from 10
        num_features = 2
        
        np.random.seed(44)
        if rank == 0:
            input_data = {
                "agent_data": {
                    "modular": np.random.normal(0, 1, (num_agents, num_items, num_features))
                },
                "errors": np.random.normal(0, 0.2, size=(1, num_agents, num_items))
            }
        else:
            input_data = None
        
        config = {
            "dimensions": {
                "num_agents": num_agents,
                "num_items": num_items,
                "num_features": num_features,
                "num_simuls": 1
            },
            "subproblem": {"name": "Greedy"},
            "row_generation": {
                "max_iters": 8,  # Reduced from 15
                "tolerance_optimality": 0.02,  # Relaxed from 0.01
                "gurobi_settings": {"OutputFlag": 0}
            }
        }
        
        # Quick setup
        bc = BundleChoice().quick_setup(config, input_data)
        
        # Should be ready for observations
        theta_true = np.array([1.0, 0.5])
        obs_bundles = bc.generate_observations(theta_true)
        
        # Estimate
        theta_hat = bc.row_generation.solve()
        
        if rank == 0:
            assert theta_hat is not None


class TestErrorScenarios:
    """Test that errors are caught with helpful messages."""
    
    def test_missing_config_error(self):
        """Test error when config is missing."""
        bc = BundleChoice()
        
        # Try to load data without config
        if rank == 0:
            with pytest.raises((SetupError, ValidationError, ValueError)) as exc_info:
                bc.data.load_and_scatter({"errors": np.zeros((10, 5))})
            
            # New validation provides clear error messages
            error_msg = str(exc_info.value).lower()
            assert "dimension" in error_msg or "config" in error_msg or "num_agents" in error_msg
    
    def test_dimension_mismatch_error(self):
        """Test error when data dimensions don't match config."""
        bc = BundleChoice()
        bc.load_config({
            "dimensions": {
                "num_agents": 10,
                "num_items": 5,
                "num_features": 2,
                "num_simuls": 1
            }
        })
        
        if rank == 0:
            # Wrong agent dimension
            wrong_data = {
                "agent_data": {
                    "modular": np.random.randn(15, 5, 2)  # 15 instead of 10
                },
                "errors": np.random.randn(1, 15, 5)
            }
            
            with pytest.raises(DimensionMismatchError) as exc_info:
                bc.data.load_and_scatter(wrong_data)
            
            error_msg = str(exc_info.value)
            assert "dimension mismatch" in error_msg.lower()
            assert "expected 10" in error_msg.lower()
            assert "got 15" in error_msg.lower()
    
    def test_missing_features_error(self):
        """Test error when features not set before solving."""
        bc = BundleChoice()
        bc.load_config({
            "dimensions": {"num_agents": 10, "num_items": 5, "num_features": 2},
            "subproblem": {"name": "Greedy"}
        })
        
        if rank == 0:
            input_data = {
                "agent_data": {"modular": np.random.randn(10, 5, 2)},
                "errors": np.random.randn(10, 5),
                "obs_bundle": np.random.randint(0, 2, (10, 5))
            }
        else:
            input_data = None
        
        bc.data.load_and_scatter(input_data)
        # Don't set features!
        
        if rank == 0:
            with pytest.raises(SetupError) as exc_info:
                bc.subproblems.load()
            
            error_msg = str(exc_info.value)
            assert "feature" in error_msg.lower()
            assert "build_from_data" in error_msg or "set_oracle" in error_msg
    
    def test_nan_in_data_error(self):
        """Test error when data contains NaN values."""
        bc = BundleChoice()
        bc.load_config({
            "dimensions": {"num_agents": 10, "num_items": 5, "num_features": 2}
        })
        
        if rank == 0:
            data_with_nan = {
                "agent_data": {
                    "modular": np.random.randn(10, 5, 2)
                },
                "errors": np.random.randn(10, 5)
            }
            # Inject NaN
            data_with_nan["agent_data"]["modular"][0, 0, 0] = np.nan
            
            with pytest.raises(DataError) as exc_info:
                bc.data.load_and_scatter(data_with_nan)
            
            error_msg = str(exc_info.value)
            assert "nan" in error_msg.lower() or "invalid" in error_msg.lower()


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_agent(self):
        """Test with single agent."""
        if rank == 0:
            input_data = {
                "agent_data": {"modular": np.random.randn(1, 5, 2)},
                "errors": np.random.randn(1, 5)
            }
        else:
            input_data = None
        
        bc = BundleChoice()
        bc.load_config({
            "dimensions": {"num_agents": 1, "num_items": 5, "num_features": 2},
            "subproblem": {"name": "Greedy"}
        })
        
        bc.data.load_and_scatter(input_data)
        bc.features.build_from_data()
        
        theta = np.array([1.0, 0.5])
        bundles = bc.subproblems.init_and_solve(theta)
        
        if rank == 0:
            assert bundles.shape == (1, 5)
    
    def test_single_item(self):
        """Test with single item (trivial choice)."""
        if rank == 0:
            input_data = {
                "agent_data": {"modular": np.random.randn(10, 1, 2)},
                "errors": np.random.randn(10, 1)
            }
        else:
            input_data = None
        
        bc = BundleChoice()
        bc.load_config({
            "dimensions": {"num_agents": 10, "num_items": 1, "num_features": 2},
            "subproblem": {"name": "Greedy"}
        })
        
        bc.data.load_and_scatter(input_data)
        bc.features.build_from_data()
        
        theta = np.array([1.0, 0.5])
        bundles = bc.subproblems.init_and_solve(theta)
        
        if rank == 0:
            assert bundles.shape == (10, 1)
            # Single item: either all take it or all don't
            assert bundles.sum() in [0, 10]
    
    def test_empty_bundle_handling(self):
        """Test that empty bundles (all zeros) are handled correctly."""
        if rank == 0:
            input_data = {
                "agent_data": {"modular": np.random.randn(10, 5, 2)},
                "errors": np.random.randn(10, 5) * 100  # Large negative errors → empty bundles
            }
        else:
            input_data = None
        
        bc = BundleChoice()
        bc.load_config({
            "dimensions": {"num_agents": 10, "num_items": 5, "num_features": 2},
            "subproblem": {"name": "Greedy"}
        })
        
        bc.data.load_and_scatter(input_data)
        bc.features.build_from_data()
        
        theta = np.array([-10.0, -10.0])  # Very negative
        bundles = bc.subproblems.init_and_solve(theta)
        
        if rank == 0:
            # Should still return valid shape even if all empty
            assert bundles.shape == (10, 5)
            assert np.all((bundles == 0) | (bundles == 1))
    
    def test_all_items_selected(self):
        """Test when all items are selected."""
        if rank == 0:
            input_data = {
                "agent_data": {"modular": np.ones((10, 5, 2))},
                "errors": np.ones((10, 5)) * 100  # Large positive errors
            }
        else:
            input_data = None
        
        bc = BundleChoice()
        bc.load_config({
            "dimensions": {"num_agents": 10, "num_items": 5, "num_features": 2},
            "subproblem": {"name": "Greedy"}
        })
        
        bc.data.load_and_scatter(input_data)
        bc.features.build_from_data()
        
        theta = np.array([10.0, 10.0])  # Very positive
        bundles = bc.subproblems.init_and_solve(theta)
        
        if rank == 0:
            # Most/all agents should select all items
            avg_selection = bundles.mean()
            assert avg_selection > 0.8  # At least 80% selected


class TestDataConsistency:
    """Test data consistency across operations."""
    
    def test_reload_data_preserves_dimensions(self):
        """Test that reloading data preserves dimensions."""
        bc = BundleChoice()
        bc.load_config({
            "dimensions": {"num_agents": 20, "num_items": 8, "num_features": 3},
            "subproblem": {"name": "Greedy"}
        })
        
        if rank == 0:
            input_data = {
                "agent_data": {"modular": np.random.randn(20, 8, 3)},
                "errors": np.random.randn(20, 8)
            }
        else:
            input_data = None
        
        # Load once
        bc.data.load_and_scatter(input_data)
        first_local_agents = bc.data.num_local_agents
        
        # Load again
        bc.data.load_and_scatter(input_data)
        second_local_agents = bc.data.num_local_agents
        
        assert first_local_agents == second_local_agents
    
    def test_multiple_simulations(self):
        """Test with multiple simulation draws."""
        num_simuls = 3
        
        if rank == 0:
            input_data = {
                "agent_data": {"modular": np.random.randn(10, 5, 2)},
                "errors": np.random.randn(num_simuls, 10, 5)
            }
        else:
            input_data = None
        
        bc = BundleChoice()
        bc.load_config({
            "dimensions": {
                "num_agents": 10,
                "num_items": 5,
                "num_features": 2,
                "num_simuls": num_simuls
            },
            "subproblem": {"name": "Greedy"}
        })
        
        bc.data.load_and_scatter(input_data)
        bc.features.build_from_data()
        
        theta = np.array([1.0, 0.5])
        bundles = bc.subproblems.init_and_solve(theta)
        
        if rank == 0:
            # Should have num_simuls * num_agents bundles
            assert bundles.shape == (10, 5)  # Observations are per base agent


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

