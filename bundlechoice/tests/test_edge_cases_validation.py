"""
Edge case and validation testing for BundleChoice.

Tests boundary conditions, error handling, and validation logic.

Run with: mpirun -n 2 python -m pytest bundlechoice/tests/test_edge_cases_validation.py -v
"""

import numpy as np
import pytest
from mpi4py import MPI
from bundlechoice import BundleChoice
from bundlechoice.validation import (
    DimensionValidator, DataQualityValidator, SetupValidator,
    validate_input_data_comprehensive
)
from bundlechoice.errors import (
    DimensionMismatchError, DataError, ValidationError, SetupError
)
from bundlechoice.config import DimensionsConfig

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


class TestDimensionValidation:
    """Test dimension validation edge cases."""
    
    def test_zero_agents_rejected(self):
        """Test that zero agents is rejected."""
        if rank == 0:
            with pytest.raises(ValueError):
                bc = BundleChoice()
                bc.load_config({
                    "dimensions": {
                        "num_agents": 0,
                        "num_items": 5,
                        "num_features": 2
                    }
                })
    
    def test_negative_items_rejected(self):
        """Test that negative items is rejected."""
        if rank == 0:
            with pytest.raises(ValueError):
                bc = BundleChoice()
                bc.load_config({
                    "dimensions": {
                        "num_agents": 10,
                        "num_items": -5,
                        "num_features": 2
                    }
                })
    
    def test_mismatched_agent_data(self):
        """Test detection of mismatched agent data dimensions."""
        if rank == 0:
            dimensions = DimensionsConfig(num_agents=10, num_items=5, num_features=2)
            
            wrong_data = {
                "modular": np.random.randn(15, 5, 2)  # 15 != 10
            }
            
            with pytest.raises(DimensionMismatchError):
                DimensionValidator.validate_agent_data(wrong_data, 10, 5)
    
    def test_mismatched_error_shape_2d(self):
        """Test 2D error array validation."""
        if rank == 0:
            dimensions = DimensionsConfig(num_agents=10, num_items=5, num_features=2, num_simuls=1)
            
            # Wrong shape
            wrong_errors = np.random.randn(10, 8)  # 8 != 5 items
            
            with pytest.raises(DimensionMismatchError):
                DimensionValidator.validate_errors(wrong_errors, 10, 5, 1)
    
    def test_mismatched_error_shape_3d(self):
        """Test 3D error array validation."""
        if rank == 0:
            dimensions = DimensionsConfig(num_agents=10, num_items=5, num_features=2, num_simuls=3)
            
            # Wrong number of simulations
            wrong_errors = np.random.randn(2, 10, 5)  # 2 != 3 simuls
            
            with pytest.raises(DimensionMismatchError):
                DimensionValidator.validate_errors(wrong_errors, 10, 5, 3)
    
    def test_obs_bundle_wrong_shape(self):
        """Test observed bundle shape validation."""
        if rank == 0:
            wrong_obs = np.random.randint(0, 2, (10, 8))  # 8 != 5 items
            
            with pytest.raises(DimensionMismatchError):
                DimensionValidator.validate_obs_bundles(wrong_obs, 10, 5)


class TestDataQualityValidation:
    """Test data quality validation."""
    
    def test_nan_detection_in_agent_data(self):
        """Test that NaN in agent data is detected."""
        if rank == 0:
            data_with_nan = {
                "agent_data": {
                    "modular": np.array([[1.0, 2.0], [np.nan, 4.0]])
                }
            }
            
            problems = DataQualityValidator.check_for_invalid_values(data_with_nan)
            assert len(problems) > 0
            assert any("nan" in p.lower() for p in problems)
    
    def test_inf_detection_in_errors(self):
        """Test that Inf in errors is detected."""
        if rank == 0:
            data_with_inf = {
                "errors": np.array([[1.0, np.inf], [3.0, 4.0]])
            }
            
            problems = DataQualityValidator.check_for_invalid_values(data_with_inf)
            assert len(problems) > 0
            assert any("inf" in p.lower() for p in problems)
    
    def test_quadratic_negative_values(self):
        """Test that negative quadratic values are caught."""
        if rank == 0:
            item_data = {
                "quadratic": np.array([
                    [[0, 1], [-1, 0]],  # Negative value
                    [[1, 0], [0, 1]]
                ])
            }
            
            with pytest.raises(ValidationError) as exc_info:
                DataQualityValidator.validate_quadratic_features(item_data)
            
            assert "negative" in str(exc_info.value).lower()
    
    def test_quadratic_nonzero_diagonal(self):
        """Test that non-zero diagonal in quadratic is caught."""
        if rank == 0:
            item_data = {
                "quadratic": np.array([
                    [[1, 0], [0, 0]],  # Non-zero diagonal
                    [[0, 1], [1, 0]]
                ])
            }
            
            with pytest.raises(ValidationError) as exc_info:
                DataQualityValidator.validate_quadratic_features(item_data)
            
            assert "diagonal" in str(exc_info.value).lower()


class TestSetupValidation:
    """Test setup validation and workflow order."""
    
    def test_check_component_config(self):
        """Test component checking for config."""
        bc = BundleChoice()
        assert not SetupValidator.check_component(bc, 'config')
        
        bc.load_config({
            "dimensions": {"num_agents": 10, "num_items": 5, "num_features": 2}
        })
        assert SetupValidator.check_component(bc, 'config')
    
    def test_check_component_data(self):
        """Test component checking for data."""
        bc = BundleChoice()
        bc.load_config({
            "dimensions": {"num_agents": 10, "num_items": 5, "num_features": 2}
        })
        
        assert not SetupValidator.check_component(bc, 'data')
        
        if rank == 0:
            input_data = {
                "agent_data": {"modular": np.random.randn(10, 5, 2)},
                "errors": np.random.randn(10, 5)
            }
        else:
            input_data = None
        
        bc.data.load_and_scatter(input_data)
        assert SetupValidator.check_component(bc, 'data')
    
    def test_validate_for_operation_missing_steps(self):
        """Test operation validation catches missing steps."""
        bc = BundleChoice()
        bc.load_config({
            "dimensions": {"num_agents": 10, "num_items": 5, "num_features": 2},
            "subproblem": {"name": "Greedy"}
        })
        
        # Missing data
        with pytest.raises(SetupError) as exc_info:
            SetupValidator.validate_for_operation(bc, 'solve_row_generation')
        
        error_msg = str(exc_info.value)
        assert "data" in error_msg.lower()
    
    def test_get_completed_steps(self):
        """Test getting list of completed steps."""
        bc = BundleChoice()
        completed = SetupValidator.get_completed_steps(bc)
        assert len(completed) == 0
        
        bc.load_config({
            "dimensions": {"num_agents": 10, "num_items": 5, "num_features": 2}
        })
        completed = SetupValidator.get_completed_steps(bc)
        assert 'config' in completed
        assert 'data' not in completed


class TestBoundaryConditions:
    """Test boundary conditions and extreme values."""
    
    def test_very_large_dimensions(self):
        """Test with large dimensions (should work, might be slow)."""
        if rank == 0:
            # Large but valid dimensions
            bc = BundleChoice()
            bc.load_config({
                "dimensions": {
                    "num_agents": 10000,
                    "num_items": 1000,
                    "num_features": 100
                }
            })
            # Should not raise
    
    def test_single_feature(self):
        """Test with single feature dimension."""
        if rank == 0:
            input_data = {
                "agent_data": {"modular": np.random.randn(10, 5, 1)},
                "errors": np.random.randn(10, 5)
            }
        else:
            input_data = None
        
        bc = BundleChoice()
        bc.load_config({
            "dimensions": {"num_agents": 10, "num_items": 5, "num_features": 1},
            "subproblem": {"name": "Greedy"}
        })
        
        bc.data.load_and_scatter(input_data)
        bc.features.build_from_data()
        
        theta = np.array([1.0])
        bundles = bc.subproblems.init_and_solve(theta)
        
        if rank == 0:
            assert bundles.shape == (10, 5)
    
    def test_very_small_tolerance(self):
        """Test with very small tolerance."""
        if rank == 0:
            bc = BundleChoice()
            bc.load_config({
                "dimensions": {"num_agents": 10, "num_items": 5, "num_features": 2},
                "row_generation": {
                    "tolerance_optimality": 1e-10,  # Very small
                    "max_iters": 5
                }
            })
            # Should not raise
    
    def test_extreme_theta_values(self):
        """Test with extreme theta values."""
        if rank == 0:
            input_data = {
                "agent_data": {"modular": np.random.randn(10, 5, 2)},
                "errors": np.random.randn(10, 5)
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
        
        # Very large positive theta
        theta_large = np.array([1000.0, 1000.0])
        bundles = bc.subproblems.init_and_solve(theta_large)
        
        if rank == 0:
            assert bundles is not None
            # Should still be valid binary bundles
            assert np.all((bundles == 0) | (bundles == 1))


class TestErrorMessageQuality:
    """Test that error messages are helpful and actionable."""
    
    def test_setup_error_has_suggestion(self):
        """Test that setup errors include suggestions."""
        bc = BundleChoice()
        bc.load_config({
            "dimensions": {"num_agents": 10, "num_items": 5, "num_features": 2},
            "subproblem": {"name": "Greedy"}
        })
        
        # Try to access subproblems without data
        with pytest.raises(SetupError) as exc_info:
            bc.subproblems.load()
        
        error_msg = str(exc_info.value)
        # Should have suggestion
        assert "suggestion" in error_msg.lower() or "💡" in error_msg
        # Should mention what to do
        assert "load_and_scatter" in error_msg or "data" in error_msg.lower()
    
    def test_dimension_error_shows_expected_vs_actual(self):
        """Test that dimension errors show expected vs actual values."""
        if rank == 0:
            dimensions = DimensionsConfig(num_agents=10, num_items=5, num_features=2)
            
            wrong_data = {
                "modular": np.random.randn(15, 5, 2)
            }
            
            with pytest.raises(DimensionMismatchError) as exc_info:
                DimensionValidator.validate_agent_data(wrong_data, 10, 5)
            
            error_msg = str(exc_info.value)
            assert "expected" in error_msg.lower()
            assert "10" in error_msg  # Expected value
            assert "15" in error_msg  # Actual value
    
    def test_validation_error_has_context(self):
        """Test that validation errors include context."""
        if rank == 0:
            item_data = {
                "quadratic": np.random.randn(5, 5, 2)  # Will have negatives
            }
            
            try:
                DataQualityValidator.validate_quadratic_features(item_data)
            except ValidationError as e:
                error_msg = str(e)
                # Should have context about the problem
                assert "quadratic" in error_msg.lower()


class TestConfigValidation:
    """Test configuration validation."""
    
    def test_negative_tolerance_rejected(self):
        """Test that negative tolerance is rejected."""
        if rank == 0:
            with pytest.raises(ValueError):
                bc = BundleChoice()
                bc.load_config({
                    "dimensions": {"num_agents": 10, "num_items": 5, "num_features": 2},
                    "row_generation": {
                        "tolerance_optimality": -0.001  # Negative!
                    }
                })
    
    def test_invalid_decay_factor_rejected(self):
        """Test that invalid decay factor is rejected."""
        if rank == 0:
            with pytest.raises(ValueError):
                bc = BundleChoice()
                bc.load_config({
                    "dimensions": {"num_agents": 10, "num_items": 5, "num_features": 2},
                    "ellipsoid": {
                        "decay_factor": 1.5  # Must be between 0 and 1
                    }
                })
    
    def test_zero_max_iterations_rejected(self):
        """Test that zero max iterations is rejected."""
        if rank == 0:
            with pytest.raises(ValueError):
                bc = BundleChoice()
                bc.load_config({
                    "dimensions": {"num_agents": 10, "num_items": 5, "num_features": 2},
                    "row_generation": {
                        "max_iters": 0  # Must be positive
                    }
                })


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

