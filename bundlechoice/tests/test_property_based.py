"""
Property-based testing for BundleChoice using Hypothesis.

This module tests invariants that should hold for all valid inputs,
helping catch edge cases and ensuring robust behavior.

Install hypothesis: pip install hypothesis
Run with: pytest bundlechoice/tests/test_property_based.py -v
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.numpy import arrays, array_shapes
from bundlechoice import BundleChoice
from bundlechoice.config import DimensionsConfig
from bundlechoice.data_manager import DataManager
from bundlechoice.feature_manager import FeatureManager
from bundlechoice.comm_manager import CommManager
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


# Custom strategies for BundleChoice data
@st.composite
def valid_dimensions(draw):
    """Strategy for generating valid dimension configurations."""
    num_agents = draw(st.integers(min_value=5, max_value=100))
    num_items = draw(st.integers(min_value=3, max_value=50))
    num_features = draw(st.integers(min_value=1, max_value=20))
    num_simuls = draw(st.integers(min_value=1, max_value=5))
    
    return DimensionsConfig(
        num_agents=num_agents,
        num_items=num_items,
        num_features=num_features,
        num_simuls=num_simuls
    )


@st.composite
def modular_input_data(draw, dimensions_cfg):
    """Strategy for generating valid modular input data."""
    num_agents = dimensions_cfg.num_agents
    num_items = dimensions_cfg.num_items
    num_features = dimensions_cfg.num_features
    num_simuls = dimensions_cfg.num_simuls
    
    # Generate agent modular data
    agent_modular = draw(arrays(
        dtype=np.float64,
        shape=(num_agents, num_items, num_features),
        elements=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)
    ))
    
    # Generate errors
    errors = draw(arrays(
        dtype=np.float64,
        shape=(num_simuls, num_agents, num_items),
        elements=st.floats(min_value=-5, max_value=5, allow_nan=False, allow_infinity=False)
    ))
    
    return {
        "agent_data": {"modular": agent_modular},
        "errors": errors
    }


@st.composite
def theta_vector(draw, num_features):
    """Strategy for generating valid parameter vectors."""
    return draw(arrays(
        dtype=np.float64,
        shape=(num_features,),
        elements=st.floats(min_value=-5, max_value=5, allow_nan=False, allow_infinity=False)
    ))


class TestDataManagerProperties:
    """Property-based tests for DataManager."""
    
    @given(dimensions=valid_dimensions())
    @settings(max_examples=20, deadline=None)
    def test_dimensions_always_match_config(self, dimensions):
        """Property: DataManager dimensions should always match config."""
        if rank == 0:
            comm_manager = CommManager(comm)
            dm = DataManager(dimensions, comm_manager)
            
            assert dm.num_agents == dimensions.num_agents
            assert dm.num_items == dimensions.num_items
            assert dm.num_features == dimensions.num_features
            assert dm.num_simuls == dimensions.num_simuls
    
    @given(dimensions=valid_dimensions())
    @settings(max_examples=20, deadline=None)
    def test_scatter_preserves_total_agents(self, dimensions):
        """Property: Scattering should preserve total number of agents across ranks."""
        comm_manager = CommManager(comm)
        dm = DataManager(dimensions, comm_manager)
        
        input_data = modular_input_data(dimensions).example()
        
        if rank == 0:
            dm.load(input_data)
        
        dm.scatter()
        
        # Gather local agent counts
        local_count = dm.num_local_agents
        total_counts = comm.gather(local_count, root=0)
        
        if rank == 0:
            total_agents = sum(total_counts)
            expected = dimensions.num_agents * dimensions.num_simuls
            assert total_agents == expected, \
                f"Total agents {total_agents} != expected {expected}"


class TestFeatureManagerProperties:
    """Property-based tests for FeatureManager."""
    
    @given(
        dimensions=valid_dimensions(),
    )
    @settings(max_examples=15, deadline=None)
    def test_features_output_correct_shape(self, dimensions):
        """Property: Feature oracle should always output correct shape."""
        comm_manager = CommManager(comm)
        dm = DataManager(dimensions, comm_manager)
        fm = FeatureManager(dimensions, comm_manager, dm)
        
        input_data = modular_input_data(dimensions).example()
        
        if rank == 0:
            dm.load(input_data)
            dm.scatter()
        else:
            dm.scatter()
        
        # Build auto-generated features
        fm.build_from_data()
        
        # Test with random bundle
        bundle = np.random.choice([0, 1], size=dimensions.num_items)
        features = fm.features_oracle(0, bundle, dm.local_data)
        
        assert features.shape == (dimensions.num_features,), \
            f"Features shape {features.shape} != ({dimensions.num_features},)"
    
    @given(dimensions=valid_dimensions())
    @settings(max_examples=15, deadline=None)
    def test_features_deterministic(self, dimensions):
        """Property: Same input should produce same features."""
        comm_manager = CommManager(comm)
        dm = DataManager(dimensions, comm_manager)
        fm = FeatureManager(dimensions, comm_manager, dm)
        
        input_data = modular_input_data(dimensions).example()
        
        if rank == 0:
            dm.load(input_data)
            dm.scatter()
        else:
            dm.scatter()
        
        fm.build_from_data()
        
        # Compute features twice with same input
        bundle = np.random.choice([0, 1], size=dimensions.num_items)
        features1 = fm.features_oracle(0, bundle, dm.local_data)
        features2 = fm.features_oracle(0, bundle, dm.local_data)
        
        assert np.allclose(features1, features2), \
            "Features are not deterministic"
    
    @given(dimensions=valid_dimensions())
    @settings(max_examples=10, deadline=None)
    def test_features_no_nan_inf(self, dimensions):
        """Property: Features should never contain NaN or Inf."""
        comm_manager = CommManager(comm)
        dm = DataManager(dimensions, comm_manager)
        fm = FeatureManager(dimensions, comm_manager, dm)
        
        input_data = modular_input_data(dimensions).example()
        
        if rank == 0:
            dm.load(input_data)
            dm.scatter()
        else:
            dm.scatter()
        
        fm.build_from_data()
        
        # Test multiple random bundles
        for _ in range(5):
            bundle = np.random.choice([0, 1], size=dimensions.num_items)
            features = fm.features_oracle(0, bundle, dm.local_data)
            
            assert not np.isnan(features).any(), "Features contain NaN"
            assert not np.isinf(features).any(), "Features contain Inf"


class TestConfigProperties:
    """Property-based tests for configuration validation."""
    
    @given(
        num_agents=st.integers(min_value=1, max_value=1000),
        num_items=st.integers(min_value=1, max_value=500),
        num_features=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=30)
    def test_valid_dimensions_never_raise(self, num_agents, num_items, num_features):
        """Property: Valid positive dimensions should never raise errors."""
        if rank == 0:
            cfg = DimensionsConfig(
                num_agents=num_agents,
                num_items=num_items,
                num_features=num_features,
                num_simuls=1
            )
            
            # Should not raise
            bc = BundleChoice()
            bc.load_config({"dimensions": {
                "num_agents": num_agents,
                "num_items": num_items,
                "num_features": num_features,
            }})
    
    @given(
        num_agents=st.integers(max_value=0),
    )
    @settings(max_examples=10)
    def test_invalid_dimensions_always_raise(self, num_agents):
        """Property: Invalid dimensions should always raise errors."""
        if rank == 0:
            with pytest.raises(ValueError):
                bc = BundleChoice()
                bc.load_config({
                    "dimensions": {
                        "num_agents": num_agents,
                        "num_items": 10,
                        "num_features": 5,
                    }
                })


class TestGreedySubproblemProperties:
    """Property-based tests for Greedy subproblem."""
    
    @given(
        dimensions=valid_dimensions(),
    )
    @settings(max_examples=10, deadline=None)
    def test_greedy_always_returns_valid_bundle(self, dimensions):
        """Property: Greedy should always return a valid binary bundle."""
        # Small problem for fast testing
        assume(dimensions.num_items <= 20)
        assume(dimensions.num_agents <= 50)
        
        bc = BundleChoice()
        bc.load_config({
            "dimensions": {
                "num_agents": dimensions.num_agents,
                "num_items": dimensions.num_items,
                "num_features": dimensions.num_features,
                "num_simuls": 1,
            },
            "subproblem": {"name": "Greedy"}
        })
        
        input_data = modular_input_data(dimensions).example()
        bc.data.load_and_scatter(input_data)
        bc.features.build_from_data()
        
        theta = theta_vector(dimensions.num_features).example()
        bundles = bc.subproblems.init_and_solve(theta)
        
        if rank == 0:
            assert bundles is not None
            assert bundles.shape == (dimensions.num_agents, dimensions.num_items)
            assert np.all((bundles == 0) | (bundles == 1)), "Bundle contains non-binary values"
    
    @given(dimensions=valid_dimensions())
    @settings(max_examples=10, deadline=None)
    def test_greedy_deterministic(self, dimensions):
        """Property: Greedy should be deterministic for same input."""
        assume(dimensions.num_items <= 15)
        assume(dimensions.num_agents <= 30)
        
        bc = BundleChoice()
        bc.load_config({
            "dimensions": {
                "num_agents": dimensions.num_agents,
                "num_items": dimensions.num_items,
                "num_features": dimensions.num_features,
                "num_simuls": 1,
            },
            "subproblem": {"name": "Greedy"}
        })
        
        input_data = modular_input_data(dimensions).example()
        bc.data.load_and_scatter(input_data)
        bc.features.build_from_data()
        
        theta = theta_vector(dimensions.num_features).example()
        
        # Solve twice
        bundles1 = bc.subproblems.init_and_solve(theta)
        bc.subproblem_manager.initialize_local()
        bundles2 = bc.subproblem_manager.solve_local(theta)
        bundles2_gathered = bc.comm_manager.concatenate_array_at_root_fast(bundles2, root=0)
        
        if rank == 0:
            assert np.array_equal(bundles1, bundles2_gathered), \
                "Greedy is not deterministic"


class TestValidationProperties:
    """Property-based tests for validation."""
    
    @given(dimensions=valid_dimensions())
    @settings(max_examples=20)
    def test_validation_accepts_valid_data(self, dimensions):
        """Property: Validation should accept all valid data."""
        if rank == 0:
            from bundlechoice.validation import validate_input_data_comprehensive
            
            input_data = modular_input_data(dimensions).example()
            
            # Should not raise
            validate_input_data_comprehensive(input_data, dimensions)
    
    @given(
        dimensions=valid_dimensions(),
        wrong_num_agents=st.integers(min_value=1, max_value=200)
    )
    @settings(max_examples=20)
    def test_validation_rejects_wrong_dimensions(self, dimensions, wrong_num_agents):
        """Property: Validation should reject data with wrong dimensions."""
        assume(wrong_num_agents != dimensions.num_agents)
        
        if rank == 0:
            from bundlechoice.validation import validate_input_data_comprehensive
            from bundlechoice.errors import DimensionMismatchError
            
            # Create data with wrong agent count
            input_data = {
                "agent_data": {
                    "modular": np.random.randn(wrong_num_agents, dimensions.num_items, dimensions.num_features)
                },
                "errors": np.random.randn(dimensions.num_simuls, wrong_num_agents, dimensions.num_items)
            }
            
            with pytest.raises(DimensionMismatchError):
                validate_input_data_comprehensive(input_data, dimensions)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])

