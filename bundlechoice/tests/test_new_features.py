"""
Tests for new features: Feature Naming, Data Loading, Bounds by Name, Result Export.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from bundlechoice.config import DimensionsConfig, RowGenerationConfig, BoundsManager
from bundlechoice.estimation.result import EstimationResult


class TestFeatureNaming:
    """Tests for DimensionsConfig feature naming."""
    
    def test_set_feature_names(self):
        dims = DimensionsConfig(num_features=3)
        dims.set_feature_names(["alpha", "beta", "gamma"])
        assert dims.feature_names == ["alpha", "beta", "gamma"]
    
    def test_get_feature_name(self):
        dims = DimensionsConfig()
        dims.feature_names = ["a", "b", "c"]
        assert dims.get_feature_name(0) == "a"
        assert dims.get_feature_name(1) == "b"
        assert dims.get_feature_name(99) == "theta_99"  # fallback
    
    def test_get_feature_index(self):
        dims = DimensionsConfig()
        dims.feature_names = ["pop", "distance", "travel"]
        assert dims.get_feature_index("distance") == 1
        with pytest.raises(KeyError):
            dims.get_feature_index("nonexistent")
    
    def test_get_indices_by_pattern(self):
        dims = DimensionsConfig()
        dims.feature_names = ["modular", "FE_0", "FE_1", "FE_2", "singular"]
        assert dims.get_indices_by_pattern("FE_*") == [1, 2, 3]
        assert dims.get_indices_by_pattern("*ular") == [0, 4]
    
    def test_set_feature_groups(self):
        dims = DimensionsConfig()
        dims.set_feature_groups(
            modular=["bidder_pop"],
            fixed_effects=3,  # auto-generates FE_0, FE_1, FE_2
            quadratic=["pop_dist", "travel", "air"]
        )
        assert dims.num_features == 7
        assert dims.feature_names == ["bidder_pop", "FE_0", "FE_1", "FE_2", "pop_dist", "travel", "air"]
        assert dims.get_group_indices("modular") == [0]
        assert dims.get_group_indices("fixed_effects") == [1, 2, 3]
        assert dims.get_group_indices("quadratic") == [4, 5, 6]
    
    def test_get_structural_indices(self):
        dims = DimensionsConfig()
        dims.set_feature_groups(
            modular=["m1", "m2"],
            fixed_effects=5,
            quadratic=["q1"]
        )
        structural = dims.get_structural_indices()
        assert structural == [0, 1, 7]  # excludes FE indices [2,3,4,5,6]


class TestBoundsManager:
    """Tests for BoundsManager."""
    
    def test_set_by_name(self):
        dims = DimensionsConfig()
        dims.set_feature_names(["pop", "dist", "travel"])
        bounds = BoundsManager(dims)
        bounds.set("pop", lower=75)
        bounds.set("dist", lower=400, upper=650)
        lbs, ubs = bounds.get_arrays(3, default_lower=0, default_upper=1000)
        assert lbs[0] == 75
        assert lbs[1] == 400
        assert ubs[1] == 650
        assert lbs[2] == 0  # default
    
    def test_set_by_index(self):
        dims = DimensionsConfig(num_features=3)
        bounds = BoundsManager(dims)
        bounds.set(0, lower=10).set(2, upper=500)
        lbs, ubs = bounds.get_arrays(3)
        assert lbs[0] == 10
        assert ubs[2] == 500
    
    def test_set_pattern(self):
        dims = DimensionsConfig()
        dims.set_feature_names(["m", "FE_0", "FE_1", "FE_2", "q"])
        bounds = BoundsManager(dims)
        bounds.set_pattern("FE_*", lower=0, upper=100)
        lbs, ubs = bounds.get_arrays(5, default_lower=-999, default_upper=999)
        assert lbs[1] == lbs[2] == lbs[3] == 0
        assert ubs[1] == ubs[2] == ubs[3] == 100
        assert lbs[0] == -999  # unaffected
        assert ubs[4] == 999   # unaffected


class TestResultExport:
    """Tests for EstimationResult export."""
    
    def test_export_csv_basic(self):
        result = EstimationResult(
            theta_hat=np.array([1.0, 2.0, 3.0]),
            converged=True,
            num_iterations=10,
            final_objective=0.001,
        )
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            path = Path(f.name)
        result.export_csv(path)
        content = path.read_text()
        assert "converged" in content
        assert "True" in content
        path.unlink()
    
    def test_export_csv_with_metadata(self):
        result = EstimationResult(
            theta_hat=np.array([10.0, 20.0]),
            converged=True,
            num_iterations=5,
        )
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            path = Path(f.name)
        result.export_csv(path, metadata={"delta": 4, "sample": "winners"})
        content = path.read_text()
        assert "delta" in content
        assert "4" in content
        assert "sample" in content
        path.unlink()
    
    def test_export_csv_with_feature_names(self):
        result = EstimationResult(
            theta_hat=np.array([100.0, 50.0]),
            converged=True,
            num_iterations=1,
        )
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            path = Path(f.name)
        result.export_csv(path, feature_names=["pop", "dist"])
        header = path.read_text().split('\n')[0]
        assert "theta_pop" in header
        assert "theta_dist" in header
        path.unlink()
    
    def test_export_csv_append(self):
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            path = Path(f.name)
        
        # First write
        r1 = EstimationResult(theta_hat=np.array([1.0]), converged=True, num_iterations=1)
        r1.export_csv(path, metadata={"run": 1})
        
        # Second write (append)
        r2 = EstimationResult(theta_hat=np.array([2.0]), converged=True, num_iterations=2)
        r2.export_csv(path, metadata={"run": 2}, append=True)
        
        lines = path.read_text().strip().split('\n')
        assert len(lines) == 3  # header + 2 rows
        path.unlink()
    
    def test_save_npy(self):
        result = EstimationResult(
            theta_hat=np.array([1.0, 2.0, 3.0]),
            converged=True,
            num_iterations=1,
        )
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            path = Path(f.name)
        result.save_npy(path)
        loaded = np.load(path)
        np.testing.assert_array_equal(loaded, result.theta_hat)
        path.unlink()
