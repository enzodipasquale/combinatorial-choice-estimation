import pytest
import numpy as np
import tempfile
from pathlib import Path
from bundlechoice.config import DimensionsConfig, RowGenerationConfig
from bundlechoice.estimation.result import EstimationResult

class TestFeatureNaming:

    def test_set_feature_names(self):
        dims = DimensionsConfig(num_features=3)
        dims.set_feature_names(['alpha', 'beta', 'gamma'])
        assert dims.feature_names == ['alpha', 'beta', 'gamma']

    def test_get_feature_name(self):
        dims = DimensionsConfig()
        dims.feature_names = ['a', 'b', 'c']
        assert dims.get_feature_name(0) == 'a'
        assert dims.get_feature_name(1) == 'b'
        assert dims.get_feature_name(99) == 'theta_99'

    def test_get_index_by_name(self):
        dims = DimensionsConfig(num_features=7)
        dims.feature_names = ['m1', 'm2', 'FE_0', 'FE_1', 'FE_2', 'FE_3', 'q1']
        structural = dims.get_index_by_name()
        assert structural == [0, 1, 6]

    def test_get_index_by_name_no_names(self):
        dims = DimensionsConfig(num_features=5)
        structural = dims.get_index_by_name()
        assert structural == [0, 1, 2, 3, 4]

class TestResultExport:

    def test_export_csv_basic(self):
        result = EstimationResult(theta_hat=np.array([1.0, 2.0, 3.0]), converged=True, num_iterations=10, final_objective=0.001)
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            path = Path(f.name)
        result.export_csv(path)
        content = path.read_text()
        assert 'converged' in content
        assert 'True' in content
        path.unlink()

    def test_export_csv_with_metadata(self):
        result = EstimationResult(theta_hat=np.array([10.0, 20.0]), converged=True, num_iterations=5)
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            path = Path(f.name)
        result.export_csv(path, metadata={'delta': 4, 'sample': 'winners'})
        content = path.read_text()
        assert 'delta' in content
        assert '4' in content
        assert 'sample' in content
        path.unlink()

    def test_export_csv_with_feature_names(self):
        result = EstimationResult(theta_hat=np.array([100.0, 50.0]), converged=True, num_iterations=1)
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            path = Path(f.name)
        result.export_csv(path, feature_names=['pop', 'dist'])
        header = path.read_text().split('\n')[0]
        assert 'theta_pop' in header
        assert 'theta_dist' in header
        path.unlink()

    def test_export_csv_append(self):
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            path = Path(f.name)
        r1 = EstimationResult(theta_hat=np.array([1.0]), converged=True, num_iterations=1)
        r1.export_csv(path, metadata={'run': 1})
        r2 = EstimationResult(theta_hat=np.array([2.0]), converged=True, num_iterations=2)
        r2.export_csv(path, metadata={'run': 2}, append=True)
        lines = path.read_text().strip().split('\n')
        assert len(lines) == 3
        path.unlink()

    def test_save_npy(self):
        result = EstimationResult(theta_hat=np.array([1.0, 2.0, 3.0]), converged=True, num_iterations=1)
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            path = Path(f.name)
        result.save_npy(path)
        loaded = np.load(path)
        np.testing.assert_array_equal(loaded, result.theta_hat)
        path.unlink()