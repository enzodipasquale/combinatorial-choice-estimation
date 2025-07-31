import numpy as np
import pytest
from typing import cast
from bundlechoice.feature_manager import FeatureManager
from bundlechoice.config import DimensionsConfig
from mpi4py import MPI

class DummyDataManager:
    def __init__(self, num_agents, num_simuls):
        self.agent_data = {"dummy": np.array([[1, 2], [3, 4]])}
        self.item_data = {"dummy": np.array([0])}
        self.input_data = {
            'agent_data': {"dummy": np.array([[1, 2], [3, 4]])},
            'item_data': {"dummy": np.array([0])}
        }
        self.local_data = {
            "agent_data": {"dummy": np.array([[1, 2], [3, 4]])},
            "item_data": {"dummy": np.array([0])},
            "errors": np.array([[0, 0], [0, 0]]),
            "obs_bundle": None
        }
        self.num_local_agents = 2

def dummy_get_x_k(i, B, data):
    # Simple feature: sum of bundle times agent index
    return np.array([i * np.sum(B)])

def test_get_agents_0():
    num_agents = 3
    num_simuls = 2
    dimensions_cfg = DimensionsConfig(
        num_agents=num_agents,
        num_items=2,
        num_features=1,
        num_simuls=num_simuls
    )
    data_manager = DummyDataManager(num_agents, num_simuls)
    features = FeatureManager(
        dimensions_cfg=dimensions_cfg,
        comm=MPI.COMM_WORLD,
        data_manager=data_manager
    )
    features.load(dummy_get_x_k)
    B_i_j = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
    x_i_k = features.get_agents_0(B_i_j)
    # Should be shape (3, 1)
    assert x_i_k is not None
    assert x_i_k.shape == (3, 1)
    # Check values
    expected = np.array([[0], [7], [22]])  # 0*3, 1*7, 2*11
    assert np.allclose(x_i_k, expected)

def test_get_all_0():
    num_agents = 2
    num_simuls = 2
    dimensions_cfg = DimensionsConfig(
        num_agents=num_agents,
        num_items=2,
        num_features=1,
        num_simuls=num_simuls
    )
    data_manager = DummyDataManager(num_agents, num_simuls)
    features = FeatureManager(
        dimensions_cfg=dimensions_cfg,
        comm=MPI.COMM_WORLD,
        data_manager=data_manager
    )
    features.load(dummy_get_x_k)
    B_si_j = [np.array([1, 1]), np.array([2, 2]), np.array([3, 3]), np.array([4, 4])]
    x_si_k = features.get_all_0(B_si_j)
    if x_si_k is None:
        pytest.fail('x_si_k is None, expected a numpy array')
    assert x_si_k.shape == (4, 1)
    expected = np.array([[0], [4], [0], [8]])
    assert np.allclose(x_si_k, expected)

def test_get_all_simulated_agent_features_vs_parallel():
    num_agents = 2
    num_simuls = 2
    dimensions_cfg = DimensionsConfig(
        num_agents=num_agents,
        num_items=2,
        num_features=1,
        num_simuls=num_simuls
    )
    data_manager = DummyDataManager(num_agents, num_simuls)
    features = FeatureManager(
        dimensions_cfg=dimensions_cfg,
        comm=MPI.COMM_WORLD,
        data_manager=data_manager
    )
    features.load(dummy_get_x_k)
    # Simulate B_si_j and B_local for local agents
    B_si_j = [np.array([1, 1]), np.array([2, 2]), np.array([3, 3]), np.array([4, 4])]
    num_local_agents = getattr(data_manager, 'num_local_agents', 2)
    B_local = B_si_j[:num_local_agents]
    x_si_k = features.get_all_0(B_si_j)
    x_si_k_MPI = features.get_all_distributed(B_local)
    # Only compare on rank 0
    if features.rank == 0:
        if x_si_k is not None and x_si_k_MPI is not None:
            assert np.allclose(x_si_k[:num_local_agents], x_si_k_MPI) 