import numpy as np
import pytest
from typing import cast
from bundlechoice.feature_manager import FeatureManager
from bundlechoice.config import DimensionsConfig
from mpi4py import MPI
from bundlechoice.comm_manager import CommManager

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

def test_compute_rank_features():
    num_agents = 30
    num_simuls = 2
    dimensions_cfg = DimensionsConfig(
        num_agents=num_agents,
        num_items=2,
        num_features=1,
        num_simuls=num_simuls
    )
    data_manager = DummyDataManager(num_agents, num_simuls)
    comm_manager = CommManager(MPI.COMM_WORLD)
    features = FeatureManager(
        dimensions_cfg=dimensions_cfg,
        comm_manager=comm_manager,
        data_manager=data_manager
    )
    features.set_oracle(dummy_get_x_k)
    # Create bundles for local agents
    local_bundles = [np.array([1, 2]) for _ in range(data_manager.num_local_agents)]
    x_i_k = features.compute_rank_features(local_bundles)
    # Should be shape (num_local_agents, 1)
    assert x_i_k is not None
    assert x_i_k.shape == (data_manager.num_local_agents, 1)
    # Check first few values
    expected_first = np.array([[0], [3]])  # 0*3, 1*3 for 2 local agents
    assert np.allclose(x_i_k[:2], expected_first)

def test_compute_gathered_features():
    num_agents = 2  # 2 agents per rank
    num_simuls = 2
    dimensions_cfg = DimensionsConfig(
        num_agents=num_agents,
        num_items=2,
        num_features=1,
        num_simuls=num_simuls
    )
    data_manager = DummyDataManager(num_agents, num_simuls)
    comm_manager = CommManager(MPI.COMM_WORLD)
    features = FeatureManager(
        dimensions_cfg=dimensions_cfg,
        comm_manager=comm_manager,
        data_manager=data_manager
    )
    features.set_oracle(dummy_get_x_k)
    # Create bundles for local agents
    local_bundles = [np.array([1, 1]) for _ in range(data_manager.num_local_agents)]
    x_si_k = features.compute_gathered_features(local_bundles)
    # Should be shape (num_global_agents, 1) on rank 0, None on other ranks
    if features.comm_manager.is_root():
        assert x_si_k is not None
        # With 10 ranks, each with 2 agents, total is 20 agents
        expected_total_agents = comm_manager.size * num_agents
        assert x_si_k.shape == (expected_total_agents, 1)
        # Check first few values
        expected_first = np.array([[0], [2]])  # 0*2, 1*2 for 2 agents
        assert np.allclose(x_si_k[:2], expected_first)
    else:
        assert x_si_k is None

def test_compute_gathered_features_consistency():
    num_agents = 2  # 2 agents per rank
    num_simuls = 2
    dimensions_cfg = DimensionsConfig(
        num_agents=num_agents,
        num_items=2,
        num_features=1,
        num_simuls=num_simuls
    )
    data_manager = DummyDataManager(num_agents, num_simuls)
    comm_manager = CommManager(MPI.COMM_WORLD)
    features = FeatureManager(
        dimensions_cfg=dimensions_cfg,
        comm_manager=comm_manager,
        data_manager=data_manager
    )
    features.set_oracle(dummy_get_x_k)
    # Create bundles for local agents
    local_bundles = [np.array([1, 1]) for _ in range(data_manager.num_local_agents)]
    x_si_k_MPI = features.compute_gathered_features(local_bundles)
    # Test that the gathered features are consistent
    if features.comm_manager.is_root():
        assert x_si_k_MPI is not None
        # With 10 ranks, each with 2 agents, total is 20 agents
        expected_total_agents = comm_manager.size * num_agents
        assert x_si_k_MPI.shape == (expected_total_agents, 1)
        # Check that the first local agent features match what we expect
        expected_first = np.array([[0], [2]])  # 0*2, 1*2 for 2 local agents
        assert np.allclose(x_si_k_MPI[:2], expected_first)
    else:
        assert x_si_k_MPI is None 