import numpy as np
import pytest
from typing import cast
from bundlechoice.oracles_manager import OraclesManager
from bundlechoice.config import DimensionsConfig
from mpi4py import MPI
from bundlechoice.comm_manager import CommManager


class DummyDataManager:
    """Mock data manager for testing feature computation."""
    
    def __init__(self, num_agents, num_simulations):
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
            "observed_bundles": None
        }
        self.num_local_agents = 2


def dummy_get_x_k(i, B, data):
    """Simple feature: sum of bundle times agent index."""
    return np.array([i * np.sum(B)])


def test_compute_rank_features():
    """Test feature computation for local rank."""
    num_agents = 30
    num_simulations = 2
    dimensions_cfg = DimensionsConfig(
        num_agents=num_agents,
        num_items=2,
        num_features=1,
        num_simulations=num_simulations
    )
    data_manager = DummyDataManager(num_agents, num_simulations)
    comm_manager = CommManager(MPI.COMM_WORLD)
    features = OraclesManager(
        dimensions_cfg=dimensions_cfg,
        comm_manager=comm_manager,
        data_manager=data_manager
    )
    features.set_features_oracle(dummy_get_x_k)
    
    # Create bundles for local agents (as numpy array)
    local_bundles = np.array([[1, 2] for _ in range(data_manager.num_local_agents)], dtype=np.float64)
    x_i_k = features.compute_rank_features(local_bundles)
    
    # Verify shape and values
    assert x_i_k is not None
    assert x_i_k.shape == (data_manager.num_local_agents, 1)
    
    # Check first few values: 0*3, 1*3 for 2 local agents
    expected_first = np.array([[0], [3]])
    assert np.allclose(x_i_k[:2], expected_first)


def test_compute_gathered_features():
    """Test feature computation with MPI gathering."""
    num_agents = 2  # 2 agents per rank
    num_simulations = 2
    dimensions_cfg = DimensionsConfig(
        num_agents=num_agents,
        num_items=2,
        num_features=1,
        num_simulations=num_simulations
    )
    data_manager = DummyDataManager(num_agents, num_simulations)
    comm_manager = CommManager(MPI.COMM_WORLD)
    features = OraclesManager(
        dimensions_cfg=dimensions_cfg,
        comm_manager=comm_manager,
        data_manager=data_manager
    )
    features.set_features_oracle(dummy_get_x_k)
    
    # Create bundles for local agents (as numpy array)
    local_bundles = np.array([[1, 1] for _ in range(data_manager.num_local_agents)], dtype=np.float64)
    x_si_k = features.compute_gathered_features(local_bundles)
    
    # Verify results on root rank
    if features.comm_manager.is_root():
        assert x_si_k is not None
        expected_total_agents = comm_manager.size * num_agents
        assert x_si_k.shape == (expected_total_agents, 1)
        
        # Check first few values: 0*2, 1*2 for 2 agents
        expected_first = np.array([[0], [2]])
        assert np.allclose(x_si_k[:2], expected_first)
    else:
        assert x_si_k is None


def test_compute_gathered_features_consistency():
    """Test consistency of gathered features across MPI ranks."""
    num_agents = 2  # 2 agents per rank
    num_simulations = 2
    dimensions_cfg = DimensionsConfig(
        num_agents=num_agents,
        num_items=2,
        num_features=1,
        num_simulations=num_simulations
    )
    data_manager = DummyDataManager(num_agents, num_simulations)
    comm_manager = CommManager(MPI.COMM_WORLD)
    features = OraclesManager(
        dimensions_cfg=dimensions_cfg,
        comm_manager=comm_manager,
        data_manager=data_manager
    )
    features.set_features_oracle(dummy_get_x_k)
    
    # Create bundles for local agents
    local_bundles = [np.array([1, 1]) for _ in range(data_manager.num_local_agents)]
    x_si_k_mpi = features.compute_gathered_features(local_bundles)
    
    # Verify consistency on root rank
    if features.comm_manager.is_root():
        assert x_si_k_mpi is not None
        expected_total_agents = comm_manager.size * num_agents
        assert x_si_k_mpi.shape == (expected_total_agents, 1)
        
        # Check that the first local agent features match expected values
        expected_first = np.array([[0], [2]])  # 0*2, 1*2 for 2 local agents
        assert np.allclose(x_si_k_mpi[:2], expected_first)
    else:
        assert x_si_k_mpi is None 