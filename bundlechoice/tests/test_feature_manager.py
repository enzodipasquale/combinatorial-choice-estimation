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

def test_get_agents_0():
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
    features.load(dummy_get_x_k)
    # Create bundles for all 30 agents
    B_i_j = [np.array([1, 2]) for _ in range(30)]
    x_i_k = features.get_agents_0(B_i_j)
    # Should be shape (30, 1) on rank 0, None on other ranks
    if features.comm_manager.is_root():
        assert x_i_k is not None
        assert x_i_k.shape == (30, 1)
        # Check first few values
        expected_first = np.array([[0], [3], [6]])  # 0*3, 1*3, 2*3
        assert np.allclose(x_i_k[:3], expected_first)
    else:
        assert x_i_k is None

def test_get_all_0():
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
    features.load(dummy_get_x_k)
    # Create bundles for all 60 simulated agents (30 agents × 2 simuls)
    B_si_j = [np.array([1, 1]) for _ in range(60)]
    x_si_k = features.get_all_0(B_si_j)
    # Should be shape (60, 1) on rank 0, None on other ranks
    if features.comm_manager.is_root():
        assert x_si_k is not None
        assert x_si_k.shape == (60, 1)
        # Check first few values
        expected_first = np.array([[0], [2], [4]])  # 0*2, 1*2, 2*2
        assert np.allclose(x_si_k[:3], expected_first)
    else:
        assert x_si_k is None

def test_get_all_simulated_agent_features_vs_parallel():
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
    features.load(dummy_get_x_k)
    # Simulate B_si_j and B_local for local agents
    B_si_j = [np.array([1, 1]) for _ in range(60)]  # 60 simulated agents
    num_local_agents = getattr(data_manager, 'num_local_agents', 2)
    B_local = B_si_j[:num_local_agents]
    x_si_k = features.get_all_0(B_si_j)
    x_si_k_MPI = features.get_all_distributed(B_local)
    # Only compare on rank 0
    if features.comm_manager.rank == 0:
        if x_si_k is not None and x_si_k_MPI is not None:
            # The distributed version returns results for all agents, not just local ones
            # So we need to compare the local portion
            assert np.allclose(x_si_k[:num_local_agents], x_si_k_MPI[:num_local_agents]) 