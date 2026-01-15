import numpy as np
import pytest
from typing import cast
from bundlechoice.oracles_manager import OraclesManager
from bundlechoice.config import DimensionsConfig
from mpi4py import MPI
from bundlechoice.comm_manager import CommManager

class DummyDataManager:

    def __init__(self, num_obs, num_simulations):
        self.agent_data = {'dummy': np.array([[1, 2], [3, 4]])}
        self.item_data = {'dummy': np.array([0])}
        self.input_data = {'agent_data': {'dummy': np.array([[1, 2], [3, 4]])}, 'item_data': {'dummy': np.array([0])}}
        self.local_data = {'agent_data': {'dummy': np.array([[1, 2], [3, 4]])}, 'item_data': {'dummy': np.array([0])}, 'errors': np.array([[0, 0], [0, 0]]), 'observed_bundles': None}
        self.num_local_agent = 2

def dummy_get_x_k(i, B, data):
    return np.array([i * np.sum(B)])

def test_compute_rank_features():
    num_obs = 30
    num_simulations = 2
    dimensions_cfg = DimensionsConfig(num_obs=num_obs, num_items=2, num_features=1, num_simulations=num_simulations)
    data_manager = DummyDataManager(num_obs, num_simulations)
    comm_manager = CommManager(MPI.COMM_WORLD)
    features = OraclesManager(dimensions_cfg=dimensions_cfg, comm_manager=comm_manager, data_manager=data_manager)
    features.set_features_oracle(dummy_get_x_k)
    local_bundles = np.array([[1, 2] for _ in range(data_manager.num_local_agent)], dtype=np.float64)
    x_i_k = features.compute_rank_features(local_bundles)
    assert x_i_k is not None
    assert x_i_k.shape == (data_manager.num_local_agent, 1)
    expected_first = np.array([[0], [3]])
    assert np.allclose(x_i_k[:2], expected_first)

def test_compute_gathered_features():
    num_obs = 2
    num_simulations = 2
    dimensions_cfg = DimensionsConfig(num_obs=num_obs, num_items=2, num_features=1, num_simulations=num_simulations)
    data_manager = DummyDataManager(num_obs, num_simulations)
    comm_manager = CommManager(MPI.COMM_WORLD)
    features = OraclesManager(dimensions_cfg=dimensions_cfg, comm_manager=comm_manager, data_manager=data_manager)
    features.set_features_oracle(dummy_get_x_k)
    local_bundles = np.array([[1, 1] for _ in range(data_manager.num_local_agent)], dtype=np.float64)
    x_si_k = features.compute_gathered_features(local_bundles)
    if features.comm_manager._is_root():
        assert x_si_k is not None
        expected_total_agents = comm_manager.comm_size * num_obs
        assert x_si_k.shape == (expected_total_agents, 1)
        expected_first = np.array([[0], [2]])
        assert np.allclose(x_si_k[:2], expected_first)
    else:
        assert x_si_k is None

def test_compute_gathered_features_consistency():
    num_obs = 2
    num_simulations = 2
    dimensions_cfg = DimensionsConfig(num_obs=num_obs, num_items=2, num_features=1, num_simulations=num_simulations)
    data_manager = DummyDataManager(num_obs, num_simulations)
    comm_manager = CommManager(MPI.COMM_WORLD)
    features = OraclesManager(dimensions_cfg=dimensions_cfg, comm_manager=comm_manager, data_manager=data_manager)
    features.set_features_oracle(dummy_get_x_k)
    local_bundles = [np.array([1, 1]) for _ in range(data_manager.num_local_agent)]
    x_si_k_mpi = features.compute_gathered_features(local_bundles)
    if features.comm_manager._is_root():
        assert x_si_k_mpi is not None
        expected_total_agents = comm_manager.comm_size * num_obs
        assert x_si_k_mpi.shape == (expected_total_agents, 1)
        expected_first = np.array([[0], [2]])
        assert np.allclose(x_si_k_mpi[:2], expected_first)
    else:
        assert x_si_k_mpi is None