import numpy as np
import pytest
from bundlechoice.oracles_manager import OraclesManager
from bundlechoice.config import DimensionsConfig
from mpi4py import MPI
from bundlechoice.comm_manager import CommManager
from bundlechoice.data_manager import DataManager

def test_features_oracle():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    dimensions_cfg = DimensionsConfig(num_obs=10, num_items=3, num_features=2, num_simulations=1)
    comm_manager = CommManager(comm)
    data_manager = DataManager(dimensions_cfg, comm_manager)
    input_data = {'agent_data': {'modular': np.random.randn(10, 3, 2)}, 'item_data': {}} if rank == 0 else {'agent_data': {}, 'item_data': {}}
    data_manager.load_input_data(input_data)
    oracles = OraclesManager(dimensions_cfg, comm_manager, data_manager)
    def simple_features(bundles, local_id, data):
        return np.sum(bundles, axis=-1, keepdims=True).astype(float)
    oracles.set_features_oracle(simple_features)
    bundles = np.array([[True, False, True], [False, True, True]])
    result = oracles.features_oracle(bundles, local_id=np.array([0, 1]))
    assert result.shape == (2, 1)

def test_error_oracle():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    dimensions_cfg = DimensionsConfig(num_obs=10, num_items=3, num_features=2, num_simulations=1)
    comm_manager = CommManager(comm)
    data_manager = DataManager(dimensions_cfg, comm_manager)
    input_data = {'agent_data': {}, 'item_data': {}} if rank == 0 else {'agent_data': {}, 'item_data': {}}
    data_manager.load_input_data(input_data)
    oracles = OraclesManager(dimensions_cfg, comm_manager, data_manager)
    oracles.build_local_modular_error_oracle(seed=42)
    assert oracles._modular_local_errors is not None
    assert oracles._modular_local_errors.shape == (data_manager.num_local_agent, 3)

def test_utilities_oracle():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    dimensions_cfg = DimensionsConfig(num_obs=10, num_items=3, num_features=2, num_simulations=1)
    comm_manager = CommManager(comm)
    data_manager = DataManager(dimensions_cfg, comm_manager)
    input_data = {'agent_data': {}, 'item_data': {}} if rank == 0 else {'agent_data': {}, 'item_data': {}}
    data_manager.load_input_data(input_data)
    oracles = OraclesManager(dimensions_cfg, comm_manager, data_manager)
    def simple_features(bundles, local_id, data):
        return np.ones((len(local_id), 2))
    oracles.set_features_oracle(simple_features)
    oracles.build_local_modular_error_oracle(seed=42)
    theta = np.array([1.0, 2.0])
    bundles = np.array([[True, False, True]])
    result = oracles.features_oracle(bundles, local_id=np.array([0])) @ theta
    assert result is not None
