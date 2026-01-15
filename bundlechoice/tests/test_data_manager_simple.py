import numpy as np
from mpi4py import MPI
import pytest
from bundlechoice.data_manager import DataManager
from bundlechoice.config import DimensionsConfig
from bundlechoice.comm_manager import CommManager

def test_data_manager_basic():
    dimensions_cfg = DimensionsConfig(num_obs=40, num_items=3, num_features=1, num_simulations=1)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    comm_manager = CommManager(comm)
    dm = DataManager(dimensions_cfg=dimensions_cfg, comm_manager=comm_manager)
    assert dm.dimensions_cfg == dimensions_cfg
    assert dm.comm_manager == comm_manager
    assert dm.comm_manager.rank == rank
    assert dm.dimensions_cfg.num_obs == 40
    assert dm.dimensions_cfg.num_items == 3
    assert dm.dimensions_cfg.num_features == 1
    assert dm.dimensions_cfg.num_simulations == 1

def test_data_manager_load_only():
    dimensions_cfg = DimensionsConfig(num_obs=40, num_items=3, num_features=1, num_simulations=1)
    input_data = {'item_data': {'a': np.array([1, 2, 3])}, 'agent_data': {'b': np.random.normal(0, 1, (40, 3))}, 'errors': np.zeros((40, 3)), 'observed_bundles': np.ones((40, 3))}
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    comm_manager = CommManager(comm)
    dm = DataManager(dimensions_cfg=dimensions_cfg, comm_manager=comm_manager)
    if rank == 0:
        dm.load(input_data)
    assert dm.dimensions_cfg.num_obs == 40
    assert dm.dimensions_cfg.num_items == 3