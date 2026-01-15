import numpy as np
from mpi4py import MPI
import pytest
from bundlechoice.data_manager import DataManager
from bundlechoice.config import DimensionsConfig
from bundlechoice.comm_manager import CommManager

def test_data_manager_scatter_mpi():
    dimensions_cfg = DimensionsConfig(num_obs=40, num_items=3, num_features=1, num_simulations=1)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    comm_manager = CommManager(comm)
    input_data = {'item_data': {'a': np.array([1, 2, 3])}, 'agent_data': {'b': np.random.normal(0, 1, (40, 3))}} if rank == 0 else {'item_data': {}, 'agent_data': {}}
    dm = DataManager(dimensions_cfg=dimensions_cfg, comm_manager=comm_manager)
    dm.load_input_data(input_data)
    assert dm.local_data is not None
    assert dm.local_data['agent_data'] is not None
    assert 'b' in dm.local_data['agent_data']
    assert dm.local_data['item_data']['a'] is not None
