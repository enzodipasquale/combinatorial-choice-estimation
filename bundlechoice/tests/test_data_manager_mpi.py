import numpy as np
from mpi4py import MPI
import pytest
from bundlechoice.data_manager import DataManager
from bundlechoice.config import DimensionsConfig
from bundlechoice.comm_manager import CommManager


def test_data_manager_scatter_mpi():
    dimensions_cfg = DimensionsConfig(
        num_agents=40,
        num_items=3,
        num_features=1,
        num_simuls=1  # Simplify to single simulation
    )
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    comm_manager = CommManager(comm)
    if rank == 0:
        input_data = {
            'item_data': {'a': np.array([1, 2, 3])},
            'agent_data': {'b': np.random.normal(0, 1, (40, 3))},
            'errors': np.zeros((40, 3)),  # 2D errors for single simulation
            'obs_bundle': np.ones((40, 3)),
        }
    else:
        input_data = None

    
    dm = DataManager(
        dimensions_cfg=dimensions_cfg,
        comm_manager=comm_manager
    )
    # Only rank 0 loads and scatters the data
    # print(f"Rank {rank}: input_data = {input_data}")
    dm.load_and_scatter(input_data)

    assert dm.local_data is not None
    assert dm.local_data["agent_data"] is not None
    assert dm.local_data["errors"] is not None
 
   