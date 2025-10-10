import numpy as np
from mpi4py import MPI
import pytest
from bundlechoice.data_manager import DataManager
from bundlechoice.config import DimensionsConfig
from bundlechoice.comm_manager import CommManager


def test_data_manager_scatter_mpi():
    """Test MPI scatter functionality of DataManager."""
    dimensions_cfg = DimensionsConfig(
        num_agents=40,
        num_items=3,
        num_features=1,
        num_simuls=1
    )
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    comm_manager = CommManager(comm)
    
    if rank == 0:
        input_data = {
            'item_data': {'a': np.array([1, 2, 3])},
            'agent_data': {'b': np.random.normal(0, 1, (40, 3))},
            'errors': np.zeros((40, 3)),
            'observed_bundles': np.ones((40, 3)),
        }
    else:
        input_data = None

    dm = DataManager(
        dimensions_cfg=dimensions_cfg,
        comm_manager=comm_manager
    )
    
    # Load and scatter data from rank 0
    dm.load_and_scatter(input_data)

    # Verify local data is available on all ranks
    assert dm.local_data is not None
    assert dm.local_data["agent_data"] is not None
    assert dm.local_data["errors"] is not None
 
   