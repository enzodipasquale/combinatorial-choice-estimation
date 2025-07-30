import numpy as np
from mpi4py import MPI
import pytest
from bundlechoice.data_manager import DataManager
from bundlechoice.config import DimensionsConfig

def test_scatter_data_input_validation():
    # Test with None comm - should raise AttributeError
    with pytest.raises(AttributeError):
        dm = DataManager(dimensions_cfg=None, comm=None)
    
    # Create a proper dimensions config
    dimensions_cfg = DimensionsConfig(
        num_agents=2,
        num_items=3,
        num_features=1,
        num_simuls=1
    )
    
    # Test with valid comm and dimensions_cfg but missing input_data on rank 0
    dm = DataManager(dimensions_cfg=dimensions_cfg, comm=MPI.COMM_WORLD)
    with pytest.raises(ValueError):
        dm.load_and_scatter(None)

def test_data_manager_scatter_mpi():
    dimensions_cfg = DimensionsConfig(
        num_agents=2,
        num_items=3,
        num_features=1,
        num_simuls=1
    )
    input_data = {
        'item_data': {'a': np.array([1, 2, 3])},
        'agent_data': {'b': np.array([[1, 2, 3], [4, 5, 6]])},
        'errors': np.zeros((1, 2, 3)),
        'obs_bundle': np.ones((2, 3)),
    }
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    comm_size = comm.Get_size()
    # Only rank 0 loads the data, others get None
    input_data_rank = input_data if rank == 0 else None
    dm = DataManager(
        dimensions_cfg=dimensions_cfg,
        comm=comm
    )
    dm.load_and_scatter(input_data_rank)
    # All ranks should have these attributes set
    assert dm.local_data is not None
    assert dm.local_data["agent_data"] is not None
    assert dm.local_data["errors"] is not None
    # Each rank gets a chunk, so num_local_agents is either 1 or 1 (for 2 agents and 2 ranks)
    assert dm.num_local_agents in {1, 2} 