import numpy as np
from mpi4py import MPI
import pytest
from bundlechoice.data_manager import DataManager
from bundlechoice.config import DimensionsConfig

def test_data_manager_basic():
    """Simple test that just creates a DataManager and checks basic functionality."""
    dimensions_cfg = DimensionsConfig(
        num_agents=40,
        num_items=3,
        num_features=1,
        num_simuls=1
    )
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Just create the data manager - no scatter
    dm = DataManager(dimensions_cfg=dimensions_cfg, comm=comm)
    
    # Check basic attributes
    assert dm.dimensions_cfg == dimensions_cfg
    assert dm.comm == comm
    assert dm.rank == rank
    assert dm.num_agents == 40
    assert dm.num_items == 3
    assert dm.num_features == 1
    assert dm.num_simuls == 1
    
    print(f"Rank {rank}: DataManager created successfully")
    
    # Test that we can access basic properties without hanging
    if rank == 0:
        print(f"Rank {rank}: num_agents = {dm.num_agents}")
        print(f"Rank {rank}: num_items = {dm.num_items}")
    else:
        print(f"Rank {rank}: num_agents = {dm.num_agents}")
        print(f"Rank {rank}: num_items = {dm.num_items}")

def test_data_manager_load_only():
    """Test loading data without scatter."""
    dimensions_cfg = DimensionsConfig(
        num_agents=40,
        num_items=3,
        num_features=1,
        num_simuls=1
    )
    input_data = {
        'item_data': {'a': np.array([1, 2, 3])},
        'agent_data': {'b': np.random.normal(0, 1, (40, 3))},
        'errors': np.zeros((40, 3)),
        'obs_bundle': np.ones((40, 3)),
    }
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    dm = DataManager(dimensions_cfg=dimensions_cfg, comm=comm)
    
    # Only rank 0 loads data
    if rank == 0:
        dm.load(input_data)
        print(f"Rank {rank}: Data loaded successfully")
        print(f"Rank {rank}: input_data keys = {list(dm.input_data.keys())}")
    else:
        print(f"Rank {rank}: No data loaded (as expected)")
    
    # All ranks should have the same basic attributes
    assert dm.num_agents == 40
    assert dm.num_items == 3 