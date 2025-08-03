import numpy as np
from mpi4py import MPI
import pytest
from bundlechoice.data_manager import DataManager
from bundlechoice.config import DimensionsConfig
from bundlechoice.comm_manager import CommManager


def test_data_manager_basic():
    """Test basic DataManager creation and attribute access."""
    dimensions_cfg = DimensionsConfig(
        num_agents=40,
        num_items=3,
        num_features=1,
        num_simuls=1
    )
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    comm_manager = CommManager(comm)

    # Create data manager without scatter
    dm = DataManager(dimensions_cfg=dimensions_cfg, comm_manager=comm_manager)
    
    # Check basic attributes
    assert dm.dimensions_cfg == dimensions_cfg
    assert dm.comm_manager == comm_manager
    assert dm.comm_manager.rank == rank
    assert dm.num_agents == 40
    assert dm.num_items == 3
    assert dm.num_features == 1
    assert dm.num_simuls == 1


def test_data_manager_load_only():
    """Test loading data without scatter operation."""
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
        'observed_bundles': np.ones((40, 3)),
    }
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    comm_manager = CommManager(comm)

    dm = DataManager(dimensions_cfg=dimensions_cfg, comm_manager=comm_manager)
    
    # Only rank 0 loads data
    if rank == 0:
        dm.load(input_data)
    
    # All ranks should have the same basic attributes
    assert dm.num_agents == 40
    assert dm.num_items == 3 