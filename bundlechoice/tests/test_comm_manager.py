import numpy as np
import pytest
from mpi4py import MPI
from bundlechoice.comm_manager import CommManager

def test_comm_manager_init():
    comm = MPI.COMM_WORLD
    cm = CommManager(comm)
    assert cm.rank == comm.Get_rank()
    assert cm.comm_size == comm.Get_size()
    assert cm.root == 0
    assert cm._is_root() == (cm.rank == 0)

def test_comm_manager_barrier():
    comm = MPI.COMM_WORLD
    cm = CommManager(comm)
    cm._barrier()

def test_comm_managerbcast():
    comm = MPI.COMM_WORLD
    cm = CommManager(comm)
    if cm._is_root():
        data = "test_data"
    else:
        data = None
    result = cm.bcast(data)
    assert result == "test_data"

def test_comm_managerBcast():
    comm = MPI.COMM_WORLD
    cm = CommManager(comm)
    if cm._is_root():
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    else:
        arr = np.empty(3, dtype=np.float64)
    result = cm.Bcast(arr)
    np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))

def test_comm_managerscatter():
    comm = MPI.COMM_WORLD
    cm = CommManager(comm)
    if cm._is_root():
        data = [i for i in range(cm.comm_size)]
    else:
        data = None
    result = cm.scatter(data)
    assert result == cm.rank

def test_comm_managergather():
    comm = MPI.COMM_WORLD
    cm = CommManager(comm)
    data = cm.rank
    result = cm.gather(data)
    if cm._is_root():
        assert result == list(range(cm.comm_size))
    else:
        assert result is None

def test_comm_managerScatterv_by_row():
    comm = MPI.COMM_WORLD
    cm = CommManager(comm)
    from bundlechoice.config import DimensionsConfig
    from bundlechoice.data_manager import DataManager
    dc = DimensionsConfig(num_obs=10, num_items=5, num_features=3, num_simulations=1)
    dm = DataManager(dc, cm)
    num_obs = 10
    num_items = 5
    if cm._is_root():
        send_array = np.random.randn(num_obs, num_items)
        agent_counts = dm.agent_counts
    else:
        send_array = None
        agent_counts = dm.agent_counts
    result = cm.Scatterv_by_row(send_array, row_counts=agent_counts)
    assert result.shape[1] == num_items
    assert result.dtype == send_array.dtype if cm._is_root() else result.dtype == np.float64

def test_comm_managerGatherv_by_row():
    comm = MPI.COMM_WORLD
    cm = CommManager(comm)
    num_items = 5
    local_array = np.random.randn(3, num_items)
    result = cm.Gatherv_by_row(local_array)
    if cm._is_root():
        assert result is not None
        assert result.shape[1] == num_items
    else:
        assert result is None

def test_comm_managerReduce():
    comm = MPI.COMM_WORLD
    cm = CommManager(comm)
    arr = np.array([1.0, 2.0, 3.0])
    result = cm.Reduce(arr, op=MPI.SUM)
    if cm._is_root():
        expected = arr * cm.comm_size
        np.testing.assert_array_almost_equal(result, expected)
    else:
        assert result is not None

def test_comm_manager_sum_row_andReduce():
    comm = MPI.COMM_WORLD
    cm = CommManager(comm)
    arr = np.random.randn(5, 3)
    result = cm.sum_row_andReduce(arr)
    if cm._is_root():
        assert result.shape == (3,)
        expected = arr.sum(0) * cm.comm_size
        np.testing.assert_array_almost_equal(result, expected)
    else:
        assert result is not None

def test_comm_manager_get_dict_metadata():
    comm = MPI.COMM_WORLD
    cm = CommManager(comm)
    if cm._is_root():
        data_dict = {'a': np.array([1, 2, 3]), 'b': 'string', 'c': np.array([[1, 2], [3, 4]])}
    else:
        data_dict = {}
    meta = cm.get_dict_metadata(data_dict if cm._is_root() else {})
    assert 'a' in meta
    assert meta['a'][0] == 'arr'
    assert meta['b'][0] == 'obj'

def test_comm_managerscatter_dict():
    comm = MPI.COMM_WORLD
    cm = CommManager(comm)
    from bundlechoice.config import DimensionsConfig
    from bundlechoice.data_manager import DataManager
    dc = DimensionsConfig(num_obs=10, num_items=5, num_features=3, num_simulations=1)
    dm = DataManager(dc, cm)
    num_obs = 10
    num_items = 5
    if cm._is_root():
        data_dict = {
            'modular': np.random.randn(num_obs, num_items, 2),
            'obs_bundles': np.random.randn(num_obs, num_items) > 0.5,
            'value': 42
        }
        agent_counts = dm.agent_counts
    else:
        data_dict = {}
        agent_counts = dm.agent_counts
    result = cm.scatter_dict(data_dict, agent_counts=agent_counts)
    assert 'modular' in result
    assert 'value' in result
    assert result['value'] == 42

def test_comm_managerbcast_dict():
    comm = MPI.COMM_WORLD
    cm = CommManager(comm)
    num_items = 5
    if cm._is_root():
        data_dict = {
            'modular': np.random.randn(num_items, 2),
            'value': 42
        }
    else:
        data_dict = {}
    result = cm.bcast_dict(data_dict if cm._is_root() else {})
    assert 'modular' in result
    assert result['value'] == 42
    if not cm._is_root():
        assert result['modular'].shape == (num_items, 2)
