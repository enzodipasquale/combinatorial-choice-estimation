import numpy as np
import pytest
from mpi4py import MPI
from bundlechoice.comm_manager import CommManager

@pytest.mark.mpi
@pytest.mark.unit
def test_broadcast_array_with_flag():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    comm_manager = CommManager(comm)
    if rank == 0:
        array = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        flag = True
    else:
        array = np.empty(5, dtype=np.float64)
        flag = False
    result_array, result_flag = comm_manager.broadcast_array_with_flag(array, flag, root=0)
    if rank == 0:
        assert np.allclose(result_array, [1.0, 2.0, 3.0, 4.0, 5.0]), 'Root: array mismatch'
        assert result_flag == True, 'Root: flag mismatch'
    else:
        assert np.allclose(result_array, [1.0, 2.0, 3.0, 4.0, 5.0]), f'Rank {rank}: array mismatch'
        assert result_flag == True, f'Rank {rank}: flag mismatch'
    array_size = 1000
    if rank == 0:
        array = np.arange(array_size, dtype=np.float64) * 1.5
        flag = False
    else:
        array = np.empty(array_size, dtype=np.float64)
        flag = False
    result_array, result_flag = comm_manager.broadcast_array_with_flag(array, flag, root=0)
    if rank == 0:
        expected = array
    else:
        expected = np.arange(array_size, dtype=np.float64) * 1.5
    assert np.allclose(result_array, expected), f'Rank {rank}: large array mismatch'
    assert result_flag == False, f'Rank {rank}: flag should be False'
    comm._barrier()
    if rank == 0:
        print('✓ broadcast_array_with_flag tests passed')

def test__gather_array_by_row():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    comm_manager = CommManager(comm)
    local_size = 10 + rank
    local_array = np.arange(local_size, dtype=np.float64) + rank * 100
    result = comm_manager._gather_array_by_row(local_array, root=0)
    if rank == 0:
        expected = np.concatenate([np.arange(10 + r, dtype=np.float64) + r * 100 for r in range(size)])
        assert result is not None, 'Result should not be None on root'
        assert np.allclose(result, expected), '1D float64 concatenation mismatch'
    else:
        assert result is None, f'Rank {rank}: Result should be None on non-root'
    local_first_dim = 5 + rank
    local_array = np.arange(local_first_dim * 3, dtype=np.float64).reshape(local_first_dim, 3) + rank * 1000
    result = comm_manager._gather_array_by_row(local_array, root=0)
    if rank == 0:
        assert result is not None, 'Result should not be None on root'
        assert result.shape[1] == 3, 'Second dimension should be preserved'
        total_first = sum((5 + r for r in range(size)))
        assert result.shape[0] == total_first, f'First dimension mismatch: {result.shape[0]} != {total_first}'
        expected = np.concatenate([np.arange((5 + r) * 3, dtype=np.float64).reshape(5 + r, 3) + r * 1000 for r in range(size)])
        assert np.allclose(result, expected), '2D array data mismatch'
    else:
        assert result is None, f'Rank {rank}: Result should be None on non-root'
    local_size = 8 + rank
    local_array = np.arange(local_size, dtype=np.int32) % 2 == 0
    result = comm_manager._gather_array_by_row(local_array, root=0)
    if rank == 0:
        assert result is not None, 'Result should not be None on root'
        assert result.dtype == np.bool_, 'Result should be bool dtype'
        expected = np.concatenate([np.arange(8 + r, dtype=np.int32) % 2 == 0 for r in range(size)])
        assert np.array_equal(result, expected), 'Bool array data mismatch'
        total_size = sum((8 + r for r in range(size)))
        assert result.shape == (total_size,), f'Bool array shape mismatch: {result.shape} != ({total_size},)'
    else:
        assert result is None, f'Rank {rank}: Result should be None on non-root'
    comm._barrier()
    if rank == 0:
        print('✓ concatenate_array_at_root_fast tests passed')

def test_edge_cases():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    comm_manager = CommManager(comm)
    if rank == 0:
        array = np.array([], dtype=np.float64)
        flag = True
    else:
        array = np.empty(0, dtype=np.float64)
        flag = False
    result_array, result_flag = comm_manager.broadcast_array_with_flag(array, flag, root=0)
    assert result_array.shape == (0,), f'Rank {rank}: Empty array shape mismatch'
    assert result_flag == True, f'Rank {rank}: Flag mismatch'
    if rank == 0:
        array = np.array([42.0], dtype=np.float64)
        flag = False
    else:
        array = np.empty(1, dtype=np.float64)
        flag = False
    result_array, result_flag = comm_manager.broadcast_array_with_flag(array, flag, root=0)
    assert result_array[0] == 42.0, f'Rank {rank}: Single element mismatch'
    assert result_flag == False, f'Rank {rank}: Flag mismatch'
    local_array = np.array([], dtype=np.float64)
    result = comm_manager._gather_array_by_row(local_array, root=0)
    if rank == 0:
        assert result is not None, 'Result should not be None on root'
        assert result.shape == (0,), 'Empty concatenation should produce empty array'
    else:
        assert result is None, f'Rank {rank}: Result should be None on non-root'
    comm._barrier()
    if rank == 0:
        print('✓ Edge case tests passed')
if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        print('=' * 70)
        print('MPI Communication Optimization Tests')
        print('=' * 70)
        print()
    try:
        test_broadcast_array_with_flag()
        test__gather_array_by_row()
        test_edge_cases()
        comm._barrier()
        if rank == 0:
            print()
            print('=' * 70)
            print('All tests passed!')
            print('=' * 70)
    except Exception as e:
        print(f'Rank {rank} failed with error: {e}')
        import traceback
        traceback.print_exc()
        comm.Abort(1)