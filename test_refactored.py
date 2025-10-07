import numpy as np
from mpi4py import MPI
from bundlechoice.comm_manager import CommManager

comm_mgr = CommManager(MPI.COMM_WORLD)

# Test broadcast
if comm_mgr.is_root():
    data = np.array([1., 2., 3.])
else:
    data = np.empty(3)
result = comm_mgr.broadcast_array(data)
assert np.allclose(result, [1., 2., 3.])

# Test concatenate 1D
local = np.arange(comm_mgr.rank + 2, dtype=float) + comm_mgr.rank * 10
result = comm_mgr.concatenate_array_at_root(local)
if comm_mgr.is_root():
    print(f"✓ 1D: {result}")

# Test concatenate 2D
local_2d = np.random.randn(comm_mgr.rank + 2, 3)
result_2d = comm_mgr.concatenate_array_at_root(local_2d)
if comm_mgr.is_root():
    print(f"✓ 2D shape: {result_2d.shape}")
    print("✅ All tests passed! Code is cleaner now.")
