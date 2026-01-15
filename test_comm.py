"""Test comm_manager patterns."""
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(f"=== Rank {rank}/{size} ===")

# Test 1: scatter_from_root_array pattern (multi-D with dtype/shape broadcast)
print("\n--- Test 1: Scatter 2D array ---")
counts = [2, 3, 5]  # rows per rank
if rank == 0:
    data = np.arange(50).reshape(10, 5)  # (10 rows, 5 cols)
    meta = (data.dtype, data.shape[1:])
else:
    data = None
    meta = None

dtype, tail_dims = comm.bcast(meta, root=0)
print(f"Rank {rank}: dtype={dtype}, tail_dims={tail_dims}")

elem_counts = [c * int(np.prod(tail_dims)) for c in counts]
if rank == 0:
    sendbuf = (data.ravel(), elem_counts)
else:
    sendbuf = None

recvbuf = np.empty(elem_counts[rank], dtype=dtype)
comm.Scatterv(sendbuf, recvbuf, root=0)
recvbuf = recvbuf.reshape((counts[rank],) + tail_dims)
print(f"Rank {rank}: received shape={recvbuf.shape}, data=\n{recvbuf}")

# Test 2: gather_at_root_array pattern
print("\n--- Test 2: Gather back ---")
local_flat = recvbuf.ravel()
local_size = np.array([local_flat.size], dtype=np.int64)
all_sizes = np.empty(size, dtype=np.int64) if rank == 0 else None
comm.Gather(local_size, all_sizes, root=0)

if rank == 0:
    gathered = np.empty(all_sizes.sum(), dtype=recvbuf.dtype)
    comm.Gatherv(local_flat, (gathered, all_sizes), root=0)
    gathered = gathered.reshape((-1,) + recvbuf.shape[1:])
    print(f"Root gathered shape={gathered.shape}")
    print(f"Data matches original: {np.array_equal(gathered, np.arange(50).reshape(10,5))}")
else:
    comm.Gatherv(local_flat, None, root=0)

# Test 3: broadcast_dict pattern
print("\n--- Test 3: Broadcast dict ---")
if rank == 0:
    data_dict = {"arr": np.array([1.0, 2.0, 3.0]), "scalar": 42}
    meta = {k: (v.shape, v.dtype) if isinstance(v, np.ndarray) else None for k, v in data_dict.items()}
else:
    data_dict = {}
    meta = None

meta = comm.bcast(meta, root=0)
print(f"Rank {rank}: meta={meta}")

result = {}
for key, info in meta.items():
    if info:
        arr = data_dict.get(key) if rank == 0 else np.empty(info[0], dtype=info[1])
        comm.Bcast(arr, root=0)
        result[key] = arr
    else:
        result[key] = comm.bcast(data_dict.get(key) if rank == 0 else None, root=0)

print(f"Rank {rank}: result={result}")
