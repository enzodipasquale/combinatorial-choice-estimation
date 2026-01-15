from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
S = 2
counts = np.array([2, 2, 3])
if rank == 0:
    data = np.arange(3)
else:
    data = None

recvbuf = np.empty(counts[rank], dtype = np.int64)
comm.Scatterv([data, counts, MPI.INT64_T], recvbuf, root=0)

print(f"Rank {rank}: {recvbuf.reshape((counts[rank], 5))}")

# Gatherv example
# local = np.arange(rank * 10, rank * 10 + (rank + 1) * 2)  # rank 0: 2 elems, rank 1: 4 elems, rank 2: 6 elems
# print(f"Rank {rank} local: {local}")

# # Gather sizes
# local_size = np.array([len(local)], dtype=np.int64)
# all_sizes = np.empty(3, dtype=np.int64) if rank == 0 else None
# comm.Gather(local_size, all_sizes, root=0)

# # Gatherv
# if rank == 0:
#     recvbuf = np.empty(sum(all_sizes), dtype=np.int64)
#     comm.Gatherv(local, (recvbuf, all_sizes), root=0)
#     print(f"Root gathered: {recvbuf}")
# else:
#     comm.Gatherv(local, None, root=0)