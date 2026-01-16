from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


# def Reduce(array, op=MPI.SUM):
#     sendbuf = np.ascontiguousarray(array)
#     recvbuf = np.empty_like(sendbuf)
#     comm.Reduce(sendbuf, recvbuf, op=op, root=0)
#     return recvbuf

# def _sum_by_row_andReduce(array):
#     sendbuf = array.sum(0)
#     return Reduce(sendbuf)


# # Reducev example: each rank has different-sized arrays, sum them all
# local_array = np.arange((rank + 1) * 3)
# print(local_array.shape)
# # repeat the array 3 times
# local_array = np.tile(local_array[:, np.newaxis], (1, 4))  
# print(f"Rank {rank} local: {local_array}")

# # # Reduce sum
# # local_size = np.array([len(local_array)], dtype=local_array.dtype)
# # sendbuf = local_size.sum(0)
# # recvbuf = np.empty_like(sendbuf)
# # comm.Reduce(sendbuf, recvbuf, op=MPI.SUM, root=0)

# # print(f"Rank {rank} all sizes: {recvbuf}")

# result = _sum_by_row_andReduce(local_array)
# print(f"Rank {rank} result: {result}")





# Reducev: sum all arrays element-wise (variable sizes)
# S = 2
# counts = np.array([2, 2, 3])
# if rank == 0:
#     data = np.arange(3)
# else:
#     data = None

# recvbuf = np.empty(counts[rank], dtype = np.int64)
# comm.Scatterv([data, counts, MPI.INT64_T], recvbuf, root=0)

# print(f"Rank {rank}: {recvbuf.reshape((counts[rank], 5))}")

# Gatherv example
# local = np.arange(rank * 10, rank * 10 + (rank + 1) * 2)  # rank 0: 2 elems, rank 1: 4 elems, rank 2: 6 elems
# print(f"Rank {rank} local: {local}")

# # Gather sizes
# local_size = np.array([len(local)], dtype=np.int64)
# size = comm.Get_size()
# all_sizes = np.empty(size, dtype=np.int64) if rank == 0 else None
# comm.Gather(local_size, all_sizes, root=0)

# # Gatherv
# if rank == 0:
#     recvbuf = np.empty(sum(all_sizes), dtype=np.int64)
#     comm.Gatherv(local, (recvbuf, all_sizes), root=0)
#     print(f"Root gathered: {recvbuf}")
# else:
#     comm.Gatherv(local, None, root=0)


# a = np.empty(4, dtype=np.int64)
# a = a.reshape((2, 2))
# print(a)


print(np.tile(np.array([1, 2, 3]), 2))
