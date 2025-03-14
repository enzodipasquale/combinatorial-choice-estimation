import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


local_pricing = np.zeros((5 if rank == 1 else 2, 3), dtype=float) + rank

pricing_results = comm.gather(local_pricing, root=0)

if rank == 0:
    print(pricing_results)
    pricing_results = np.concatenate(pricing_results)

    print(pricing_results)