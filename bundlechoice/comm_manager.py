import numpy as np
from mpi4py import MPI

class CommManager:

    def __init__(self, comm):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.comm_size = comm.Get_size()
        self.root = 0

    def is_root(self):
        return self.rank == self.root

    def _barrier(self):
        self.comm.Barrier()

    def scatter(self, data):
        return self.comm.scatter(data, root=self.root)

    def bcast(self, data):
        return self.comm.bcast(data, root=self.root)

    def gather(self, data):
        return self.comm.gather(data, root=self.root)

    def Scatterv_by_row(self, send_array, row_counts, dtype=None, shape=None):
        if dtype is None or shape is None:
            dtype, shape = self.bcast((send_array.dtype, send_array.shape) if self.is_root() else (None, None))
        tail_shape = shape[1:]
        tail_stride = int(np.prod(tail_shape)) if tail_shape else 1
        counts = row_counts * tail_stride
        if self.is_root():
            send_array = np.ascontiguousarray(send_array, dtype=dtype).ravel()
            sendbuf = (send_array, counts)
        else:
            sendbuf = None
        recvbuf = np.empty(int(counts[self.rank]), dtype=dtype)
        self.comm.Scatterv(sendbuf, recvbuf, root=self.root)
        if len(tail_shape):
            recvbuf = recvbuf.reshape((int(row_counts[self.rank]),) + tail_shape)
        return recvbuf

    def Gatherv_by_row(self, local_array, row_counts = None):
        local_flat = np.ascontiguousarray(local_array).ravel()
        if row_counts is None:
            local_size = np.array([local_flat.size], dtype=np.int64)
            counts = np.empty(self.comm_size, dtype=np.int64) if self.is_root() else None
            self.comm.Gather(local_size, counts, root=self.root)
        else:
            counts = row_counts * int(np.prod(local_array.shape[1:]))
        if self.is_root():
            recvbuf = np.empty(int(counts.sum()), dtype=local_array.dtype)
            self.comm.Gatherv(local_flat, (recvbuf, counts), root=self.root)
            if local_array.ndim > 1:
                return recvbuf.reshape((-1,) + local_array.shape[1:])
            return recvbuf
        else:
            self.comm.Gatherv(local_flat, None, root=self.root)
            return None
    def Allgather(self, array):
        sendbuf = np.ascontiguousarray(array)
        recvbuf = np.empty(self.comm_size * sendbuf.size, dtype=sendbuf.dtype)
        self.comm.Allgather(sendbuf, recvbuf)
        return recvbuf    

    def Reduce(self, array, op = MPI.SUM):
        sendbuf = np.ascontiguousarray(array)
        recvbuf = np.empty_like(sendbuf)
        self.comm.Reduce(sendbuf, recvbuf, op=op, root=self.root)
        recvbuf = None if not self.is_root() else recvbuf
        return recvbuf

    def sum_row_andReduce(self, array):
        sendbuf = array.sum(0)
        return self.Reduce(sendbuf)

    def Bcast(self, array):
        if self.is_root():
            array = np.ascontiguousarray(array)
        self.comm.Bcast(array, root=self.root) 
        return array


    def get_dict_metadata(self, data_dict):
        if self.is_root():
            meta = {k: ('arr', v.shape, v.dtype) if isinstance(v, np.ndarray) else ('obj', None, None) for k, v in data_dict.items()}
        else:
            meta = None
        return self.comm.bcast(meta, root=self.root)

    def scatter_dict(self, data_dict, agent_counts=None, return_metadata=False):
        meta = self.get_dict_metadata(data_dict)
        out = {}
        for k, (kind, shape, dtype) in meta.items():
            if kind == 'arr':
                send_arr = data_dict[k] if self.is_root() else None
                out[k] = self.Scatterv_by_row(send_arr, row_counts=agent_counts, dtype= dtype, shape= shape)
            else:
                out[k] = self.comm.bcast(data_dict[k] if self.is_root() else None, root=self.root)
        if return_metadata:
            return out, meta
        return out

    def bcast_dict(self, data_dict, return_metadata=False):
        meta = self.get_dict_metadata(data_dict)
        out = {}
        for k, (kind, shape, dtype) in meta.items():
            if kind == 'arr':
                arr = data_dict[k] if self.is_root() else np.empty(shape, dtype=dtype)
                self.Bcast(arr)
                out[k] = arr
            else:
                out[k] = self.comm.bcast(data_dict[k] if self.is_root() else None, root=self.root)
        if return_metadata:
            return out, meta
        return out