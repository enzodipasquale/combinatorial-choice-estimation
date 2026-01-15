import numpy as np

class CommManager:

    def __init__(self, comm):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.comm_size = comm.Get_size()
        self.root = 0

    def _is_root(self):
        return self.rank == self.root

    def _barrier(self):
        self.comm.Barrier()

    def _scatter(self, data):
        return self.comm.scatter(data, root=self.root)

    def _broadcast(self, data):
        return self.comm.bcast(data, root=self.root)

    def _gather(self, data):
        return self.comm.gather(data, root=self.root)

    def _scatter_array_by_row(self, send_array, row_counts):
        dtype, shape = self._broadcast((send_array.dtype, send_array.shape) if self._is_root() else (None, None))
        tail_shape = shape[1:]
        tail_stride = int(np.prod(tail_shape)) if tail_shape else 1
        counts = row_counts * tail_stride
        if self._is_root():
            send_array = np.ascontiguousarray(send_array, dtype=dtype).ravel()
            sendbuf = (send_array, counts)
        else:
            sendbuf = None
        recvbuf = np.empty(int(counts[self.rank]), dtype=dtype)
        self.comm.Scatterv(sendbuf, recvbuf, root=self.root)
        if len(tail_shape):
            recvbuf = recvbuf.reshape((int(row_counts[self.rank]),) + tail_shape)
        return recvbuf

    def _gather_array_by_row(self, local_array):
        local_flat = np.ascontiguousarray(local_array).ravel()
        local_size = np.array([local_flat.size], dtype=np.int64)
        all_sizes = np.empty(self.comm_size, dtype=np.int64) if self._is_root() else None
        self.comm.Gather(local_size, all_sizes, root=self.root)
        if self._is_root():
            recvbuf = np.empty(all_sizes.sum(), dtype=local_array.dtype)
            self.comm.Gatherv(local_flat, (recvbuf, all_sizes), root=self.root)
            if local_array.ndim > 1:
                return recvbuf.reshape((-1,) + local_array.shape[1:])
            return recvbuf
        else:
            self.comm.Gatherv(local_flat, None, root=self.root)
            return None

    def _broadcast_array(self, array):
        self.comm.Bcast(array, root=self.root)
        return array


    def get_dict_metadata(self, data_dict):
        if self._is_root():
            meta = {k: ('arr', v.shape, v.dtype) if isinstance(v, np.ndarray) else ('obj', None, None) for k, v in data_dict.items()}
        else:
            meta = None
        return self.comm.bcast(meta, root=self.root)

    def _scatter_dict(self, data_dict, agent_counts=None):
        meta = self.get_dict_metadata(data_dict)
        out = {}
        for k, (kind, shape, dtype) in meta.items():
            if kind == 'arr':
                send_arr = data_dict[k] if self._is_root() else None
                out[k] = self._scatter_array_by_row(send_arr, row_counts=agent_counts)
            else:
                out[k] = self.comm.bcast(data_dict[k] if self._is_root() else None, root=self.root)
        return out

    def _broadcast_dict(self, data_dict):
        meta = self.get_dict_metadata(data_dict)
        out = {}
        for k, (kind, shape, dtype) in meta.items():
            if kind == 'arr':
                arr = data_dict[k] if self._is_root() else np.empty(shape, dtype=dtype)
                self.comm.Bcast(arr, root=self.root)
                out[k] = arr
            else:
                out[k] = self.comm.bcast(data_dict[k] if self._is_root() else None, root=self.root)
        return out