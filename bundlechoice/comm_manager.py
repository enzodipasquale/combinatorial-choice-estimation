import numpy as np
from dataclasses import dataclass
from mpi4py import MPI


@dataclass
class NodeSpreadAssignment:
    task_to_rank: np.ndarray   # (num_tasks,) rank responsible for each task
    my_tasks: np.ndarray       # task indices owned by this rank
    has_tasks: bool             # whether this rank owns any task


class CommManager:

    def __init__(self, comm):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.comm_size = comm.Get_size()
        self.root = 0

        # Node topology via shared-memory splitting (portable, no rank-ordering assumptions)
        self.node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
        self.node_rank = self.node_comm.Get_rank()
        self.node_size = self.node_comm.Get_size()

        # Assign a consistent node_id across all ranks:
        # each node's local root (node_rank==0) gets a unique color via the inter-node comm
        inter_comm = comm.Split(0 if self.node_rank == 0 else MPI.UNDEFINED)
        if self.node_rank == 0:
            self.node_id = inter_comm.Get_rank()
            self.num_nodes = inter_comm.Get_size()
            inter_comm.Free()
        else:
            self.node_id = None
            self.num_nodes = None
        self.node_id = self.node_comm.bcast(self.node_id, root=0)
        self.num_nodes = self.node_comm.bcast(self.num_nodes, root=0)

    def is_root(self):
        return self.rank == self.root

    def _barrier(self):
        self.comm.Barrier()

    def spread_tasks_across_nodes(self, num_tasks):
        base, remainder = divmod(num_tasks, self.num_nodes)
        per_node = np.array([base + (1 if n < remainder else 0) for n in range(self.num_nodes)], dtype=np.int64)
        offset = np.zeros(self.num_nodes + 1, dtype=np.int64)
        np.cumsum(per_node, out=offset[1:])

        lo, hi = int(offset[self.node_id]), int(offset[self.node_id + 1])
        my_tasks = np.array([k for k in range(lo, hi)
                             if (k - lo) % self.node_size == self.node_rank],
                            dtype=np.int64)

        local_map = np.full(num_tasks, -1, dtype=np.int64)
        local_map[my_tasks] = self.rank
        task_to_rank = np.empty(num_tasks, dtype=np.int64)
        self.comm.Allreduce(local_map, task_to_rank, op=MPI.MAX)

        return NodeSpreadAssignment(task_to_rank=task_to_rank,
                                    my_tasks=my_tasks,
                                    has_tasks=len(my_tasks) > 0)

    def scatter(self, data):
        return self.comm.scatter(data, root=self.root)

    def bcast(self, data):
        return self.comm.bcast(data, root=self.root)

    def gather(self, data):
        return self.comm.gather(data, root=self.root)

    def Scatterv_by_row(self, send_array, row_counts, dtype=None, shape=None, root=0):
        if dtype is None or shape is None:
            dtype, shape = self.comm.bcast((send_array.dtype, send_array.shape) if self.rank == root else (None, None), root=root)
        tail_shape = shape[1:]
        tail_stride = int(np.prod(tail_shape)) if tail_shape else 1
        counts = row_counts * tail_stride
        if self.rank == root:
            send_array = np.ascontiguousarray(send_array, dtype=dtype).ravel()
            sendbuf = (send_array, counts)
        else:
            sendbuf = None
        recvbuf = np.empty(int(counts[self.rank]), dtype=dtype)
        self.comm.Scatterv(sendbuf, recvbuf, root=root)
        if len(tail_shape):
            recvbuf = recvbuf.reshape((int(row_counts[self.rank]),) + tail_shape)
        return recvbuf

    def Gatherv_by_row(self, local_array, row_counts=None, root=0):
        local_flat = np.ascontiguousarray(local_array).ravel()
        if row_counts is None:
            local_size = np.array([local_flat.size], dtype=np.int64)
            counts = np.empty(self.comm_size, dtype=np.int64) if self.rank == root else None
            self.comm.Gather(local_size, counts, root=root)
        else:
            counts = row_counts * int(np.prod(local_array.shape[1:]))
        if self.rank == root:
            recvbuf = np.empty(int(counts.sum()), dtype=local_array.dtype)
            self.comm.Gatherv(local_flat, (recvbuf, counts), root=root)
            if local_array.ndim > 1:
                return recvbuf.reshape((-1,) + local_array.shape[1:])
            return recvbuf
        else:
            self.comm.Gatherv(local_flat, None, root=root)
            return None
    def Allgather(self, array):
        sendbuf = np.ascontiguousarray(array)
        recvbuf = np.empty(self.comm_size * sendbuf.size, dtype=sendbuf.dtype)
        self.comm.Allgather(sendbuf, recvbuf)
        return recvbuf    

    def Reduce(self, array, op = MPI.SUM, root = None):
        sendbuf = np.ascontiguousarray(array)
        recvbuf = np.zeros_like(sendbuf)
        root = self.root if root is None else root
        self.comm.Reduce(sendbuf, recvbuf, op=op, root=root)
        recvbuf = None if not self.is_root() else recvbuf
        return recvbuf

    def sum_row_andReduce(self, array):
        sendbuf = array.sum(0)
        return self.Reduce(sendbuf)

    def Bcast(self, array, root=0):
        if self.rank == root:
            array = np.ascontiguousarray(array)
        self.comm.Bcast(array, root=root)
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


    def alltoallv_rows(self, rows_by_dest):
        send_counts = np.zeros(self.comm_size, dtype=np.int64)
        for dest, data in rows_by_dest.items():
            send_counts[dest] = len(data)

        recv_counts = np.empty(self.comm_size, dtype=np.int64)
        self.comm.Alltoall(send_counts, recv_counts)

        sdispls = np.zeros(self.comm_size, dtype=np.int64)
        rdispls = np.zeros(self.comm_size, dtype=np.int64)
        np.cumsum(send_counts[:-1], out=sdispls[1:])
        np.cumsum(recv_counts[:-1], out=rdispls[1:])

        sendbuf = np.concatenate([rows_by_dest[k] for k in sorted(rows_by_dest)]) \
                  if rows_by_dest else np.empty(0, dtype=np.float64)
        recvbuf = np.empty(int(recv_counts.sum()), dtype=np.float64)

        self.comm.Alltoallv(
            [sendbuf, send_counts, sdispls, MPI.DOUBLE],
            [recvbuf, recv_counts, rdispls, MPI.DOUBLE])
        return recvbuf
        

