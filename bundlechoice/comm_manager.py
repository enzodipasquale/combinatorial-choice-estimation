"""
MPI communication manager for distributed operations.

Provides buffer-based and pickle-based communication primitives with optional profiling.
"""

from typing import Any, Optional, Callable, List, Dict, Tuple, Iterator
import numpy as np
from mpi4py import MPI
from functools import wraps
import time
import logging
import traceback
import sys
from contextlib import contextmanager


# ============================================================================
# Helper Functions
# ============================================================================

def _get_mpi_type(dtype: np.dtype) -> MPI.Datatype:
    """Map numpy dtype to MPI datatype."""
    type_map = {
        np.float64: MPI.DOUBLE, np.float32: MPI.FLOAT,
        np.int32: MPI.INT, np.int64: MPI.LONG, 
        np.uint8: MPI.UNSIGNED_CHAR, np.int8: MPI.CHAR,
        np.bool_: MPI.BOOL
    }
    return type_map.get(np.dtype(dtype).type, MPI.DOUBLE)


def _mpi_error_handler(func: Callable) -> Callable:
    """Decorator: handles MPI errors and optionally profiles communication time."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        profiling_enabled = self.enable_profiling
        start = time.time() if profiling_enabled else None
        try:
            result = func(self, *args, **kwargs)
            if profiling_enabled and self._comm_times is not None and start is not None:
                elapsed = time.time() - start
                self._comm_times[func.__name__] = self._comm_times.get(func.__name__, 0.0) + elapsed
            return result
        except Exception as e:
            self._handle_failure(e, operation=func.__name__)
            raise e
    return wrapper


# ============================================================================
# CommManager
# ============================================================================

class CommManager:
    """MPI communication manager with buffer-based and pickle-based operations."""
    
    def __init__(self, comm: MPI.Comm, enable_profiling: bool = False) -> None:
        """Initialize with MPI communicator and optional profiling."""
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.enable_profiling = enable_profiling
        self._comm_times: Optional[Dict[str, float]] = {} if enable_profiling else None
        self._logger = logging.getLogger(__name__)
        try:
            self.comm.Set_errhandler(MPI.ERRORS_RETURN)
        except Exception:
            # Not all MPI implementations allow customizing the error handler.
            pass

    # ============================================================================
    # Helper Methods
    # ============================================================================

    def is_root(self) -> bool:
        """Check if current rank is root (rank 0)."""
        return self.rank == 0
    
    def _compute_counts(self, total_size: int) -> List[int]:
        """Compute balanced chunk sizes for scatter (last rank gets remainder)."""
        base = total_size // self.size
        counts = [base] * self.size
        counts[-1] += total_size % self.size
        return counts
    
    def _compute_displacements(self, counts: List[int]) -> List[int]:
        """Compute displacements for MPI vector operations."""
        return [0] + list(np.cumsum(counts)[:-1])
    
    def _extract_dict_metadata(self, data_dict: Dict[str, np.ndarray]) -> Tuple[List[str], Dict[str, Tuple[int, ...]], Dict[str, np.dtype]]:
        """Extract keys, shapes, and dtypes from dictionary of arrays."""
        return (list(data_dict.keys()), 
                {k: v.shape for k, v in data_dict.items()},
                {k: v.dtype for k, v in data_dict.items()})

    # ============================================================================
    # Pickle-Based Methods (Flexible, Works with Any Python Object)
    # ============================================================================
    
    @_mpi_error_handler
    def scatter_from_root(self, data: Any, root: int = 0) -> Any:
        """Scatter data from root to all ranks (pickle-based)."""
        return self.comm.scatter(data, root=root)
    
    @_mpi_error_handler
    def broadcast_from_root(self, data: Any, root: int = 0) -> Any:
        """Broadcast data from root to all ranks (pickle-based)."""
        return self.comm.bcast(data, root=root)
    
    @_mpi_error_handler
    def gather_at_root(self, data: Any, root: int = 0) -> Any:
        """Gather data from all ranks to root (pickle-based)."""
        return self.comm.gather(data, root=root)
    
    @_mpi_error_handler
    def concatenate_at_root(self, data: Any, root: int = 0) -> Optional[Any]:
        """Gather and concatenate arrays at root (pickle-based)."""
        gathered = self.gather_at_root(data, root=root)
        return np.concatenate(gathered) if self.is_root() else None
    
    @_mpi_error_handler
    def all_reduce(self, data: Any, op: MPI.Op = MPI.SUM) -> Any:
        """Perform reduction across all ranks."""
        return self.comm.allreduce(data, op=op)
    
    @_mpi_error_handler
    def reduce_at_root(self, data: Any, op: MPI.Op = MPI.SUM, root: int = 0) -> Any:
        """Perform reduction and send result to root."""
        return self.comm.reduce(data, op=op, root=root)
    
    @_mpi_error_handler
    def barrier(self) -> None:
        """Synchronize all ranks at a barrier."""
        self.comm.Barrier()
    
    def execute_at_root(self, func: Callable, *args, **kwargs) -> Optional[Any]:
        """Execute function only on root rank."""
        return func(*args, **kwargs) if self.is_root() else None

    # ============================================================================
    # Buffer-Based Methods (2-20x Faster for NumPy Arrays)
    # ============================================================================
    
    @_mpi_error_handler
    def broadcast_array(self, array: np.ndarray, root: int = 0) -> np.ndarray:
        """Broadcast numpy array using MPI buffers. Array must exist on all ranks with same shape/dtype."""
        self.comm.Bcast(array, root=root)
        return array
    
    @_mpi_error_handler
    def broadcast_array_with_flag(self, array: Optional[np.ndarray], flag: bool, root: int = 0) -> Tuple[np.ndarray, bool]:
        """
        Broadcast numpy array + boolean flag using buffer-based operations (no pickle).
        
        This is optimized for frequent broadcasts (e.g., per-iteration theta + stop flag).
        The array is broadcast using Bcast (buffer-based), and the flag is broadcast as uint8.
        
        Args:
            array: NumPy array to broadcast (on root) or empty array with correct shape/dtype (on non-root)
            flag: Boolean flag to broadcast
            root: Root rank
            
        Returns:
            Tuple of (broadcasted_array, broadcasted_flag)
        """
        if self.is_root():
            if array is None:
                raise ValueError("array required on root")
            array_shape = array.shape
            array_dtype = array.dtype
        else:
            if array is None:
                raise ValueError("array required on non-root (must pre-allocate with correct shape/dtype)")
            array_shape = array.shape
            array_dtype = array.dtype
        
        # Broadcast array using buffer-based Bcast
        self.comm.Bcast(array, root=root)
        
        # Broadcast flag as uint8 (1 byte, buffer-based)
        flag_uint8 = np.array([1 if flag else 0], dtype=np.uint8) if self.is_root() else np.array([0], dtype=np.uint8)
        self.comm.Bcast(flag_uint8, root=root)
        flag_result = bool(flag_uint8[0])
        
        return array, flag_result
    
    @_mpi_error_handler
    def scatter_array(self, send_array: Optional[np.ndarray] = None, counts: Optional[List[int]] = None, 
                     root: int = 0, dtype: Optional[np.dtype] = None) -> np.ndarray:
        """Scatter numpy array using MPI buffers (5-20x faster than pickle)."""
        if self.is_root():
            if send_array is None:
                raise ValueError("send_array required on root")
            counts = counts or self._compute_counts(len(send_array))
            dtype = send_array.dtype
        elif dtype is None:
            raise ValueError("dtype required on non-root if send_array is None")
        
        counts, dtype = self.comm.bcast((counts, dtype), root=root)
        
        sendbuf = [send_array, counts, self._compute_displacements(counts), _get_mpi_type(dtype)] if self.is_root() else None
        recvbuf = np.empty(counts[self.rank], dtype=dtype)
        self.comm.Scatterv(sendbuf, recvbuf, root=root)
        return recvbuf
    
    @_mpi_error_handler
    def scatter_dict(self, data_dict: Optional[Dict[str, np.ndarray]] = None, 
                    counts: Optional[List[int]] = None, root: int = 0) -> Dict[str, np.ndarray]:
        """Scatter dictionary of arrays using MPI buffers (2-9x faster)."""
        if self.is_root():
            if not data_dict:
                raise ValueError("data_dict required on root")
            keys, shapes, dtypes = self._extract_dict_metadata(data_dict)
            counts = counts or self._compute_counts(data_dict[keys[0]].shape[0])
        else:
            keys = shapes = dtypes = None
        
        keys, shapes, dtypes, counts = self.comm.bcast((keys, shapes, dtypes, counts), root=root)
        
        result = {}
        for key in keys:
            shape = data_dict[key].shape if self.is_root() else shapes[key]
            flat_counts = [c * int(np.prod(shape[1:])) for c in counts]
            send_flat = data_dict[key].reshape(shape[0], -1) if self.is_root() else None
            local_flat = self.scatter_array(send_flat, counts=flat_counts, root=root, dtype=dtypes[key])
            result[key] = local_flat.reshape((counts[self.rank],) + shape[1:])
        return result
    
    @_mpi_error_handler  
    def broadcast_dict(self, data_dict: Optional[Dict[str, np.ndarray]] = None, 
                      root: int = 0) -> Dict[str, np.ndarray]:
        """Broadcast dictionary of arrays using MPI buffers (1.5-3x faster)."""
        meta = self._extract_dict_metadata(data_dict) if self.is_root() else None
        keys, shapes, dtypes = self.comm.bcast(meta, root=root)
        
        result = {}
        for key in keys:
            array = data_dict[key] if self.is_root() else np.empty(shapes[key], dtype=dtypes[key])
            self.comm.Bcast(array, root=root)
            result[key] = array
        return result
    
    @_mpi_error_handler
    def concatenate_array_at_root_fast(self, local_array: np.ndarray, root: int = 0) -> Optional[np.ndarray]:
        """
        Gather and concatenate arrays using MPI buffers (optimized, no pickle).
        
        Handles bool arrays by packing as uint8, and uses buffer-based shape gathering
        instead of pickle for better performance with many ranks.
        """
        # Handle bool arrays by packing as uint8 (1 byte per bool, same as bool_)
        if local_array.dtype == np.bool_:
            local_array = local_array.astype(np.uint8)
            unpack_bool = True
        else:
            unpack_bool = False
        
        local_flat = local_array.ravel()
        ndim = local_array.ndim
        
        # Gather sizes using Allgather (fast, buffer-based)
        all_sizes_array = np.empty(self.size, dtype=np.int64)
        self.comm.Allgather(np.array([local_flat.size], dtype=np.int64), all_sizes_array)
        all_sizes = all_sizes_array.tolist()
        
        # Gather shapes using buffer-based Allgather instead of pickle
        # Shape is a tuple of length ndim, send as int array
        shape_array = np.array(local_array.shape, dtype=np.int64)
        all_shapes_buffer = np.empty(self.size * ndim, dtype=np.int64)
        self.comm.Allgather(shape_array, all_shapes_buffer)
        
        # Reshape to (size, ndim) and extract shapes on root
        if self.is_root():
            all_shapes = all_shapes_buffer.reshape(self.size, ndim)
            # Verify all ranks have same shape (except first dimension)
            if ndim > 1:
                shape_template = all_shapes[0, 1:]
                if not np.all(all_shapes[:, 1:] == shape_template):
                    raise ValueError("Inconsistent shapes across ranks (non-first dimensions must match)")
        else:
            all_shapes = None
        
        # Gather actual data using Gatherv (fast, buffer-based)
        recvbuf = [np.empty(sum(all_sizes), dtype=local_array.dtype), all_sizes, 
                   self._compute_displacements(all_sizes), _get_mpi_type(local_array.dtype)] if self.is_root() else None
        self.comm.Gatherv(local_flat, recvbuf, root=root)
        
        if not self.is_root():
            return None
        
        result = recvbuf[0]
        
        # Reshape result
        if ndim == 1:
            final_result = result
        else:
            # Reconstruct shape: first dim is sum of all first dims, rest from template
            first_dim = sum(all_shapes[:, 0])
            rest_shape = tuple(all_shapes[0, 1:])
            final_result = result.reshape((first_dim,) + rest_shape)
        
        # Unpack bool if needed
        if unpack_bool:
            final_result = final_result.astype(np.bool_)
        
        return final_result

    # ============================================================================
    # Profiling
    # ============================================================================

    def get_comm_profile(self) -> Optional[Dict[str, float]]:
        """Get communication profiling data (operation name → total time)."""
        return self._comm_times if self.enable_profiling else None
    
    def reset_comm_profile(self) -> None:
        """Reset communication profiling counters."""
        if self.enable_profiling:
            self._comm_times = {}

    # ============================================================================
    # Failure Handling
    # ============================================================================

    def _handle_failure(self, exc: Exception, operation: Optional[str] = None, errorcode: int = 1) -> None:
        """Log the local failure and abort all ranks."""
        op = operation or "MPI operation"
        formatted_traceback = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        message = (
            f"Rank {self.rank}/{self.size} encountered an error during '{op}': {exc}\n"
            f"{formatted_traceback}"
        )
        try:
            if self._logger:
                self._logger.error(message)
            else:
                sys.stderr.write(message)
        finally:
            sys.stderr.flush()
            try:
                self.comm.Abort(errorcode)
            except Exception:
                # As a last resort, ensure the process exits to avoid hanging.
                sys.exit(errorcode)

    @contextmanager
    def fail_fast(self, operation: Optional[str] = None, errorcode: int = 1) -> Iterator[None]:
        """
        Context manager that aborts all ranks if an exception escapes the block.

        Usage:
            with comm_manager.fail_fast("setup phase"):
                ...
        """
        try:
            yield
        except Exception as exc:
            self._handle_failure(exc, operation=operation, errorcode=errorcode)
            raise

    def abort_all(self, message: Optional[str] = None, errorcode: int = 1) -> None:
        """
        Explicitly abort all ranks, propagating a message for diagnostics.
        """
        if message:
            diagnostic = f"Rank {self.rank}/{self.size} aborting all ranks: {message}"
            if self._logger:
                self._logger.error(diagnostic)
            else:
                sys.stderr.write(diagnostic + "\n")
                sys.stderr.flush()
        try:
            self.comm.Abort(errorcode)
        except Exception:
            sys.exit(errorcode)