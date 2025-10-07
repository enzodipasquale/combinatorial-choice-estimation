"""
Communication manager for MPI operations.

This module provides a clean interface for MPI communication operations,
wrapping the underlying MPI functionality to improve readability and scalability.
"""

from typing import Any, Optional, Callable
import numpy as np
from mpi4py import MPI
from functools import wraps


def _get_mpi_type(dtype: np.dtype) -> MPI.Datatype:
    """Map numpy dtype to MPI datatype."""
    type_map = {
        np.float64: MPI.DOUBLE, np.float32: MPI.FLOAT,
        np.int32: MPI.INT, np.int64: MPI.LONG, np.bool_: MPI.BOOL
    }
    return type_map.get(np.dtype(dtype).type, MPI.DOUBLE)


def _mpi_error_handler(func: Callable) -> Callable:
    """
    Decorator to handle MPI errors by aborting all processes to prevent deadlock.
    
    Args:
        func: MPI operation function to wrap
        
    Returns:
        Wrapped function with error handling
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            # If any rank fails, abort all processes to prevent deadlock
            self.comm.Abort(1)
            raise e
    return wrapper


class CommManager:
    """
    Manager for MPI communication operations.
    
    This class provides a clean interface for common MPI operations like
    scatter, broadcast, gather, and reduction operations. It wraps the
    underlying MPI functionality to improve readability and maintainability.
    
    Attributes:
        comm: The underlying MPI communicator
        rank: Current process rank
        size: Total number of processes
    """
    
    def __init__(self, comm: MPI.Comm):
        """
        Initialize the communication manager.
        
        Args:
            comm: MPI communicator to use for operations
        """
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
    
    def is_root(self) -> bool:
        """
        Check if the current rank is the root rank (rank 0).
        
        Returns:
            True if current rank is 0, False otherwise
        """
        return self.rank == 0
    
    @_mpi_error_handler
    def scatter_from_root(self, data: Any, root: int = 0) -> Any:
        """
        Scatter data from root to all ranks.
        
        Args:
            data: Data to scatter (only used on root rank)
            root: Root rank that holds the data (default: 0)
            
        Returns:
            Local chunk of data for this rank
        """
        return self.comm.scatter(data, root=root)
    
    @_mpi_error_handler
    def broadcast_from_root(self, data: Any, root: int = 0) -> Any:
        """
        Broadcast data from root to all ranks.
        
        Args:
            data: Data to broadcast (only used on root rank)
            root: Root rank that holds the data (default: 0)
            
        Returns:
            Broadcasted data for all ranks
        """
        return self.comm.bcast(data, root=root)
    
    @_mpi_error_handler
    def gather_at_root(self, data: Any, root: int = 0) -> Any:
        """
        Gather data from all ranks to root.
        
        Args:
            data: Local data to gather
            root: Root rank that will receive all data (default: 0)
            
        Returns:
            Gathered data (only meaningful on root rank)
        """
        return self.comm.gather(data, root=root)
    
    @_mpi_error_handler
    def all_reduce(self, data: Any, op: MPI.Op = MPI.SUM) -> Any:
        """
        Perform reduction operation across all ranks.
        
        Args:
            data: Local data to reduce
            op: Reduction operation (default: MPI.SUM)
            
        Returns:
            Result of reduction operation on all ranks
        """
        return self.comm.allreduce(data, op=op)
    
    @_mpi_error_handler
    def reduce_at_root(self, data: Any, op: MPI.Op = MPI.SUM, root: int = 0) -> Any:
        """
        Perform reduction operation and send result to root.
        
        Args:
            data: Local data to reduce
            op: Reduction operation (default: MPI.SUM)
            root: Root rank that will receive the result (default: 0)
            
        Returns:
            Result of reduction operation (only meaningful on root rank)
        """
        return self.comm.reduce(data, op=op, root=root)
    
    @_mpi_error_handler
    def concatenate_at_root(self, data: Any, root: int = 0) -> Optional[Any]:
        """
        Gather data from all ranks and concatenate at root.
        
        Args:
            data: Local data to gather and concatenate
            root: Root rank that will receive concatenated data (default: 0)
            
        Returns:
            Concatenated data on root rank, None on other ranks
        """
        gathered = self.gather_at_root(data, root=root)
        if self.is_root():
            return np.concatenate(gathered)
        return None
    
    def execute_at_root(self, func: callable, *args, **kwargs) -> Optional[Any]:
        """
        Execute a function only on the root rank.
        
        Args:
            func: Function to execute
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Result of function execution on root rank, None on other ranks
        """
        if self.is_root():
            return func(*args, **kwargs)
        return None
    
    @_mpi_error_handler
    def barrier(self) -> None:
        """
        Synchronize all ranks at a barrier.
        """
        self.comm.Barrier()
    
    # --- High-performance numpy array methods (use MPI buffers, not pickle) ---
    
    @_mpi_error_handler
    def broadcast_array(self, array: np.ndarray, root: int = 0) -> np.ndarray:
        """
        Broadcast numpy array using MPI buffers (faster than pickle for large arrays).
        Array must exist on all ranks with same shape/dtype.
        """
        self.comm.Bcast(array, root=root)
        return array
    
    @_mpi_error_handler
    def concatenate_array_at_root(self, local_array: np.ndarray, root: int = 0) -> Optional[np.ndarray]:
        """
        Gather and concatenate numpy arrays from all ranks at root using MPI buffers.
        Concatenates along axis 0. Returns result at root, None elsewhere.
        """
        # Flatten, gather metadata
        local_flat = local_array.ravel()
        all_sizes = self.comm.gather(local_flat.size, root=root)
        
        if not self.is_root():
            self.comm.Gatherv(local_flat, None, root=root)
            return None
        
        # Root: prepare buffer and gather
        result_flat = np.empty(sum(all_sizes), dtype=local_array.dtype)
        displacements = [0] + list(np.cumsum(all_sizes)[:-1])
        self.comm.Gatherv(local_flat, [result_flat, all_sizes, displacements, _get_mpi_type(local_array.dtype)], root=root)
        
        # Reshape if needed
        if local_array.ndim == 1:
            return result_flat
        
        all_shapes = self.comm.gather(local_array.shape, root=root)
        result_shape = (sum(s[0] for s in all_shapes),) + all_shapes[0][1:]
        return result_flat.reshape(result_shape)