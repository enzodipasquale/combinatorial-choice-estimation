"""
Utility functions for logging, output suppression, and timing.
"""

import logging
import os
import sys
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional
import numpy as np
from numpy.typing import NDArray
try:
    from mpi4py import MPI
except ImportError:
    MPI = None


# ============================================================================
# Theta Extraction
# ============================================================================

def extract_theta(theta: Any) -> NDArray[np.float64]:
    """
    Extract theta array from EstimationResult or raw array.
    
    Args:
        theta: Either an EstimationResult object (with theta_hat attribute) or a numpy array
        
    Returns:
        Numpy array of theta values
    """
    if hasattr(theta, 'theta_hat'):
        return np.asarray(theta.theta_hat, dtype=np.float64)
    return np.asarray(theta, dtype=np.float64)


# ============================================================================
# Logging Utilities
# ============================================================================

class MPIRankFilter(logging.Filter):
    """Logging filter: only allows messages from MPI rank 0."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter log records to only allow rank 0."""
        if MPI is None:
            return True
        return MPI.COMM_WORLD.Get_rank() == 0


def get_logger(name: str = __name__) -> logging.Logger:
    """Get logger with MPI rank filtering (only rank 0 logs)."""
    logger = logging.getLogger(name)
    if not any(isinstance(f, MPIRankFilter) for f in logger.filters):
        logger.addFilter(MPIRankFilter())
    return logger

# ============================================================================
# Output Suppression
# ============================================================================

@contextmanager
def suppress_output():
    """Context manager: suppress stdout/stderr (useful for Gurobi output)."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        old_logging_level = logging.getLogger().level
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            logging.getLogger().setLevel(logging.ERROR)
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            logging.getLogger().setLevel(old_logging_level)

# ============================================================================
# Timing Statistics
# ============================================================================

def make_timing_stats(elapsed: float, num_iterations: int,
                      pricing_time: float = 0.0, master_time: float = 0.0, 
                      mpi_time: float = 0.0, init_time: float = 0.0) -> Dict[str, Any]:
    """
    Create timing statistics dictionary from running sums.
    
    Args:
        elapsed: Total elapsed time
        num_iterations: Number of iterations completed
        pricing_time: Total time in pricing/subproblems
        master_time: Total time in master problem
        mpi_time: Total time in MPI communication
        init_time: Initialization time
        
    Returns:
        Dict with timing statistics
    """
    return {
        'total_time': elapsed,
        'num_iterations': num_iterations,
        'pricing_time': pricing_time,
        'master_time': master_time,
        'mpi_time': mpi_time,
        'init_time': init_time,
        'pricing_pct': 100 * pricing_time / elapsed if elapsed > 0 else 0,
        'master_pct': 100 * master_time / elapsed if elapsed > 0 else 0,
        'mpi_pct': 100 * mpi_time / elapsed if elapsed > 0 else 0,
    }

# ============================================================================
# Timing Utilities
# ============================================================================

@contextmanager
def time_operation(name: str, timing_dict: Dict[str, float]):
    """
    Context manager for timing operations and storing results in a dictionary.
    
    Args:
        name: Name of the operation to time
        timing_dict: Dictionary to store timing results (will be updated in-place)
    
    Example:
        timing = {}
        with time_operation('pricing', timing):
            result = solve_subproblems()
        # timing['pricing'] now contains elapsed time in seconds
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        timing_dict[name] = elapsed

