"""
Utility functions for logging, output suppression, and timing.
"""

import logging
import os
import sys
from contextlib import contextmanager
from typing import Any, Dict
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
    
    # Configure handler if not already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        # Ensure root logger doesn't block
        root_logger = logging.getLogger()
        if root_logger.level > logging.INFO:
            root_logger.setLevel(logging.INFO)
    
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
                      pricing_time: float = 0.0, other_time: float = None) -> Dict[str, Any]:
    """
    Create simple timing statistics dictionary.
    
    Args:
        elapsed: Total elapsed time (wall-clock)
        num_iterations: Number of iterations completed
        pricing_time: Total time in pricing/subproblems (on this rank)
        other_time: Time for everything else (master + sync). If None, computed as elapsed - pricing_time.
        
    Returns:
        Dict with timing statistics
    """
    if other_time is None:
        other_time = elapsed - pricing_time
    
    return {
        'total_time': elapsed,
        'num_iterations': num_iterations,
        'time_per_iter': elapsed / num_iterations if num_iterations > 0 else 0,
        'pricing_time': pricing_time,
        'pricing_pct': 100 * pricing_time / elapsed if elapsed > 0 else 0,
        'other_time': other_time,
        'other_pct': 100 * other_time / elapsed if elapsed > 0 else 0,
    }
