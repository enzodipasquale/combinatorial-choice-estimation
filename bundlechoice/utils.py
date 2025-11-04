"""
Utility functions for logging and output suppression.
"""

import logging
import os
import sys
from contextlib import contextmanager
from typing import Optional
try:
    from mpi4py import MPI
except ImportError:
    MPI = None


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