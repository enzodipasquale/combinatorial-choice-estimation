import logging
import os
import sys
from contextlib import contextmanager
try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
except ImportError:
    rank = 0  # fallback for non-MPI runs

class MPIRankFilter(logging.Filter):
    """
    Logging filter that only allows messages from MPI rank 0.
    
    This filter ensures that logging output only comes from the root process
    in MPI applications, preventing duplicate log messages from multiple ranks.
    """
    def filter(self, record):
        """Filter log records to only allow rank 0."""
        return MPI.COMM_WORLD.Get_rank() == 0


def get_logger(name=__name__):
    """
    Get a logger with MPI rank filtering.
    
    This function creates a logger that automatically filters messages
    to only show output from MPI rank 0, preventing duplicate logging
    in distributed applications.
    
    Args:
        name: Logger name (defaults to module name)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    # Add the rank 0 filter only once
    if not any(isinstance(f, MPIRankFilter) for f in logger.filters):
        logger.addFilter(MPIRankFilter())
    return logger

@contextmanager
def suppress_output():
    """
    Context manager to suppress stdout and stderr output.
    
    This context manager temporarily redirects stdout and stderr to /dev/null,
    useful for silencing verbose output from external libraries like Gurobi.
    
    Example:
        with suppress_output():
            # Code that produces unwanted output
            solver.solve()
    """
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr 