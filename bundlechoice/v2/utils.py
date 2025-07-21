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
    def filter(self, record):
        return MPI.COMM_WORLD.Get_rank() == 0


def get_logger(name=__name__):
    logger = logging.getLogger(name)
    # Add the rank 0 filter only once
    if not any(isinstance(f, MPIRankFilter) for f in logger.filters):
        logger.addFilter(MPIRankFilter())
    return logger

@contextmanager
def suppress_output():
    """
    Context manager to suppress stdout and stderr (useful for silencing Gurobi or other verbose libraries).
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