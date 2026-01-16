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


class MPIRankFilter(logging.Filter):
    def filter(self, record):
        if MPI is None:
            return True
        return MPI.COMM_WORLD.Get_rank() == 0

def get_logger(name=__name__):
    logger = logging.getLogger(name)
    if not any((isinstance(f, MPIRankFilter) for f in logger.filters)):
        logger.addFilter(MPIRankFilter())
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        root_logger = logging.getLogger()
        if root_logger.level > logging.INFO:
            root_logger.setLevel(logging.INFO)
    return logger

@contextmanager
def suppress_output():
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

