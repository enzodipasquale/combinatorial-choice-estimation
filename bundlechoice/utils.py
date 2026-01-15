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

def make_timing_stats(elapsed, num_iterations, pricing_times=None, master_times=None):
    if pricing_times is None:
        pricing_times = []
    pricing_arr = np.atleast_1d(np.asarray(pricing_times, dtype=np.float64))
    total_pricing = float(pricing_arr.sum())
    if master_times is None:
        master_times = []
    master_arr = np.atleast_1d(np.asarray(master_times, dtype=np.float64))
    total_master = float(master_arr.sum())
    other_time = max(0, elapsed - total_pricing - total_master)
    stats = {'total_time': elapsed, 'num_iterations': num_iterations, 'time_per_iter': elapsed / num_iterations if num_iterations > 0 else 0, 'pricing_time': total_pricing, 'pricing_pct': 100 * total_pricing / elapsed if elapsed > 0 else 0, 'master_time': total_master, 'master_pct': 100 * total_master / elapsed if elapsed > 0 else 0, 'other_time': other_time, 'other_pct': 100 * other_time / elapsed if elapsed > 0 else 0}
    if len(pricing_arr) > 1:
        stats['pricing_per_iter'] = {'avg': float(pricing_arr.mean()), 'min': float(pricing_arr.min()), 'max': float(pricing_arr.max())}
    if len(master_arr) > 1:
        stats['master_per_iter'] = {'avg': float(master_arr.mean()), 'min': float(master_arr.min()), 'max': float(master_arr.max())}
    return stats