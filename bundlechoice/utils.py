import logging
import os
import sys
from contextlib import contextmanager
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

def format_number(value, width=None, precision=6, use_scientific_threshold=1e-4):
    if value == 0:
        formatted = "0"
    elif abs(value) < use_scientific_threshold or abs(value) >= 1.0 / use_scientific_threshold:
        exp_str = f"{value:.{precision-1}e}"
        if 'e' in exp_str:
            base, exp = exp_str.split('e')
            exp_val = int(exp)
            if exp_val >= 0:
                formatted = f"{base}e+{exp_val:02d}"
            else:
                formatted = f"{base}e{exp_val:03d}"
        else:
            formatted = exp_str
    else:
        formatted = f"{value:.{precision}f}".rstrip('0').rstrip('.')
    
    if width is not None:
        return f"{formatted:>{width}}"
    return formatted
