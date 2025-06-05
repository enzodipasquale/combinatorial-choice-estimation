import sys
import os
import contextlib
from datetime import datetime
import numpy as np
import logging
import textwrap

logger = logging.getLogger(__name__)

def price_term(p_j, bundle_j):
        if p_j is None:
            return 0
        else:
            return bundle_j @ p_j


@contextlib.contextmanager
def suppress_output():
    """Context manager to suppress stdout and stderr (e.g. Gurobi license banner)."""
    with open(os.devnull, 'w') as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

def log_iteration(iteration, rank=0):
    if rank != 0:
        return
    logger.info("========== Iteration %d ==========", iteration)


def log_solution(master_pb, lambda_k_iter, rank):
    if rank != 0:
        return

    lines = ["========== Final Solution =========="]
    lines += [f"lambda_{k+1} = {val}" for k, val in enumerate(lambda_k_iter)]
    logger.info("\n%s", "\n".join(lines))

    # os.makedirs("output", exist_ok=True)
    # master_pb.write("output/master_pb.mps")
    # master_pb.write("output/master_pb.bas")
    # logger.info("Model written to 'output/master_pb.mps' and 'output/master_pb.bas'")

def log_init_master(self, x_hat_k):
    log_msg = f"""
        ========== Problem Details ==========
        Agents         : {self.num_agents}
        Items          : {self.num_items}
        Features       : {self.num_features}
        Simulations    : {self.num_simuls}
        First moments  : {np.array2string(x_hat_k, precision=2, separator=', ')}

        ========== Solver Settings ==========
        Tol. Certificate    : {self.config.tol_certificate}
        Max Slack Counter   : {self.config.max_slack_counter}
        Tol. Row Generation : {self.config.tol_row_generation}
        Row Gen. Decay      : {self.config.row_generation_decay}
        Min Iterations      : {self.config.min_iters}
        Max Iterations      : {self.config.max_iters}

        ========== MPI / SLURM Info ==========
        Comm Size      : {self.comm_size}
    """
    logger.info("\n%s", textwrap.dedent(log_msg).strip())

