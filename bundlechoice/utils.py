import sys
import os
import contextlib
from datetime import datetime
import numpy as np

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


def log_iteration(iteration, lambda_k, rank=0):
    if rank == 0:
        print("#" * 100)
        print(f"ITERATION: {iteration}")
        print("Time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

def log_solution(master_pb, lambda_k_iter, rank):
    if rank == 0:
        print("-"*80)
        print("Solution found:", lambda_k_iter)
        print("-"*80)

        os.makedirs("output", exist_ok=True)
        master_pb.write('output/master_pb.mps')
        master_pb.write('output/master_pb.bas')


def log_init_master(self, x_hat_k):
    print("#"*100)
    print("PROBLEM DETAILS")
    print("number of agents     :", self.num_agents)
    print("number of items      :", self.num_items)
    print("number of features   :", self.num_features)
    print("number of simulations:", self.num_simuls)
    print("first moments:", x_hat_k)
    print('-'*100)
    print("SETTINGS")
    print("tol certificate     :", self.config.tol_certificate)
    print("max slack_counter   : ", self.config.max_slack_counter)
    print("tol row_generation  :", self.config.tol_row_generation)
    print("row generation decay:", self.config.row_generation_decay)
    print("mininmum iterations :", self.config.min_iters)
    print("maximum iterations  :", self.config.max_iters)
    print("SLURM")
    print("comm size           :", self.comm_size)
    print("#"*100)

