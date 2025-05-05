import sys
import os
import contextlib
from datetime import datetime
import numpy as np

def price_term(p_j, bundle_j = None):
        if p_j is None:
            return np.array([0])
        if bundle_j is None:
            return p_j
        else:
            return bundle_j @ p_j 

def update_slack_counter(master_pb, slack_counter):
    num_constrs_removed = 0
    for constr in master_pb.getConstrs():
        if constr.ConstrName not in slack_counter:
            slack_counter[constr.ConstrName] = 0
        if constr.Slack < 0:
            slack_counter[constr.ConstrName] += 1

        if slack_counter[constr.ConstrName] >= slack_counter["MAX_SLACK_COUNTER"]:
            master_pb.remove(constr)
            slack_counter.pop(constr.ConstrName)
            num_constrs_removed += 1

    return slack_counter, num_constrs_removed


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
    if rank != 0:
        return
    print("#" * 80)
    print(f"ITERATION: {iteration}")
    print("Time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    # print("Parameter:", np.array2string(lambda_k, precision=4, separator=', '))
    print("#" * 80)
