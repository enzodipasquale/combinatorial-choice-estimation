"""Shared MILP shell for multi-stage facility location."""

import numpy as np
import gurobipy as gp


def build_milp_shell(mdl, ng, P, nm, N, L1, L2, feasible, cell_groups, platforms):
    """Variables + constraints for the multi-stage MILP. No objective."""
    y1 = mdl.addMVar((ng, L1), vtype=gp.GRB.BINARY)
    y2 = mdl.addMVar((P, L2), vtype=gp.GRB.BINARY)
    z = mdl.addMVar((nm, N), vtype=gp.GRB.BINARY,
                     ub=feasible.astype(float))
    x = mdl.addMVar((nm, N, L1, L2), lb=0, ub=1)

    # Path sum == market entry
    x_sum = x.reshape(nm * N, L1 * L2) @ np.ones(L1 * L2)
    mdl.addConstr(x_sum == z.reshape(nm * N))

    # Facility constraints per model (group/platform assignments differ)
    for m in range(nm):
        g, p = int(cell_groups[m]), int(platforms[m])
        for l1 in range(L1):
            mdl.addConstr(x[m, :, l1, :] <= y1[g, l1])
        for l2 in range(L2):
            mdl.addConstr(x[m, :, :, l2] <= y2[p, l2])

    return y1, y2, z, x
