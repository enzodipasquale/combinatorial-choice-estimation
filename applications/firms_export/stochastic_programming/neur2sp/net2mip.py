import numpy as np
import gurobipy as gp

BIG = 1e15


def compute_layer_bounds(weights, biases, input_lb, input_ub):
    lb = np.asarray(input_lb, dtype=float)
    ub = np.asarray(input_ub, dtype=float)
    pre_bounds = []
    for W, b in zip(weights, biases):
        Wp, Wn = np.maximum(W, 0), np.minimum(W, 0)
        with np.errstate(over="ignore", invalid="ignore"):
            pre_lb = np.clip(Wp @ lb + Wn @ ub + b, -BIG, BIG)
            pre_ub = np.clip(Wp @ ub + Wn @ lb + b, -BIG, BIG)
        pre_bounds.append((pre_lb, pre_ub))
        lb = np.maximum(pre_lb, 0)
        ub = np.maximum(pre_ub, 0)
    return pre_bounds


def embed_relu(model, weights, biases, input_vars, pre_bounds):
    n_layers = len(weights)
    prev = list(input_vars)

    for l in range(n_layers):
        W, b = weights[l], biases[l]
        d_out, d_in = W.shape
        pl, pu = pre_bounds[l]

        if l == n_layers - 1:
            out = model.addVar(lb=-gp.GRB.INFINITY, name="nn_out")
            expr = gp.LinExpr(float(b[0]))
            for k in range(d_in):
                expr.addTerms(float(W[0, k]), prev[k])
            model.addConstr(out == expr, name="nn_out_eq")
            model.update()
            return out

        hp = [model.addVar(lb=0, ub=max(float(pu[j]), 0),
                           name=f"hp{l}_{j}") for j in range(d_out)]
        hn = [model.addVar(lb=0, ub=max(float(-pl[j]), 0),
                           name=f"hn{l}_{j}") for j in range(d_out)]
        z = [model.addVar(vtype=gp.GRB.BINARY,
                          name=f"z{l}_{j}") for j in range(d_out)]
        model.update()

        for j in range(d_out):
            expr = gp.LinExpr(float(b[j]))
            for k in range(d_in):
                expr.addTerms(float(W[j, k]), prev[k])

            if pu[j] <= 1e-8:
                model.addConstr(hp[j] == 0, name=f"fix0_{l}_{j}")
            elif pl[j] >= -1e-8:
                model.addConstr(hp[j] == expr, name=f"fixA_{l}_{j}")
            else:
                model.addConstr(expr == hp[j] - hn[j], name=f"pre{l}_{j}")
                model.addConstr(hp[j] <= float(pu[j]) * z[j],
                                name=f"bMp{l}_{j}")
                model.addConstr(hn[j] <= float(-pl[j]) * (1 - z[j]),
                                name=f"bMn{l}_{j}")

        prev = hp

    raise ValueError("Network has no output layer")
