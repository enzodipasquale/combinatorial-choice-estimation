"""Embed a trained ReLU neural network into a Gurobi MIP model.

The network has ReLU activations on hidden layers and a linear output layer.
Each ReLU unit is encoded with one binary variable (active/inactive indicator),
two non-negative continuous variables (positive and negative parts), and big-M
constraints derived from interval-arithmetic bounds.
"""
import numpy as np
import gurobipy as gp


def compute_layer_bounds(weights, biases, input_lb, input_ub):
    """Interval-arithmetic bounds on pre-activation values at each layer.

    Parameters
    ----------
    weights : list of ndarray, shapes [(d1,d0), (d2,d1), ..., (d_out,d_{L-1})]
    biases  : list of ndarray, shapes [(d1,), (d2,), ..., (d_out,)]
    input_lb, input_ub : ndarray of shape (d0,)

    Returns
    -------
    pre_bounds : list of (lb, ub) ndarrays for each layer's pre-activation
    """
    lb = np.asarray(input_lb, dtype=float)
    ub = np.asarray(input_ub, dtype=float)
    pre_bounds = []
    for W, b in zip(weights, biases):
        Wp = np.maximum(W, 0)
        Wn = np.minimum(W, 0)
        pre_lb = Wp @ lb + Wn @ ub + b
        pre_ub = Wp @ ub + Wn @ lb + b
        pre_bounds.append((pre_lb, pre_ub))
        # post-ReLU bounds become next layer's input bounds
        lb = np.maximum(pre_lb, 0)
        ub = np.maximum(pre_ub, 0)
    return pre_bounds


def embed_relu(model, weights, biases, input_vars, pre_bounds):
    """Encode a ReLU network as MIP constraints in *model*.

    Hidden layers use ReLU; the final layer is linear (no activation).

    Parameters
    ----------
    model      : gurobipy.Model
    weights    : list of ndarray  (one per layer, including output)
    biases     : list of ndarray  (one per layer, including output)
    input_vars : list of gurobipy.Var, length = input_dim
    pre_bounds : from ``compute_layer_bounds``

    Returns
    -------
    output_var : gurobipy.Var  (the scalar network output)
    """
    n_layers = len(weights)
    prev = list(input_vars)

    for l in range(n_layers):
        W, b = weights[l], biases[l]
        d_out, d_in = W.shape
        pl, pu = pre_bounds[l]
        is_output = (l == n_layers - 1)

        if is_output:
            # linear output layer (no ReLU)
            out = model.addVar(lb=-gp.GRB.INFINITY, name="nn_out")
            expr = gp.LinExpr(float(b[0]))
            for k in range(d_in):
                expr.addTerms(float(W[0, k]), prev[k])
            model.addConstr(out == expr, name="nn_out_eq")
            model.update()
            return out

        # --- ReLU hidden layer ---
        hp = [model.addVar(lb=0, ub=max(float(pu[j]), 0),
                           name=f"hp{l}_{j}") for j in range(d_out)]
        hn = [model.addVar(lb=0, ub=max(float(-pl[j]), 0),
                           name=f"hn{l}_{j}") for j in range(d_out)]
        z = [model.addVar(vtype=gp.GRB.BINARY,
                          name=f"z{l}_{j}") for j in range(d_out)]
        model.update()

        for j in range(d_out):
            # build pre-activation expression
            expr = gp.LinExpr(float(b[j]))
            for k in range(d_in):
                expr.addTerms(float(W[j, k]), prev[k])

            if pu[j] <= 1e-8:
                # always inactive  →  ReLU output = 0
                model.addConstr(hp[j] == 0, name=f"fix0_{l}_{j}")
            elif pl[j] >= -1e-8:
                # always active  →  ReLU output = pre-activation
                model.addConstr(hp[j] == expr, name=f"fixA_{l}_{j}")
            else:
                # general case: big-M encoding
                M_pos = float(pu[j])
                M_neg = float(-pl[j])
                model.addConstr(expr == hp[j] - hn[j],
                                name=f"pre{l}_{j}")
                model.addConstr(hp[j] <= M_pos * z[j],
                                name=f"bMp{l}_{j}")
                model.addConstr(hn[j] <= M_neg * (1 - z[j]),
                                name=f"bMn{l}_{j}")

        prev = hp  # ReLU outputs feed next layer

    raise ValueError("Network has no output layer")
