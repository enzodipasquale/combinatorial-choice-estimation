import argparse
import time
import numpy as np
import gurobipy as gp
import torch
from neur2sp.net2mip import compute_layer_bounds, embed_relu


def _make_chars(M, n_rev, seed=42):
    rng = np.random.default_rng(seed)
    rev_base = rng.uniform(0, 1.0, (n_rev, M))
    rev_chars_1 = rev_base + rng.uniform(-0.1, 0.1, (n_rev, M))
    rev_chars_2 = rev_base + rng.uniform(-0.1, 0.1, (n_rev, M))
    _state = (rng.random((1000, M)) > 0.9).astype(float)
    _raw = rng.uniform(0, 1, (M, M))
    syn_chars = (_raw + _raw.T) / 2
    np.fill_diagonal(syn_chars, 0)
    return rev_chars_1, rev_chars_2, syn_chars


def solve_exact(b_1_var, b_2_r_vars, model, theta, rev_chars_1, rev_chars_2,
                syn_chars, b_0, eps1, eps2, beta, R, M):
    n_rev = rev_chars_1.shape[0]
    theta_rev, theta_s, theta_c = theta[:n_rev], theta[n_rev], theta[n_rev + 1]
    C, bR = syn_chars, beta / R

    mod_1 = rev_chars_1.T @ theta_rev + (1 - b_0) * theta_s + eps1
    rev2 = rev_chars_2.T @ theta_rev
    obj = mod_1 @ b_1_var + theta_c * (b_1_var @ C @ b_1_var)
    for r in range(R):
        mod_2 = rev2 + (1 - b_1_var) * theta_s + eps2[r]
        obj += bR * (mod_2 @ b_2_r_vars[r, :] +
                     theta_c * (b_2_r_vars[r, :] @ C @ b_2_r_vars[r, :]))

    model.setObjective(obj, gp.GRB.MAXIMIZE)
    model.optimize()
    return np.array(b_1_var.X) > 0.5, model.ObjVal


def eval_true_obj(b_1, theta, rev_chars_1, rev_chars_2, syn_chars,
                  b_0, eps1, eps2, beta, R, M, K, q_model, q_var):
    n_rev = rev_chars_1.shape[0]
    theta_rev, theta_s, theta_c = theta[:n_rev], theta[n_rev], theta[n_rev + 1]
    C = syn_chars
    b1f = b_1.astype(float)

    mod_1 = rev_chars_1.T @ theta_rev + (1 - b_0) * theta_s + eps1
    v1 = mod_1 @ b1f + theta_c * (b1f @ C @ b1f)

    rev2 = rev_chars_2.T @ theta_rev
    v2_total = 0.0
    for r in range(R):
        c_r = rev2 + (1 - b1f) * theta_s + eps2[r]
        q_model.setObjective(c_r @ q_var + theta_c * (q_var @ C @ q_var),
                             gp.GRB.MAXIMIZE)
        q_model.optimize()
        v2_total += q_model.ObjVal

    return v1 + beta * v2_total / R


def solve_surrogate(theta, rev_chars_1, syn_chars, b_0, eps1,
                    nn_weights, nn_biases, pre_bounds, M, K):
    n_rev = rev_chars_1.shape[0]
    theta_rev, theta_s, theta_c = theta[:n_rev], theta[n_rev], theta[n_rev + 1]
    C = syn_chars

    m = gp.Model("surr")
    m.Params.OutputFlag = 0
    m.Params.Threads = 1
    m.ModelSense = gp.GRB.MAXIMIZE

    b_1 = m.addMVar(M, vtype=gp.GRB.BINARY, name="b_1")
    m.addConstr(b_1.sum() <= K)

    tvars = [m.addVar(lb=float(theta[t]), ub=float(theta[t]),
                      name=f"th_{t}") for t in range(len(theta))]
    m.update()

    nn_out = embed_relu(m, nn_weights, nn_biases,
                        list(b_1.tolist()) + tvars, pre_bounds)

    mod_1 = rev_chars_1.T @ theta_rev + (1 - b_0) * theta_s + eps1
    m.setObjective(mod_1 @ b_1 + theta_c * (b_1 @ C @ b_1) + nn_out,
                   gp.GRB.MAXIMIZE)
    m.optimize()
    return np.array(b_1.X) > 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="neur2sp/model.pt")
    ap.add_argument("--n_test", type=int, default=200)
    ap.add_argument("--R_eval", type=int, default=100)
    ap.add_argument("--seed_chars", type=int, default=42)
    ap.add_argument("--seed_test", type=int, default=777)
    args = ap.parse_args()

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    nn_weights, nn_biases = ckpt["weights"], ckpt["biases"]
    M, K = int(ckpt["M"]), int(ckpt["K"])
    n_rev, beta = int(ckpt["n_rev"]), float(ckpt["beta"])
    theta_lb, theta_ub = ckpt["theta_lb"], ckpt["theta_ub"]

    input_lb = np.concatenate([np.zeros(M), theta_lb])
    input_ub = np.concatenate([np.ones(M), theta_ub])
    pre_bounds = compute_layer_bounds(nn_weights, nn_biases, input_lb, input_ub)

    rev_chars_1, rev_chars_2, syn_chars = _make_chars(M, n_rev, args.seed_chars)
    R = args.R_eval
    rng = np.random.default_rng(args.seed_test)

    print(f"Validating b_1 quality: M={M}, K={K}, R_eval={R}, n_test={args.n_test}")

    m_ex = gp.Model("exact")
    m_ex.Params.OutputFlag = 0
    m_ex.Params.Threads = 1
    m_ex.ModelSense = gp.GRB.MAXIMIZE
    b1_ex = m_ex.addMVar(M, vtype=gp.GRB.BINARY, name="b1")
    b2r_ex = m_ex.addMVar((R, M), vtype=gp.GRB.BINARY, name="b2r")
    m_ex.addConstr(b1_ex.sum() <= K)
    for r in range(R):
        m_ex.addConstr(b2r_ex[r, :].sum() <= K)
    m_ex.update()

    m_q = gp.Model("q")
    m_q.Params.OutputFlag = 0
    m_q.Params.Threads = 1
    m_q.ModelSense = gp.GRB.MAXIMIZE
    b2_q = m_q.addMVar(M, vtype=gp.GRB.BINARY, name="b2")
    m_q.addConstr(b2_q.sum() <= K)
    m_q.update()

    gaps, b1_match = [], 0
    t0 = time.time()

    for t in range(args.n_test):
        theta = np.concatenate([
            rng.uniform(theta_lb[0], theta_ub[0], size=n_rev),
            [rng.uniform(theta_lb[n_rev], theta_ub[n_rev])],
            [rng.uniform(theta_lb[n_rev+1], theta_ub[n_rev+1])],
        ])
        b_0 = (rng.random(M) > 0.9).astype(float)
        eps1 = rng.normal(0, 1, M)
        eps2 = rng.normal(0, 1, (R, M))

        b1_exact, obj_exact = solve_exact(
            b1_ex, b2r_ex, m_ex, theta,
            rev_chars_1, rev_chars_2, syn_chars, b_0, eps1, eps2, beta, R, M)

        b1_nn = solve_surrogate(
            theta, rev_chars_1, syn_chars, b_0, eps1,
            nn_weights, nn_biases, pre_bounds, M, K)

        if np.array_equal(b1_exact, b1_nn):
            obj_nn = obj_exact
            b1_match += 1
        else:
            obj_nn = eval_true_obj(
                b1_nn, theta, rev_chars_1, rev_chars_2, syn_chars,
                b_0, eps1, eps2, beta, R, M, K, m_q, b2_q)

        gap = (obj_exact - obj_nn) / abs(obj_exact) * 100 if abs(obj_exact) > 1e-6 else 0.0
        gaps.append(gap)

        if (t + 1) % 50 == 0:
            print(f"  [{t+1}/{args.n_test}]  {time.time()-t0:.0f}s  "
                  f"mean_gap={np.mean(gaps):.3f}%  b1_match={b1_match}/{t+1}")

    gaps = np.array(gaps)
    print(f"\n{'='*55}")
    print(f"B_1 SOLUTION QUALITY  (n={args.n_test}, R_eval={R})")
    print(f"{'='*55}")
    print(f"  b_1 exact match   : {b1_match}/{args.n_test} "
          f"({b1_match/args.n_test*100:.1f}%)")
    print(f"  Mean gap          : {gaps.mean():.4f}%")
    print(f"  Median gap        : {np.median(gaps):.4f}%")
    print(f"  Max gap           : {gaps.max():.4f}%")
    print(f"  P90 gap           : {np.percentile(gaps, 90):.4f}%")
    print(f"  P95 gap           : {np.percentile(gaps, 95):.4f}%")
    print(f"  P99 gap           : {np.percentile(gaps, 99):.4f}%")
    print(f"  Samples > 0.5%    : {(gaps > 0.5).sum()}/{args.n_test}")
    print(f"  Samples > 1.0%    : {(gaps > 1.0).sum()}/{args.n_test}")


if __name__ == "__main__":
    main()
