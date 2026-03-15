import argparse
import time
import numpy as np
import gurobipy as gp
import torch
from neur2sp.net2mip import compute_layer_bounds, embed_relu


def solve_exact(b_1_var, b_2_r_vars, model, b_0, eps1, eff_rev, eps_2,
                entry, entry_2, syn_1, syn_2, rev1, R, M):
    b_1, b_2_r = b_1_var, b_2_r_vars

    mod_1 = rev1 + (1 - b_0) * entry + eps1
    obj = mod_1 @ b_1 + b_1 @ syn_1 @ b_1

    for r in range(R):
        mod_2_r = eff_rev + eps_2[r] + (1 - b_1) * entry_2
        obj += (1 / R) * (mod_2_r @ b_2_r[r, :]
                          + b_2_r[r, :] @ syn_2 @ b_2_r[r, :])

    model.setObjective(obj, gp.GRB.MAXIMIZE)
    model.optimize()
    return np.array(b_1_var.X) > 0.5, model.ObjVal


def eval_true_obj(b_1, b_0, eps1, eff_rev, eps_2,
                  entry, entry_2, syn_1, syn_2, rev1,
                  R, M, q_model, q_var):
    b1f = b_1.astype(float)
    mod_1 = rev1 + (1 - b_0) * entry + eps1
    v1 = mod_1 @ b1f + b1f @ syn_1 @ b1f

    v2_total = 0.0
    for r in range(R):
        c_r = eff_rev + eps_2[r] + (1 - b1f) * entry_2
        q_model.setObjective(c_r @ q_var + q_var @ syn_2 @ q_var,
                             gp.GRB.MAXIMIZE)
        q_model.optimize()
        v2_total += q_model.ObjVal

    return v1 + v2_total / R


def solve_surrogate(b_0, eps1, eff_rev, entry, syn_1, rev1,
                    nn_weights, nn_biases, pre_bounds,
                    theta_s, theta_sc, theta_c, M):
    m = gp.Model("surr")
    m.Params.OutputFlag = 0
    m.Params.Threads = 1

    b_1 = m.addMVar(M, vtype=gp.GRB.BINARY, name="b_1")
    eff_vars = [m.addVar(lb=float(eff_rev[j]), ub=float(eff_rev[j]),
                         name=f"eff_{j}") for j in range(M)]
    tvars = [m.addVar(lb=float(v), ub=float(v), name=f"th_{t}")
             for t, v in enumerate([theta_s, theta_sc, theta_c])]
    m.update()

    nn_in = list(b_1.tolist()) + eff_vars + tvars
    nn_out = embed_relu(m, nn_weights, nn_biases, nn_in, pre_bounds)

    mod_1 = rev1 + (1 - b_0) * entry + eps1
    m.setObjective(mod_1 @ b_1 + b_1 @ syn_1 @ b_1 + nn_out,
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
    M = int(ckpt["M"])
    beta = float(ckpt["beta"])
    beta_perpetual = float(ckpt["beta_perpetual"])
    sigma_nu_2 = float(ckpt["sigma_nu_2"])
    input_lb, input_ub = ckpt["input_lb"], ckpt["input_ub"]
    pre_bounds = compute_layer_bounds(nn_weights, nn_biases, input_lb, input_ub)

    rng_c = np.random.default_rng(args.seed_chars)
    entry_chars = rng_c.uniform(0, 1, M)
    _raw = rng_c.uniform(0, 1, (M, M))
    syn_chars = (_raw + _raw.T) / 2
    np.fill_diagonal(syn_chars, 0)

    R = args.R_eval
    rng = np.random.default_rng(args.seed_test)
    perpetual = 1 / (1 - beta)
    theta_bounds_s = ckpt["theta_bounds_s"]
    theta_bounds_sc = ckpt["theta_bounds_sc"]
    theta_bounds_c = ckpt["theta_bounds_c"]
    eff_rev_bounds = ckpt["eff_rev_bounds"]

    print(f"Validating b_1 quality: M={M}, R_eval={R}, n_test={args.n_test}")

    m_ex = gp.Model("exact")
    m_ex.Params.OutputFlag = 0
    m_ex.Params.Threads = 1
    b1_ex = m_ex.addMVar(M, vtype=gp.GRB.BINARY, name="b1")
    b2r_ex = m_ex.addMVar((R, M), vtype=gp.GRB.BINARY, name="b2r")
    m_ex.update()

    m_q = gp.Model("q")
    m_q.Params.OutputFlag = 0
    m_q.Params.Threads = 1
    b2_q = m_q.addMVar(M, vtype=gp.GRB.BINARY, name="b2")
    m_q.update()

    gaps, b1_match = [], 0
    t0 = time.time()

    for t in range(args.n_test):
        theta_rev = rng.uniform(-5.0, 5.0, size=1)
        theta_s = rng.uniform(*theta_bounds_s)
        theta_sc = rng.uniform(*theta_bounds_sc)
        theta_c = rng.uniform(*theta_bounds_c)

        b_0 = (rng.random(M) > 0.9).astype(float)
        rev_chars_1 = rng.uniform(0, 2, (1, M))
        rev_chars_2 = rng.uniform(0, 2, (1, M))
        eps_perm = rng.normal(0, 1, M)
        nu1 = rng.normal(0, 0.5, M)
        nu2 = rng.normal(0, sigma_nu_2, (R, M))

        eps1 = eps_perm + nu1
        rev1 = (rev_chars_1 * theta_rev).sum(0)  # (M,)
        rev2_d = beta * perpetual * (rev_chars_2 * theta_rev).sum(0)
        eff_rev = rev2_d + beta * perpetual * eps_perm
        eps_2_transient = beta * nu2  # (R, M)

        entry = theta_s + theta_sc * entry_chars
        entry_2 = beta * entry
        syn_1 = theta_c * syn_chars
        syn_2 = theta_c * beta * perpetual * syn_chars

        b1_exact, obj_exact = solve_exact(
            b1_ex, b2r_ex, m_ex, b_0, eps1, eff_rev, eps_2_transient,
            entry, entry_2, syn_1, syn_2, rev1, R, M)

        b1_nn = solve_surrogate(
            b_0, eps1, eff_rev, entry, syn_1, rev1,
            nn_weights, nn_biases, pre_bounds,
            theta_s, theta_sc, theta_c, M)

        if np.array_equal(b1_exact, b1_nn):
            obj_nn = obj_exact
            b1_match += 1
        else:
            obj_nn = eval_true_obj(
                b1_nn, b_0, eps1, eff_rev, eps_2_transient,
                entry, entry_2, syn_1, syn_2, rev1,
                R, M, m_q, b2_q)

        gap = ((obj_exact - obj_nn) / abs(obj_exact) * 100
               if abs(obj_exact) > 1e-6 else 0.0)
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


if __name__ == "__main__":
    main()
