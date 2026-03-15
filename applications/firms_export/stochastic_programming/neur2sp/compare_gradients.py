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


def covariates(b_1, b_2_r, b_0, rev_chars_1, rev_chars_2, syn_chars,
               beta, R, n_rev):
    C = syn_chars
    b1f = b_1.astype(float)
    b2f = b_2_r.astype(float)
    x_rev = b1f @ rev_chars_1.T + beta * np.einsum('rm,km->k', b2f, rev_chars_2) / R
    x_s = ((1 - b_0) * b1f).sum() + beta * ((1 - b1f)[None, :] * b2f).sum(-1).mean()
    x_c = b1f @ C @ b1f + beta * np.einsum('rj,jk,rk->', b2f, C, b2f) / R
    return np.concatenate([x_rev, [x_s, x_c]])


def errors(b_1, b_2_r, eps1, eps2, beta):
    b1f = b_1.astype(float)
    b2f = b_2_r.astype(float)
    e1 = (eps1 * b1f).sum()
    e2 = (eps2 * b2f).sum(-1).mean()
    return e1 + beta * e2


def solve_exact_joint(b1_var, b2r_var, model, theta, rev_chars_1, rev_chars_2,
                      syn_chars, b_0, eps1, eps2, beta, R, M):
    n_rev = rev_chars_1.shape[0]
    theta_rev, theta_s, theta_c = theta[:n_rev], theta[n_rev], theta[n_rev + 1]
    C, bR = syn_chars, beta / R

    mod_1 = rev_chars_1.T @ theta_rev + (1 - b_0) * theta_s + eps1
    rev2 = rev_chars_2.T @ theta_rev
    obj = mod_1 @ b1_var + theta_c * (b1_var @ C @ b1_var)
    for r in range(R):
        mod_2 = rev2 + (1 - b1_var) * theta_s + eps2[r]
        obj += bR * (mod_2 @ b2r_var[r, :] +
                     theta_c * (b2r_var[r, :] @ C @ b2r_var[r, :]))
    model.setObjective(obj, gp.GRB.MAXIMIZE)
    model.optimize()
    return np.array(b1_var.X) > 0.5, np.array(b2r_var.X) > 0.5


def solve_b2_given_b1(b1, theta, rev_chars_2, syn_chars, eps2, R, M,
                      q_model, q_var):
    n_rev = rev_chars_2.shape[0]
    theta_rev, theta_s, theta_c = theta[:n_rev], theta[n_rev], theta[n_rev + 1]
    C = syn_chars
    b1f = b1.astype(float)
    rev2 = rev_chars_2.T @ theta_rev
    b2_r = np.zeros((R, M), dtype=bool)
    for r in range(R):
        c_r = rev2 + (1 - b1f) * theta_s + eps2[r]
        q_model.setObjective(c_r @ q_var + theta_c * (q_var @ C @ q_var),
                             gp.GRB.MAXIMIZE)
        q_model.optimize()
        b2_r[r] = np.array(q_var.X) > 0.5
    return b2_r


def solve_surrogate(theta, rev_chars_1, syn_chars, b_0, eps1,
                    nn_weights, nn_biases, pre_bounds, M):
    n_rev = rev_chars_1.shape[0]
    theta_rev, theta_s, theta_c = theta[:n_rev], theta[n_rev], theta[n_rev + 1]
    C = syn_chars

    m = gp.Model("surr")
    m.Params.OutputFlag = 0
    m.Params.Threads = 1
    m.ModelSense = gp.GRB.MAXIMIZE

    b_1 = m.addMVar(M, vtype=gp.GRB.BINARY, name="b_1")
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
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--R_eval", type=int, default=100)
    ap.add_argument("--seed_chars", type=int, default=42)
    ap.add_argument("--seed_test", type=int, default=555)
    args = ap.parse_args()

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    nn_weights, nn_biases = ckpt["weights"], ckpt["biases"]
    M = int(ckpt["M"])
    n_rev, beta = int(ckpt["n_rev"]), float(ckpt["beta"])
    theta_lb, theta_ub = ckpt["theta_lb"], ckpt["theta_ub"]
    n_cov = n_rev + 2

    input_lb = np.concatenate([np.zeros(M), theta_lb])
    input_ub = np.concatenate([np.ones(M), theta_ub])
    pre_bounds = compute_layer_bounds(nn_weights, nn_biases, input_lb, input_ub)

    rev_chars_1, rev_chars_2, syn_chars = _make_chars(M, n_rev, args.seed_chars)
    R = args.R_eval
    rng = np.random.default_rng(args.seed_test)

    print(f"Exact joint MIP vs NN surrogate + exact second-stage")
    print(f"M={M}, R={R}, n_test={args.n_test}")
    print(f"Covariates: {n_cov} (n_rev={n_rev}, theta_s, theta_c)\n")

    # reusable Gurobi models
    m_ex = gp.Model("exact")
    m_ex.Params.OutputFlag = 0
    m_ex.Params.Threads = 1
    m_ex.ModelSense = gp.GRB.MAXIMIZE
    b1_ex = m_ex.addMVar(M, vtype=gp.GRB.BINARY, name="b1")
    b2r_ex = m_ex.addMVar((R, M), vtype=gp.GRB.BINARY, name="b2r")
    m_ex.update()

    m_q = gp.Model("q")
    m_q.Params.OutputFlag = 0
    m_q.Params.Threads = 1
    m_q.ModelSense = gp.GRB.MAXIMIZE
    b2_q = m_q.addMVar(M, vtype=gp.GRB.BINARY, name="b2")
    m_q.update()

    # second q-model for NN method (so we don't interfere)
    m_q2 = gp.Model("q2")
    m_q2.Params.OutputFlag = 0
    m_q2.Params.Threads = 1
    m_q2.ModelSense = gp.GRB.MAXIMIZE
    b2_q2 = m_q2.addMVar(M, vtype=gp.GRB.BINARY, name="b2")
    m_q2.update()

    cosine_sims_match, cosine_sims_miss = [], []
    abs_errors_match, abs_errors_miss = [], []
    obj_gaps = []
    grad_exact_all, grad_nn_all = [], []
    b1_matches = 0
    time_exact_b1, time_nn_b1 = 0.0, 0.0
    time_exact_knap, time_nn_knap = 0.0, 0.0
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
        b_1_obs = rng.integers(0, 2, size=M).astype(bool)

        # ── EXACT: joint MIP → b_1*, b_2_r* ──
        t1 = time.time()
        b1_exact, b2r_exact = solve_exact_joint(
            b1_ex, b2r_ex, m_ex, theta,
            rev_chars_1, rev_chars_2, syn_chars, b_0, eps1, eps2, beta, R, M)
        time_exact_b1 += time.time() - t1

        t1 = time.time()
        b2r_obs_ex = solve_b2_given_b1(b_1_obs, theta, rev_chars_2, syn_chars,
                                       eps2, R, M, m_q, b2_q)
        time_exact_knap += time.time() - t1

        cov_V_ex = covariates(b1_exact, b2r_exact, b_0,
                              rev_chars_1, rev_chars_2, syn_chars, beta, R, n_rev)
        cov_Q_ex = covariates(b_1_obs, b2r_obs_ex, b_0,
                              rev_chars_1, rev_chars_2, syn_chars, beta, R, n_rev)
        err_V_ex = errors(b1_exact, b2r_exact, eps1, eps2, beta)
        err_Q_ex = errors(b_1_obs, b2r_obs_ex, eps1, eps2, beta)

        grad_exact = cov_V_ex - cov_Q_ex
        obj_exact = grad_exact @ theta + (err_V_ex - err_Q_ex)

        # ── NN HYBRID: surrogate MIP → b_1_nn, then exact second-stage ──
        t1 = time.time()
        b1_nn = solve_surrogate(theta, rev_chars_1, syn_chars, b_0, eps1,
                                nn_weights, nn_biases, pre_bounds, M)
        time_nn_b1 += time.time() - t1

        t1 = time.time()
        b2r_V_nn = solve_b2_given_b1(b1_nn, theta, rev_chars_2, syn_chars,
                                     eps2, R, M, m_q2, b2_q2)
        b2r_Q_nn = solve_b2_given_b1(b_1_obs, theta, rev_chars_2, syn_chars,
                                     eps2, R, M, m_q2, b2_q2)
        time_nn_knap += time.time() - t1

        cov_V_nn = covariates(b1_nn, b2r_V_nn, b_0,
                              rev_chars_1, rev_chars_2, syn_chars, beta, R, n_rev)
        cov_Q_nn = covariates(b_1_obs, b2r_Q_nn, b_0,
                              rev_chars_1, rev_chars_2, syn_chars, beta, R, n_rev)
        err_V_nn = errors(b1_nn, b2r_V_nn, eps1, eps2, beta)
        err_Q_nn = errors(b_1_obs, b2r_Q_nn, eps1, eps2, beta)

        grad_nn = cov_V_nn - cov_Q_nn
        obj_nn = grad_nn @ theta + (err_V_nn - err_Q_nn)

        # ── Compare ──
        b1_match = np.array_equal(b1_exact, b1_nn)
        if b1_match:
            b1_matches += 1

        grad_exact_all.append(grad_exact)
        grad_nn_all.append(grad_nn)
        abs_err = np.abs(grad_exact - grad_nn)

        norm_ex = np.linalg.norm(grad_exact)
        norm_nn = np.linalg.norm(grad_nn)
        if norm_ex > 1e-10 and norm_nn > 1e-10:
            cos = np.dot(grad_exact, grad_nn) / (norm_ex * norm_nn)
            if b1_match:
                cosine_sims_match.append(cos)
                abs_errors_match.append(abs_err)
            else:
                cosine_sims_miss.append(cos)
                abs_errors_miss.append(abs_err)

        if abs(obj_exact) > 1e-6:
            obj_gaps.append(abs(obj_exact - obj_nn) / abs(obj_exact) * 100)

        if (t + 1) % 50 == 0:
            all_cos = cosine_sims_match + cosine_sims_miss
            cos_mean = np.mean(all_cos) if all_cos else float('nan')
            print(f"  [{t+1}/{args.n_test}]  {time.time()-t0:.0f}s  "
                  f"cos_sim={cos_mean:.4f}  "
                  f"b1_match={b1_matches}/{t+1}  "
                  f"mean_obj_gap={np.mean(obj_gaps):.2f}%")

    # ── Report ──
    grad_exact_all = np.array(grad_exact_all)
    grad_nn_all = np.array(grad_nn_all)
    obj_gaps = np.array(obj_gaps)
    labels = [f"theta_rev_{j}" for j in range(n_rev)] + ["theta_s", "theta_c"]

    print(f"\n{'='*60}")
    print(f"GRADIENT COMPARISON  (n={args.n_test}, R={R})")
    print(f"Exact joint MIP  vs  NN surrogate + exact second-stage")
    print(f"{'='*60}")
    print(f"\nb_1 match: {b1_matches}/{args.n_test} "
          f"({b1_matches/args.n_test*100:.1f}%)")

    for label, cos_list, err_list in [
        ("b_1 MATCH", cosine_sims_match, abs_errors_match),
        ("b_1 MISMATCH", cosine_sims_miss, abs_errors_miss),
        ("ALL", cosine_sims_match + cosine_sims_miss,
         abs_errors_match + abs_errors_miss),
    ]:
        if not cos_list:
            continue
        cos = np.array(cos_list)
        errs = np.array(err_list)
        print(f"\n── {label} (n={len(cos_list)}) ──")
        print(f"  Cosine sim:  mean={cos.mean():.4f}  "
              f"median={np.median(cos):.4f}  "
              f"min={cos.min():.4f}  P5={np.percentile(cos, 5):.4f}")
        print(f"  Abs error per component:")
        for j, lab in enumerate(labels):
            print(f"    {lab:15s}  mean={errs[:, j].mean():.4f}  "
                  f"median={np.median(errs[:, j]):.4f}  "
                  f"P95={np.percentile(errs[:, j], 95):.4f}")

    print(f"\nPer-component mean gradient:")
    for j, lab in enumerate(labels):
        print(f"  {lab:15s}  exact={grad_exact_all[:, j].mean():8.4f}  "
              f"nn={grad_nn_all[:, j].mean():8.4f}")

    print(f"\nObjective gap:")
    print(f"  Mean={obj_gaps.mean():.2f}%  Median={np.median(obj_gaps):.2f}%  "
          f"P95={np.percentile(obj_gaps, 95):.2f}%  Max={obj_gaps.max():.2f}%")

    n = args.n_test
    print(f"\nTiming (per solve):")
    print(f"  EXACT:  joint MIP = {time_exact_b1/n*1000:.1f}ms  "
          f"Q second-stage = {time_exact_knap/n*1000:.1f}ms  "
          f"total = {(time_exact_b1+time_exact_knap)/n*1000:.1f}ms")
    print(f"  NN:     surr MIP  = {time_nn_b1/n*1000:.1f}ms  "
          f"V+Q second-stage = {time_nn_knap/n*1000:.1f}ms  "
          f"total = {(time_nn_b1+time_nn_knap)/n*1000:.1f}ms")
    print(f"  Speedup: {(time_exact_b1+time_exact_knap)/(time_nn_b1+time_nn_knap):.2f}x")


if __name__ == "__main__":
    main()
