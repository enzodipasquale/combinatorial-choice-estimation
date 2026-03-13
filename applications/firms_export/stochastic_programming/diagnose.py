"""
DEEP DIAGNOSTIC for two-stage stochastic programming estimation.
Verifies every aspect of V and Q computations, especially the entry cost.
"""
import numpy as np
import combest as ce
from solver import TwoStageSolver
from oracles import build_oracles

# ── Setup (identical to test_bundle.py) ──────────────────────────────
beta = 4
M, K = 5, 3
R_dgp = 5
R_est = 5
S_est = 1
n_obs = 100
n_rev = 1
n_cov = n_rev + 2
theta_true = np.array([1.0] * n_rev + [-2.0, 0.3])
seed_dgp = 43
seed_est = 43

rng = np.random.default_rng(seed_dgp)
rev_chars = rng.uniform(0.5, 2.0, (n_rev, M))
state_chars = (rng.random((n_obs, M)) > 0.5).astype(float)
_raw = rng.uniform(0, 1, (M, M))
syn_chars = (_raw + _raw.T) / 2
np.fill_diagonal(syn_chars, 0)

input_data = {
    "id_data": {"state_chars": state_chars, "capacity": np.full(n_obs, K)},
    "item_data": {"rev_chars": rev_chars, "syn_chars": syn_chars,
                  "beta": beta, "R": R_dgp, "seed": seed_dgp},
}
cfg = {
    "dimensions": {"n_obs": n_obs, "n_items": M, "n_covariates": n_cov},
    "subproblem": {"gurobi_params": {"TimeLimit": 10}},
}

# DGP
dgp = ce.Model()
dgp.load_config(cfg)
dgp.data.load_and_distribute_input_data(input_data)
cov_oracle_dgp, err_oracle_dgp = build_oracles(dgp, seed=seed_dgp)
dgp.subproblems.load_solver(TwoStageSolver)
dgp.subproblems.initialize_solver()
dgp.features.set_covariates_oracle(cov_oracle_dgp)
dgp.features.set_error_oracle(err_oracle_dgp)
obs_b_dgp = dgp.subproblems.generate_obs_bundles(theta_true)

# Estimation
model = ce.Model()
cfg["dimensions"]["n_simulations"] = S_est
input_data["id_data"]["obs_bundles"] = obs_b_dgp
input_data["item_data"]["R"] = R_est
model.load_config(cfg)
model.data.load_and_distribute_input_data(input_data)
cov_oracle_est, err_oracle_est = build_oracles(model, seed=seed_est)
model.subproblems.load_solver(TwoStageSolver)
model.subproblems.initialize_solver()
model.features.set_covariates_oracle(cov_oracle_est)
model.features.set_error_oracle(err_oracle_est)

# ── Shorthand ────────────────────────────────────────────────────────
pt = model.point_estimation
ld = model.data.local_data
obs_b = ld.id_data["obs_bundles"]
b_0 = ld.id_data["state_chars"]
eps1 = ld.errors["eps1"]
eps2 = ld.errors["eps2"]
R = ld.item_data["R"]
C_mat = ld.item_data["syn_chars"]
rev = rev_chars.T  # (M, n_rev)
w = model.data.local_obs_quantity
is_root = model.comm_manager.is_root()


def manual_agent_utility(b1_i, b2r_i, b0_i, eps1_i, eps2_i, theta):
    """Single-agent utility from (b1, b2_r) and theta."""
    theta_rev, theta_s, theta_c = theta[:n_rev], theta[n_rev], theta[n_rev + 1]
    rv = rev @ theta_rev  # (M,)
    b1 = b1_i.astype(float)
    # Period 1
    u1 = np.sum(b1 * (rv + (1 - b0_i) * theta_s + eps1_i))
    u1 += theta_c * (b1 @ C_mat @ b1)
    # Period 2
    u2 = 0.0
    for r in range(R):
        b2 = b2r_i[r].astype(float)
        u2_r = np.sum(b2 * (rv + (1 - b1) * theta_s + eps2_i[r]))
        u2_r += theta_c * (b2 @ C_mat @ b2)
        u2 += u2_r / R
    return u1 + beta * u2


def manual_agent_covariates(b1_i, b2r_i, b0_i):
    """Single-agent covariates from (b1, b2_r)."""
    b1 = b1_i.astype(float)
    b2r = b2r_i.astype(float)  # (R, M)
    # Revenue
    x_rev = b1 @ rev + beta * np.einsum('rm,mk->k', b2r, rev) / R
    # Entry cost
    x_s_first = np.sum((1 - b0_i) * b1)
    x_s_second = beta * np.sum((1 - b1)[None, :] * b2r) / R
    x_s = x_s_first + x_s_second
    # Synergy
    x_c = b1 @ C_mat @ b1 + beta * np.einsum('rj,jk,rk->', b2r, C_mat, b2r) / R
    return np.concatenate([x_rev, [x_s, x_c]])


def manual_agent_errors(b1_i, b2r_i, eps1_i, eps2_i):
    """Single-agent errors from (b1, b2_r)."""
    b1 = b1_i.astype(float)
    b2r = b2r_i.astype(float)
    e1 = np.sum(eps1_i * b1)
    e2 = beta * np.sum(eps2_i * b2r, axis=-1).mean()
    return e1 + e2


def full_check(theta, label=""):
    """Full inspection at a given theta."""
    print(f"\n{'='*80}")
    print(f"  {label}  theta = {theta}")
    print(f"{'='*80}")

    # 1. Call the oracle (this runs the solver internally)
    f_val, g_val = pt.compute_nonlinear_obj_and_grad_at_root(theta)
    if not is_root:
        return

    pol = ld.id_data["policies"]
    b1s = pol["b_1_star"].copy()     # (n, M) bool
    b2v = pol["b_2_r_V"].copy()      # (n, R, M) bool
    b2q = pol["b_2_r_Q"].copy()      # (n, R, M) bool
    obs = obs_b.astype(float)

    # 2. Per-agent manual utility vs oracle decomposition
    V_man = np.zeros(n_obs)
    Q_man = np.zeros(n_obs)
    V_orc = np.zeros(n_obs)
    Q_orc = np.zeros(n_obs)
    cov_V_all = np.zeros((n_obs, n_cov))
    cov_Q_all = np.zeros((n_obs, n_cov))
    err_V_all = np.zeros(n_obs)
    err_Q_all = np.zeros(n_obs)

    for i in range(n_obs):
        V_man[i] = manual_agent_utility(b1s[i], b2v[i], b_0[i], eps1[i], eps2[i], theta)
        Q_man[i] = manual_agent_utility(obs_b[i], b2q[i], b_0[i], eps1[i], eps2[i], theta)
        cov_V_all[i] = manual_agent_covariates(b1s[i], b2v[i], b_0[i])
        cov_Q_all[i] = manual_agent_covariates(obs_b[i], b2q[i], b_0[i])
        err_V_all[i] = manual_agent_errors(b1s[i], b2v[i], eps1[i], eps2[i])
        err_Q_all[i] = manual_agent_errors(obs_b[i], b2q[i], eps1[i], eps2[i])
        V_orc[i] = cov_V_all[i] @ theta + err_V_all[i]
        Q_orc[i] = cov_Q_all[i] @ theta + err_Q_all[i]

    # 3. Check V_manual == V_oracle (covariates decomposition)
    V_diff = np.max(np.abs(V_man - V_orc))
    Q_diff = np.max(np.abs(Q_man - Q_orc))
    print(f"\n  ORACLE DECOMPOSITION CHECK:")
    print(f"    max|V_manual - V_oracle| = {V_diff:.2e}")
    print(f"    max|Q_manual - Q_oracle| = {Q_diff:.2e}")

    # 4. Aggregate objective
    f_man = np.sum(w * (V_man - Q_man))
    grad_man = np.sum(w[:, None] * (cov_V_all - cov_Q_all), axis=0)
    const_man = np.sum(w * (err_V_all - err_Q_all))
    f_from_grad = grad_man @ theta + const_man

    print(f"\n  OBJECTIVE:")
    print(f"    f (oracle call)          = {f_val:.10f}")
    print(f"    f (manual Σw*(V-Q))      = {f_man:.10f}")
    print(f"    f (grad@theta + const)   = {f_from_grad:.10f}")
    print(f"    |f_oracle - f_manual|    = {abs(f_val - f_man):.2e}")

    print(f"\n  GRADIENT:")
    print(f"    g (oracle)  = {g_val}")
    print(f"    g (manual)  = {grad_man}")
    print(f"    |diff|      = {np.max(np.abs(g_val - grad_man)):.2e}")

    # 5. Bundle match
    b1_match = np.all(b1s == obs_b.astype(bool))
    n_b1_diff = np.sum(np.any(b1s != obs_b.astype(bool), axis=1))
    b2_match = np.all(b2v == b2q)
    n_b2_diff = np.sum(np.any(b2v != b2q, axis=(1, 2)))
    print(f"\n  BUNDLE MATCH:")
    print(f"    b_1_star == obs_b:     {b1_match}  ({n_b1_diff} agents differ)")
    print(f"    b_2_r_V == b_2_r_Q:   {b2_match}  ({n_b2_diff} agents differ)")

    # 6. Entry cost feature detail
    xs_V = cov_V_all[:, n_rev]
    xs_Q = cov_Q_all[:, n_rev]
    print(f"\n  ENTRY COST FEATURE (x_s):")
    print(f"    Σw*x_s_V = {np.sum(w * xs_V):.6f}")
    print(f"    Σw*x_s_Q = {np.sum(w * xs_Q):.6f}")
    print(f"    grad_entry_cost = {grad_man[n_rev]:.6f}")
    for i in range(min(5, n_obs)):
        b1_str = ''.join(str(int(x)) for x in b1s[i])
        ob_str = ''.join(str(int(x)) for x in obs_b[i].astype(bool))
        same = "==" if b1_str == ob_str else "!="
        print(f"    Agent {i}: b1s={b1_str} {same} obs={ob_str}  "
              f"x_s_V={xs_V[i]:.4f}  x_s_Q={xs_Q[i]:.4f}  diff={xs_V[i]-xs_Q[i]:.4f}")

    # 7. V >= Q check (V is the max, so V_i >= Q_i for each agent?)
    # NOTE: V_i is from optimal b_1_star + re-optimized b_2_r_V
    # Q_i is from obs_b + re-optimized b_2_r_Q
    # V_i >= Q_i should hold because b_1_star is optimal
    # BUT: only if b_2_r_V is truly optimal given b_1_star
    violations = np.sum(V_man < Q_man - 1e-8)
    if violations > 0:
        worst_idx = np.argmin(V_man - Q_man)
        print(f"\n  *** WARNING: {violations} agents have V < Q! ***")
        print(f"    Worst agent {worst_idx}: V={V_man[worst_idx]:.6f} Q={Q_man[worst_idx]:.6f} diff={V_man[worst_idx]-Q_man[worst_idx]:.6f}")
    else:
        print(f"\n  V >= Q check: PASS (all agents)")

    return f_val, g_val


# ══════════════════════════════════════════════════════════════════════
print("\n" + "#" * 80)
print("  DEEP DIAGNOSTIC: Two-Stage Stochastic Programming")
print(f"  theta_true = {theta_true}")
print(f"  n_obs={n_obs}  M={M}  K={K}  R={R_est}  beta={beta}")
print(f"  seed_dgp={seed_dgp}  seed_est={seed_est}")
print("#" * 80)

# TEST 1: At theta_true
full_check(theta_true, "TEST 1: theta_true (SHOULD BE f=0, grad=0)")

# TEST 2: At zero
full_check(np.zeros(n_cov), "TEST 2: theta = [0, 0, 0]")

# TEST 3: At midpoint
full_check(theta_true * 0.5, "TEST 3: theta = theta_true/2")

# TEST 4: With entry cost only changed
theta_no_entry = theta_true.copy()
theta_no_entry[n_rev] = 0.0  # zero out entry cost
full_check(theta_no_entry, "TEST 4: theta_true but theta_s=0")

# TEST 5: At [0, -2, 0] — only entry cost
theta_entry_only = np.zeros(n_cov)
theta_entry_only[n_rev] = -2.0
full_check(theta_entry_only, "TEST 5: only theta_s=-2")

# ══════════════════════════════════════════════════════════════════════
# TEST 6: Line search theta0 → theta_true
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("  TEST 6: f along line from [0,0,0] to theta_true")
print(f"{'='*80}")
if is_root:
    print(f"  {'alpha':>6} {'f':>12} {'|g|':>12} {'g·dir':>12}")
    print("  " + "-" * 50)
direction = theta_true / np.linalg.norm(theta_true)
for alpha in np.linspace(0, 1, 11):
    theta_t = alpha * theta_true
    f_v, g_v = pt.compute_nonlinear_obj_and_grad_at_root(theta_t)
    if is_root:
        dot = g_v @ direction
        print(f"  {alpha:6.2f}  {f_v:12.6f}  {np.linalg.norm(g_v):12.6f}  {dot:12.6f}")

# ══════════════════════════════════════════════════════════════════════
# TEST 7: Finite difference gradient at multiple points
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("  TEST 7: Finite difference gradient check")
print(f"{'='*80}")

test_thetas = [
    ("theta_true/2", theta_true * 0.5),
    ("theta_true*0.8", theta_true * 0.8),
    ("[0.5, -1.0, 0.1]", np.array([0.5, -1.0, 0.1])),
    ("[1.5, -1.5, 0.2]", np.array([1.5, -1.5, 0.2])),
]

for label, theta_t in test_thetas:
    f0, g0 = pt.compute_nonlinear_obj_and_grad_at_root(theta_t)
    if is_root:
        h = 1e-4
        fd = np.zeros(n_cov)
        for j in range(n_cov):
            tp = theta_t.copy(); tp[j] += h
            tm = theta_t.copy(); tm[j] -= h
            fp, _ = pt.compute_nonlinear_obj_and_grad_at_root(tp)
            fm, _ = pt.compute_nonlinear_obj_and_grad_at_root(tm)
            fd[j] = (fp - fm) / (2 * h)
        names = [f"θ_rev{i}" for i in range(n_rev)] + ["θ_s", "θ_c"]
        print(f"\n  At {label}:")
        print(f"    f = {f0:.8f}")
        for j in range(n_cov):
            print(f"    {names[j]:>6}: analytic={g0[j]:+12.6f}  FD={fd[j]:+12.6f}  diff={g0[j]-fd[j]:+10.6f}")

# ══════════════════════════════════════════════════════════════════════
# TEST 8: Simple steepest descent from theta0
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("  TEST 8: Steepest descent with Armijo line search (100 iters)")
print(f"{'='*80}")

theta_sd = np.zeros(n_cov)
if is_root:
    print(f"  {'it':>4} {'f':>12} {'|g|':>10} {'step':>10} {'err':>10}  theta")
for it in range(100):
    f_v, g_v = pt.compute_nonlinear_obj_and_grad_at_root(theta_sd)
    if is_root:
        gnorm = np.linalg.norm(g_v)
        err = np.linalg.norm(theta_sd - theta_true)
        if it % 10 == 0 or it < 5 or gnorm < 1e-6:
            print(f"  {it:4d}  {f_v:12.6f}  {gnorm:10.4f}  {'':>10}  {err:10.4f}  {np.array2string(theta_sd, precision=4)}")
        if gnorm < 1e-8:
            print(f"  CONVERGED at iter {it}")
            break
        # Armijo line search along -gradient
        d = -g_v
        alpha = 1.0 / gnorm  # initial step: unit-length step
        c1 = 1e-4
        best_f = f_v
        best_alpha = 0.0
        for ls in range(20):
            theta_try = theta_sd + alpha * d
            f_try, _ = pt.compute_nonlinear_obj_and_grad_at_root(theta_try)
            if f_try < best_f:
                best_f = f_try
                best_alpha = alpha
            if f_try <= f_v + c1 * alpha * (g_v @ d):
                break
            alpha *= 0.5
        if best_alpha > 0:
            theta_sd = theta_sd + best_alpha * d
        else:
            # Fallback: tiny step
            theta_sd = theta_sd + 1e-4 * d / gnorm
    theta_sd = model.comm_manager.Bcast(theta_sd)

f_fin, g_fin = pt.compute_nonlinear_obj_and_grad_at_root(theta_sd)
if is_root:
    print(f"\n  Final: f={f_fin:.10f}  |g|={np.linalg.norm(g_fin):.10f}")
    print(f"  theta_sd   = {theta_sd}")
    print(f"  theta_true = {theta_true}")
    print(f"  error      = {np.linalg.norm(theta_sd - theta_true):.6f}")

# ══════════════════════════════════════════════════════════════════════
# TEST 9: Verify V >= Q at theta != theta_true
# (the solver should always find V >= Q because b_1_star is optimal)
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("  TEST 9: Check V_i >= Q_i for all agents at various theta")
print(f"{'='*80}")

for label, theta_t in [("theta=0", np.zeros(n_cov)),
                        ("theta_true/2", theta_true * 0.5),
                        ("theta=[-1, 3, -0.5]", np.array([-1.0, 3.0, -0.5])),
                        ("theta_true", theta_true)]:
    # Run solver
    pt.compute_nonlinear_obj_and_grad_at_root(theta_t)
    if is_root:
        pol = ld.id_data["policies"]
        b1s = pol["b_1_star"]
        b2v = pol["b_2_r_V"]
        b2q = pol["b_2_r_Q"]
        V = np.array([manual_agent_utility(b1s[i], b2v[i], b_0[i], eps1[i], eps2[i], theta_t) for i in range(n_obs)])
        Q = np.array([manual_agent_utility(obs_b[i], b2q[i], b_0[i], eps1[i], eps2[i], theta_t) for i in range(n_obs)])
        viols = np.sum(V < Q - 1e-8)
        worst = np.min(V - Q)
        print(f"  {label:>25}: violations={viols}/{n_obs}  min(V-Q)={worst:+.6f}  f={np.sum(w*(V-Q)):.6f}")
        if viols > 0:
            for i in np.where(V < Q - 1e-8)[0][:5]:
                b1_str = ''.join(str(int(x)) for x in b1s[i])
                ob_str = ''.join(str(int(x)) for x in obs_b[i].astype(bool))
                print(f"    Agent {i}: V={V[i]:.4f} Q={Q[i]:.4f} b1s={b1_str} obs={ob_str}")

print("\n" + "#" * 80)
print("  DIAGNOSTIC COMPLETE")
print("#" * 80)
