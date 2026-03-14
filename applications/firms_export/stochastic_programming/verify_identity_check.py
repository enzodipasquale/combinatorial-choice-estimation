"""Verify: does the 'bundles is data.id_data["obs_bundles"]' identity check
actually work in the oracle?

If it FAILS for Q, then Q uses b_2_r_V instead of b_2_r_Q,
which would be a critical bug causing wrong gradients.
"""
import numpy as np
import combest as ce
from solver import TwoStageSolver
from oracles import build_oracles

M, K = 5, 5
R = 10
n_obs = 20
n_rev = 1
n_cov = n_rev + 2
theta_true = np.array([1.0, -5.0, 0.1])
seed_dgp = 42
seed_est = 43

rng = np.random.default_rng(seed_dgp)
rev_base = rng.uniform(0, 1.0, (n_rev, M))
rev_chars_1 = rev_base + rng.uniform(-0.1, 0.1, (n_rev, M))
rev_chars_2 = rev_base + rng.uniform(-0.1, 0.1, (n_rev, M))
state_chars = np.zeros((n_obs, M))
_raw = rng.uniform(0, 1, (M, M))
syn_chars = (_raw + _raw.T) / 2
np.fill_diagonal(syn_chars, 0)

# DGP
dgp = ce.Model()
dgp.load_config({"dimensions": {"n_obs": n_obs, "n_items": M, "n_covariates": n_cov},
                  "subproblem": {"gurobi_params": {"TimeLimit": 10}}})
dgp.data.load_and_distribute_input_data({
    "id_data": {"state_chars": state_chars, "capacity": np.full(n_obs, K)},
    "item_data": {"rev_chars_1": rev_chars_1, "rev_chars_2": rev_chars_2,
                  "syn_chars": syn_chars, "beta": 3, "R": R, "seed": seed_dgp}})
co, eo = build_oracles(dgp, seed=seed_dgp)
dgp.subproblems.load_solver(TwoStageSolver)
dgp.subproblems.initialize_solver()
dgp.features.set_covariates_oracle(co)
dgp.features.set_error_oracle(eo)
obs_b = dgp.subproblems.generate_obs_bundles(theta_true)

# Estimation model
model = ce.Model()
model.load_config({"dimensions": {"n_obs": n_obs, "n_items": M,
                                   "n_covariates": n_cov, "n_simulations": 1},
                    "subproblem": {"gurobi_params": {"TimeLimit": 10}}})
model.data.load_and_distribute_input_data({
    "id_data": {"state_chars": state_chars, "capacity": np.full(n_obs, K),
                "obs_bundles": obs_b},
    "item_data": {"rev_chars_1": rev_chars_1, "rev_chars_2": rev_chars_2,
                  "syn_chars": syn_chars, "beta": 3, "R": R, "seed": seed_est}})

# Monkey-patch the oracle to log identity checks
original_build = build_oracles

def patched_build(model, seed=42):
    co, eo = original_build(model, seed)
    ld = model.data.local_data

    def patched_get_b_2_r(bundles, ids, data):
        pol = data.id_data["policies"]
        is_obs = bundles is data.id_data["obs_bundles"]
        print(f"  _get_b_2_r: is_obs={is_obs}  "
              f"id(bundles)={id(bundles)}  id(obs_b)={id(data.id_data['obs_bundles'])}")
        if not is_obs:
            # Double check: are the VALUES equal even if identity fails?
            vals_match = np.array_equal(bundles, data.id_data["obs_bundles"])
            if vals_match:
                print(f"  *** WARNING: values match but identity fails! This is a BUG! ***")
        return pol["b_2_r_Q"][ids] if is_obs else pol["b_2_r_V"][ids]

    def patched_cov(bundles, ids, data):
        b_0 = data.id_data["state_chars"][ids]
        b_1 = bundles.astype(float)
        b_2_r = patched_get_b_2_r(bundles, ids, data).astype(float)
        beta = ld.item_data["beta"]
        R_val = ld.item_data["R"]
        rev_c1 = ld.item_data["rev_chars_1"]
        rev_c2 = ld.item_data["rev_chars_2"]
        C = ld.item_data["syn_chars"]
        x_rev = b_1 @ rev_c1.T + beta * np.einsum('nrm,km->nk', b_2_r, rev_c2) / R_val
        x_s = ((1 - b_0) * b_1).sum(-1) + beta * ((1 - b_1)[:, None, :] * b_2_r).sum(-1).mean(-1)
        x_c = np.einsum('nj,jk,nk->n', b_1, C, b_1) + beta * np.einsum('nrj,jk,nrk->n', b_2_r, C, b_2_r) / R_val
        return np.column_stack([x_rev, x_s, x_c])

    eps1 = ld.errors["eps1"]
    eps2 = ld.errors["eps2"]

    def patched_err(bundles, ids, data):
        b_1 = bundles.astype(float)
        e1 = (eps1[ids] * b_1).sum(-1)
        b_2_r = patched_get_b_2_r(bundles, ids, data).astype(float)
        beta = ld.item_data["beta"]
        e2 = (eps2[ids] * b_2_r).sum(-1).mean(-1)
        return e1 + beta * e2

    return patched_cov, patched_err

co2, eo2 = patched_build(model, seed=seed_est)
model.subproblems.load_solver(TwoStageSolver)
model.subproblems.initialize_solver()
model.features.set_covariates_oracle(co2)
model.features.set_error_oracle(eo2)

print("=== Computing gradient at theta_true ===")
print("Expected: 2 calls to _get_b_2_r per oracle call (cov + err)")
print("Expected: is_obs=False for V, is_obs=True for Q")
print()
f_val, g_val = model.point_estimation.compute_nonlinear_obj_and_grad_at_root(theta_true)
print()
print(f"f = {f_val:.6f}")
print(f"g = {g_val}")

# Also check: what is the object identity chain?
print("\n=== Object identity chain ===")
print(f"id(model.data.local_data.id_data['obs_bundles']) = {id(model.data.local_data.id_data['obs_bundles'])}")
print(f"id(model.data.local_obs_bundles) = {id(model.data.local_obs_bundles)}")
print(f"Same? {model.data.local_obs_bundles is model.data.local_data.id_data['obs_bundles']}")

# Check b_2_r_V vs b_2_r_Q difference
pol = model.data.local_data.id_data["policies"]
b2_diff = (pol["b_2_r_V"] != pol["b_2_r_Q"])
print(f"\nb_2_r_V != b_2_r_Q: {b2_diff.any()} (in {b2_diff.any(axis=(1,2)).sum()}/{n_obs} agents)")
