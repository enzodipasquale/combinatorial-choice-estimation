import numpy as np
import combest as ce
from solver import TwoStageSolver
from oracles import build_oracles, static_covariates

M, R, beta, K = 5, 50, 0.9, 3
n_obs, seed = 30, 42
rng = np.random.default_rng(seed)

theta_true = np.array([1.0, -0.5])
revenue = rng.uniform(0.5, 2.0, M)
state = (rng.random((n_obs, M)) > 0.5).astype(float)

model = ce.Model()
model.load_config({
    "dimensions": {"n_obs": n_obs, "n_items": M, "n_covariates": 2, "n_simulations": 1},
    "subproblem": {"gurobi_params": {"TimeLimit": 10}},
})
model.data.load_and_distribute_input_data({
    "id_data": {"obs_bundles": np.zeros((n_obs, M), dtype=bool),
                "state": state, "capacity": np.full(n_obs, K)},
    "item_data": {"revenue": revenue, "beta": beta, "R": R, "seed": seed},
})

cov_oracle, err_oracle = build_oracles(model, seed)
solver = model.subproblems.load_solver(TwoStageSolver)
model.subproblems.initialize_solver()
model.features.set_covariates_oracle(cov_oracle)
model.features.set_error_oracle(err_oracle)

obs_b = solver.solve(theta_true).copy()
solver.obs_b = obs_b.astype(float)
model.data.local_data["id_data"]["obs_bundles"] = obs_b

# at theta_true, V=Q so obj=0, grad=0
obj, grad = model.point_estimation.compute_nonlinear_obj_and_grad_at_root(theta_true)
print(f"At theta_true: obj={obj:.6f}  grad={grad}")
assert np.isclose(obj, 0), f"obj should be 0 at theta_true, got {obj}"
assert np.allclose(grad, 0), f"grad should be 0 at theta_true, got {grad}"

# at a different theta, obj > 0
theta_off = np.array([1.5, -0.2])
obj, grad = model.point_estimation.compute_nonlinear_obj_and_grad_at_root(theta_off)
print(f"At theta_off:  obj={obj:.4f}  grad={grad}")
assert obj >= -1e-8, f"obj should be >= 0, got {obj}"

print("All tests passed.")
