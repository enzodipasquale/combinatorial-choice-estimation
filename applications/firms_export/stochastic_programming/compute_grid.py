import sys
import numpy as np
import combest as ce
from solver import TwoStageSolver
from oracles import build_oracles

beta = float(sys.argv[1]) if len(sys.argv) > 1 else 0.9
out = sys.argv[2] if len(sys.argv) > 2 else "grid_results.npz"

M, R, K = 5, 50, 3
n_obs, seed = 20, 42
rng = np.random.default_rng(seed)

theta_true = np.array([1.0, -0.5])
revenue = rng.uniform(0.5, 2.0, M)
state = (rng.random((n_obs, M)) > 0.5).astype(float)

model = ce.Model()
model.load_config({
    "dimensions": {"n_obs": n_obs, "n_items": M, "n_covariates": 2, "n_simulations": 10},
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

is_root = model.comm_manager.is_root()
N = 30
t1 = np.linspace(-2, 2, N)
t2 = np.linspace(-2, 2, N)
obj_grid = np.zeros((N, N)) if is_root else None

if is_root:
    print(f"beta={beta}, grid {N}x{N} on [{t1[0]},{t1[-1]}]x[{t2[0]},{t2[-1]}]")

for j in range(N):
    for k in range(N):
        obj, _ = model.point_estimation.compute_nonlinear_obj_and_grad_at_root(
            np.array([t1[j], t2[k]])
        )
        if is_root:
            obj_grid[k, j] = obj
    if is_root:
        print(f"theta_0={t1[j]:.2f} done ({j+1}/{N})")

obj_true, grad_true = model.point_estimation.compute_nonlinear_obj_and_grad_at_root(theta_true)

if is_root:
    np.savez(out, t1=t1, t2=t2, obj=obj_grid, theta_true=theta_true, beta=beta,
             obj_true=obj_true)
    ij = np.unravel_index(obj_grid.argmin(), obj_grid.shape)
    print(f"At theta_true: obj={obj_true:.6f}  grad={grad_true}")
    print(f"Saved {out} | grid min at ({t1[ij[1]]:.2f}, {t2[ij[0]]:.2f}) obj={obj_grid.min():.4f}")
