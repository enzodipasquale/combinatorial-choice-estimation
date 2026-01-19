import bundlechoice as bc
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

n_agents, n_items = 40, 10
k_mod, k_quad = 3, 1
n_features = k_mod + k_quad
theta_star = np.array([1.0, 1, 1, 5])

if rank == 0:
    cfg = {'dimensions': {'n_obs': n_agents, 'n_items': n_items, 'n_features': n_features},
           'row_generation': {'max_iters': 100, 'tol_row_generation': 1e-8,
                              'theta_lbs': [-10, -10, -10, -10]}}
    
    np.random.seed(42)
    weights = np.ones(n_items)
    capacity = np.random.randint(1, weights.sum(), n_agents)
    
    modular_agent = np.random.randn(n_agents, n_items, k_mod)
    quadratic_item = np.random.randn(n_items, n_items, k_quad) * 0.1
    for k in range(k_quad):
        np.fill_diagonal(quadratic_item[:, :, k], 0)
        quadratic_item[:, :, k] = (quadratic_item[:, :, k] + quadratic_item[:, :, k].T) / 2
    
    input_data = {
        "id_data": {"modular": modular_agent, "capacity": capacity},
        "item_data": {"quadratic": quadratic_item, "weights": weights}
    }
else:
    cfg = None
    input_data = None

bc = bc.BundleChoice()
bc.load_config(cfg)
bc.data.load_input_data(input_data)
bc.oracles.build_quadratic_features_from_data()
bc.oracles.build_local_modular_error_oracle(seed=42)

# Generate obs_bundles by solving subproblems with theta_star
bc.subproblems.load('QuadraticKnapsackGRB')
bc.subproblems.generate_obs_bundles(theta_star)

# subproblem = bc.subproblems.load('QuadraticKnapsackGRB')
# subproblem.initialize()
# obs_bundles_local = subproblem.solve(theta_star)
# bc.data.local_data["id_data"]["obs_bundles"]  = obs_bundles_local
# bc.data.local_obs_bundles = obs_bundles_local.astype(bool)


# # Estimation via row generation
# bc.oracles.build_local_modular_error_oracle(seed=47)
# result = bc.row_generation.solve(verbose=True)
# bounds_info = bc.row_generation._check_bounds_hit()

# if rank == 0:
#     print("Theta star:", theta_star)
#     print("Bounds info:", bounds_info)
#     print("--------------------------------")


# # Test update_objective_for_weights works
# if rank == 0:
#     print("\n=== Testing objective weight update ===")
#     theta_before = bc.row_generation.master_variables[0].X.copy()
    
#     # Random weights (different from uniform)
#     np.random.seed(999)
#     weights = np.random.exponential(1.0, n_agents)
#     weights = weights / weights.sum()
# else:
#     weights = None


# local_weights = bc.comm_manager.Scatterv_by_row(weights, row_counts=bc.data.agent_counts)
# bc.row_generation.update_objective_for_weights(local_weights)
# if rank == 0:
#     bc.row_generation.master_model.optimize()
#     theta_after = bc.row_generation.master_variables[0].X
    
#     diff = np.abs(theta_after - theta_before).max()
#     print(f"Theta before: {theta_before}")
#     print(f"Theta after:  {theta_after}")
#     print(f"Max diff: {diff:.6f}, Iters: {bc.row_generation.master_model.IterCount}")
#     print("PASS" if diff > 1e-6 else "FAIL: theta unchanged")



# Test Bayesian bootstrap
results = bc.standard_errors.compute_bayesian_bootstrap(num_bootstrap=30, seed=123)
if rank == 0:
    print(results.mean)
    print(results.se)



