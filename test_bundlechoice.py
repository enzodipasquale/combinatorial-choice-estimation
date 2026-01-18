import bundlechoice as bc
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

n_agents, n_items = 20, 10
k_mod, k_quad = 2, 1
n_features = k_mod + k_quad
theta_star = np.array([1.0, 0.5, 10])

if rank == 0:
    cfg = {'dimensions': {'n_obs': n_agents, 'n_items': n_items, 'n_features': n_features},
           'row_generation': {'max_iters': 100, 'tol_row_generation': 1e-8}}
    
    np.random.seed(42)
    weights = np.ones(n_items)
    capacity = np.random.randint(1, weights.sum(), n_agents)
    
    modular_agent = np.random.randn(n_agents, n_items, k_mod)
    quadratic_item = np.random.randn(n_items, n_items, k_quad) * 0.1
    for k in range(k_quad):
        np.fill_diagonal(quadratic_item[:, :, k], 0)
        quadratic_item[:, :, k] = (quadratic_item[:, :, k] + quadratic_item[:, :, k].T) / 2
    
    obs_bundles_placeholder = np.zeros((n_agents, n_items))
    input_data = {
        "id_data": {"obs_bundles": obs_bundles_placeholder, "modular": modular_agent, "capacity": capacity},
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
obs_bundles_local = bc.subproblems.initialize_and_solve_subproblems(theta_star)
bc.data.local_data["id_data"]["obs_bundles"]  = obs_bundles_local
bc.data.local_obs_bundles = obs_bundles_local.astype(bool)

# subproblem = bc.subproblems.load('QuadraticKnapsackGRB')
# subproblem.initialize()
# obs_bundles_local = subproblem.solve(theta_star)
# bc.data.local_data["id_data"]["obs_bundles"]  = obs_bundles_local
# bc.data.local_obs_bundles = obs_bundles_local.astype(bool)


# Estimation via row generation
bc.oracles.build_local_modular_error_oracle(seed=43)
result = bc.row_generation.solve(
    obs_weights=np.ones(bc.data_manager.num_local_agent),
    init_master=True,
    init_subproblems=False
)

if rank == 0:
    print("Converged:", result.converged)
    print("Iterations:", result.num_iterations)
    print("Theta hat:", result.theta_hat)
    print("Theta star:", theta_star)