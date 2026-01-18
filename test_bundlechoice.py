import bundlechoice as bc
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

n_agents, n_items = 5, 3
k_mod, k_quad = 2, 1
n_features = k_mod + k_quad


if rank == 0:
    cfg = {'dimensions': {'n_obs': n_agents, 'n_items': n_items, 'n_features': n_features}}
    obs_bundles = np.ones((n_agents, n_items))
    modular_agent = np.arange(n_agents * n_items * k_mod).reshape(n_agents, n_items, k_mod)
    quadratic_item = np.arange(n_items * n_items * k_quad).reshape(n_items, n_items, k_quad)
    for k in range(k_quad):
        np.fill_diagonal(quadratic_item[:, :, k], 0)
        quadratic_item[:, :, k] = (quadratic_item[:, :, k] + quadratic_item[:, :, k].T) / 2
    weights = np.arange(n_items)+1
    capacity = np.arange(n_agents)
    
    input_data = {
        "id_data": {"obs_bundles": obs_bundles, "modular": modular_agent, "capacity": capacity},
        "item_data": {"quadratic": quadratic_item, "weights": weights}
    }
else:
    cfg = None
    input_data = None

bc = bc.BundleChoice()
bc.load_config(cfg)

# print("rank", rank, "n_obs", bc.n_obs, "n_items", bc.n_items, "n_features", bc.n_features)
bc.data.load_input_data(input_data)
bc.oracles.build_quadratic_features_from_data()
bc.oracles.build_local_modular_error_oracle()





bundles = np.ones((bc.data.num_local_agent, bc.n_items))
features = bc.oracles.features_oracle(bundles)
error_1 = bc.oracles.error_oracle(bundles)
error_2 = bc.oracles.error_oracle(bundles)
# print("rank", rank, "features", features, "error_1", error_1, "error_2", error_2)
# print(bc.data.local_data["id_data"]["capacity"])
capacity = bc.data.local_data["id_data"]["capacity"]

# subproblems: load from registry
subproblem = bc.subproblems.load('QuadraticKnapsackGRB')
subproblem.initialize()
solution = subproblem.solve(np.ones(n_features))
# print(rank, capacity, solution)

# Estimation via row generation
result = bc.row_generation.solve(
    obs_weights=np.ones(bc.data_manager.num_local_agent),
    init_master=True,
    init_subproblems=False  # already initialized above
)
# print("theta_obj_coef", bc.row_generation.theta_obj_coef)