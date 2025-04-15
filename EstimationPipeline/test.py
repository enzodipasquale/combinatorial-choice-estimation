from bundle_choice import BundleChoice
import numpy as np

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


# Define config dictionary (normally loaded from YAML)
config = {
    "tol_certificate": 1e-3,
    "max_slack_counter": 3,
    "tol_row_generation": 5,
    "row_generation_decay": 0.5,
    "num_simulations": 2,
    "max_iters": 100,
}



if rank == 0:
    # # Agent independent data be in any form
    # item_data = np.array([3,2,3])

    # # Agent specific data is a dictionary with characteristics as keys

    # agent_data = {
    #     "modular": np.array([[1, 2, 4], 
    #                         [5, 6, 8]]),
    #     "capacity": np.array([10, 20]),
    # }

    # # errors are a 3D tensor (modular!)
    # errors = np.array([[[0.1, 0.2, 0.3], 
    #                     [0.4, 0.5, 0.6]], 

    #                     [[0.7, 0.8, 0.9], 
    #                     [1.0, 1.1, 1.2]]])

    

    data = {
        "item_data": item_data,
        "agent_data": agent_data,
        "errors": errors
    }
else:
    data = None

# Compute min_iters
config["min_iters"] = np.log(config["tol_certificate"] / (config["tol_row_generation"] - 1)) / np.log(config["row_generation_decay"])



####################################################################################################


my_test = BundleChoice(data, config=config)




my_test.scatter_data()

print("Rank:", my_test.rank, my_test.item_data)