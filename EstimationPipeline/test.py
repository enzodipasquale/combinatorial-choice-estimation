from bundle_choice import BundleChoice
import numpy as np
import yaml
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Load configuration from YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Load data on rank 0
if rank == 0:
    item_data = {
                "quadratic" : np.load("./data/quadratic_characteristic_j_j_k.npy"),
                "weights" : np.load("./data/weight_j.npy")
                }

    # Agent specific data is a dictionary with characteristics as keys
    agent_data = {
                "modular": np.load("./data/modular_characteristics_i_j_k.npy"),
                "capacity": np.load("./data/capacity_i.npy"),
                }

    # errors are a 3D tensor (modular!)
    np.random.seed(0)

    data = {
            "item_data": item_data,
            "agent_data": agent_data,
            "errors": np.random.normal(0,1,size = (config["num_simuls"], 255, 493)),
            "obs_bundle": np.load("./data/matching_i_j.npy")
            }
else:
    data = None

dims = (255, 493, 4)





# Characteristics oracle
def compute_features(self, bundle_i_j):
    modular_i_j_k = self.agent_data["modular"]
    quadratic_j_j_k = self.item_data["quadratic"]

    features_hat_i_k = np.concatenate((
                                    np.einsum('ijk,ij->ik', modular_i_j_k, bundle_i_j),
                                    np.einsum('jlk,ij,il->ik', quadratic_j_j_k, bundle_i_j, bundle_i_j)
                                    ), axis = 1)

    return features_hat_i_k


#####################################################################################################

my_test = BundleChoice(data, dims, config, compute_features)

my_test.scatter_data()

# print("Rank:", my_test.rank, my_test.item_data["quadratic"].shape)
# print("Rank:", my_test.rank, my_test.local_agent_data["modular"].shape)

if rank == 0:
    my_test.initialize_master()
    # my_test.compute_features(my_test.obs_bundle)



# # Pricing problem
# def init_pricing(self, weight_j, capacity):

#     # Create subproblem
#     subproblem = gp.Model() 
#     subproblem.setParam('OutputFlag', 0)
#     subproblem.setAttr('ModelSense', gp.GRB.MAXIMIZE)
#     # Create variables
#     B_j = subproblem.addMVar(len(weight_j), vtype = gp.GRB.BINARY)

#     # Knapsack constraint
#     subproblem.addConstr(weight_j @ B_j <= capacity)
#     subproblem.update()

#     return subproblem 



# def solve_pricing(subproblem, modular_j_k, quadratic_j_j_k ,lambda_k, p_j):

#     ### Define objective from data and master solution 
#     num_MOD = modular_j_k.shape[1] - 1
#     L_j =  modular_j_k[:,0] + modular_j_k[:,1:] @ lambda_k[:num_MOD] -  p_j
#     Q_j_j = quadratic_j_j_k @ lambda_k[num_MOD: ]

#     # Set objective
#     B_j = subproblem.getVars()
#     subproblem.setObjective(B_j @ L_j + B_j @ Q_j_j @ B_j)

#     # Solve the updated subproblem
#     subproblem.optimize()
#     optimal_bundle = np.array(subproblem.x, dtype=bool)
#     value = subproblem.objVal
#     check_gap(subproblem.MIPGap)

#     ### Compute value of characteristics at optimal bundle (1+1+K+J)
#     row =   np.concatenate(([value],
#                             (modular_j_k[optimal_bundle]).sum(0), 
#                             quadratic_j_j_k[optimal_bundle][:, optimal_bundle].sum((0, 1)),
#                             subproblem.x
#                             ))
#     return row
