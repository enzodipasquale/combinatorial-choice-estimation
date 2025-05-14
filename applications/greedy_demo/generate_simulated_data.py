import numpy as np
from bundlechoice.subproblems import get_subproblem

num_agents = 1000
num_items = 300
num_features = 5
num_k = 3

lambda_k_star = np.ones(num_k)
np.random.seed(0)
eps_i_j = np.random.normal(0, 1, (num_agents, num_items))
phi_i_j_k = np.random.normal(0, 1, (num_agents, num_items, num_features-1))

def get_x_k(i_id, B_j):    
    return np.concatenate((phi_i_j_k[i_id, B_j].sum(1), [B_j.sum(0) **2]))




