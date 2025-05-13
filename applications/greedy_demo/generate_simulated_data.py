import numpy as np
from bundlechoice.subproblems import get_subproblem

num_agents = 1000
num_items = 300
num_features = 5
num_k = 3

lambda_k_star = np.ones(3)
eps_i_j = np.random.normal(0, 1, (num_agents, num_items))

