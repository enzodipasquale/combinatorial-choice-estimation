import numpy as np
from bundlechoice.utils import price_term


def solve_greedy(self, _,local_id, lambda_k, p_j):

    error_j = self.local_errors[local_id]
    B_j = np.zeros(self.num_items, dtype=bool)
    items_left = np.arange(self.num_items)

    while True:
        best_val = - np.inf
        best_item = -1
        for j in items_left:
            base_x_k = self.get_x_k(local_id, B_j, local =True)
            B_j[j] = True
            marginal_j = error_j[j] + (self.get_x_k(local_id, B_j, local =True) - base_x_k) @ lambda_k 
            marginal_j -= p_j[j] if p_j is not None else 0
            B_j[j] = False
            if marginal_j > best_val:
                best_val = marginal_j
                best_item = j
        if best_val <= 0:
            break
        B_j[best_item] = True
        items_left = items_left[items_left != best_item]

    optimal_bundle = B_j

    return optimal_bundle

