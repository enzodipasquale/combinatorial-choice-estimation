import numpy as np
from bundlechoice.utils import price_term


def greedy(self, local_id, lambda_k, p_j):

    i_id = self.local_indeces[local_id]
    error_j = self.local_errors[local_id]

    B_j = np.zeros(self.num_items, dtype=bool)
    items_left = np.arange(self.num_items)

    while True:
        best_val = - np.inf
        best_item = -1
        for j in items_left:
            B_j[j] = True
            marginal_j = error_j[j] + self.get_x_k(i_id, B_j) @ lambda_k - price_term(p_j, B_j) 
            B_j[j] = False
            if marginal_j > best_val:
                best_val = marginal_j
                best_item = j
        if best_val <= 0:
            break
        B_j[best_item] = True
        items_left = items_left[items_left != best_item]

    optimal_bundle = B_j

    x_hat_k = self.get_x_k(i_id, optimal_bundle)
    # Compute value, characteristics and error at optimal bundle
    pricing_result =   np.concatenate(( [value],
                                        [error_j[optimal_bundle].sum(0)],
                                        x_hat_k,
                                        optimal_bundle.astype(float)
                                        ))
    return pricing_result
