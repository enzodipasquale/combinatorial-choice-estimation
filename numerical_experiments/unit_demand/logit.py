import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


def estimate_logit_mle_individual(X, choices):
    N, J, K = X.shape

    def neg_ll(beta):
        V = np.einsum('ijk,k->ij', X, beta)
        V_full = np.column_stack([np.zeros(N), V])
        log_denom = logsumexp(V_full, axis=1)
        return -(V_full[np.arange(N), choices] - log_denom).sum()

    def grad(beta):
        V = np.einsum('ijk,k->ij', X, beta)
        V_full = np.column_stack([np.zeros(N), V])
        log_denom = logsumexp(V_full, axis=1, keepdims=True)
        probs = np.exp(V_full - log_denom)[:, 1:]
        obs = np.zeros((N, J))
        inside = choices > 0
        obs[inside, choices[inside] - 1] = 1.0
        return -np.einsum('ij,ijk->k', obs - probs, X)

    opts = {'ftol': 1e-12, 'gtol': 1e-10, 'maxiter': 1000}
    result = minimize(neg_ll, np.zeros(K), jac=grad, method='L-BFGS-B',
                      options=opts)
    return result.x, result
