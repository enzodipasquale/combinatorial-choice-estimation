import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


def _logit_log_probs(V):
    V_full = np.concatenate([[0], V])
    return V_full - logsumexp(V_full)


def estimate_logit_mle(X, shares):
    J, K = X.shape

    def neg_ll(beta):
        log_p = _logit_log_probs(X @ beta)
        return -(shares * log_p).sum()

    def grad(beta):
        log_p = _logit_log_probs(X @ beta)
        probs = np.exp(log_p)
        return -(shares[1:] - probs[1:]) @ X

    result = minimize(neg_ll, np.zeros(K), jac=grad, method='L-BFGS-B')
    return result.x, result


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

    result = minimize(neg_ll, np.zeros(K), jac=grad, method='L-BFGS-B')
    return result.x, result
