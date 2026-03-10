import numpy as np


def simulate_probit(N, J, K, beta, Sigma=None, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (J, K))
    if Sigma is None:
        Sigma = np.eye(J)
    eps = rng.multivariate_normal(np.zeros(J), Sigma, size=N)
    U = np.column_stack([np.zeros(N), X @ beta + eps])
    choices = np.argmax(U, axis=1)
    shares = np.bincount(choices, minlength=J + 1) / N
    return X, choices, shares


def simulate_probit_individual(N, J, K, beta, Sigma=None, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (N, J, K))
    if Sigma is None:
        Sigma = np.eye(J)
    eps = rng.multivariate_normal(np.zeros(J), Sigma, size=N)
    V = np.einsum('ijk,k->ij', X, beta)
    U = np.column_stack([np.zeros(N), V + eps])
    choices = np.argmax(U, axis=1)
    return X, choices


def simulate_logit(N, J, K, beta, sigma=1.0, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (J, K))
    eps = sigma * rng.gumbel(size=(N, J + 1))
    V = np.concatenate([[0], X @ beta])
    choices = np.argmax(V + eps, axis=1)
    shares = np.bincount(choices, minlength=J + 1) / N
    return X, choices, shares


def simulate_logit_individual(N, J, K, beta, sigma=1.0, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (N, J, K))
    eps = sigma * rng.gumbel(size=(N, J + 1))
    V = np.einsum('ijk,k->ij', X, beta)
    U = np.column_stack([np.zeros(N), V]) + eps
    choices = np.argmax(U, axis=1)
    return X, choices
