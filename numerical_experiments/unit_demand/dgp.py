import numpy as np


def _draw_covariates(rng, shape, rho=0.0):
    """Draw covariates from N(0, Sigma_X) with equicorrelation rho."""
    X = rng.normal(0, 1, shape)
    if rho > 0:
        K = shape[-1]
        Sigma_X = (1 - rho) * np.eye(K) + rho * np.ones((K, K))
        L = np.linalg.cholesky(Sigma_X)
        X = X @ L.T
    return X


def _draw_errors(rng, N, J, Sigma):
    """Draw J-dimensional errors, avoiding BLAS warnings for diagonal Sigma."""
    diag = np.diag(Sigma)
    if np.allclose(Sigma, np.diag(diag)):
        return rng.normal(0, 1, (N, J)) * np.sqrt(diag)
    return rng.multivariate_normal(np.zeros(J), Sigma, size=N)


def simulate_probit_individual(N, J, K, beta, Sigma=None, rho=0.0, seed=42):
    rng = np.random.default_rng(seed)
    X = _draw_covariates(rng, (N, J, K), rho)
    if Sigma is None:
        Sigma = np.eye(J)
    eps = _draw_errors(rng, N, J, Sigma)
    V = np.einsum('ijk,k->ij', X, beta)
    U = np.column_stack([np.zeros(N), V + eps])
    choices = np.argmax(U, axis=1)
    return X, choices


def simulate_logit_individual(N, J, K, beta, sigma=1.0, rho=0.0, seed=42):
    rng = np.random.default_rng(seed)
    X = _draw_covariates(rng, (N, J, K), rho)
    eps = sigma * rng.gumbel(size=(N, J + 1))
    V = np.einsum('ijk,k->ij', X, beta)
    U = np.column_stack([np.zeros(N), V]) + eps
    choices = np.argmax(U, axis=1)
    return X, choices
