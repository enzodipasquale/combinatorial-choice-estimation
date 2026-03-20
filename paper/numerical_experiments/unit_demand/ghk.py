import numpy as np
from scipy.stats import norm
from scipy.special import logsumexp
from scipy.optimize import minimize


def _build_diff_operators(J, Sigma):
    ops = []
    for j in range(J + 1):
        if j == 0:
            M = -np.eye(J)
        else:
            idx = j - 1
            others = np.concatenate([np.arange(idx), np.arange(idx + 1, J)])
            M = np.zeros((J, J))
            M[0, idx] = 1.0
            M[1:, idx] = 1.0
            M[np.arange(1, J), others] = -1.0
        Omega = M @ Sigma @ M.T
        L = np.linalg.cholesky(Omega)
        ops.append((M, L))
    return ops


def _ghk_log_probs_and_grad(m, dm_dbeta, L, uniforms):
    """
    GHK simulator with forward-mode AD for gradient.

    m:         (N_g, D)      differenced mean utilities
    dm_dbeta:  (N_g, D, K)   Jacobian of m w.r.t. beta
    L:         (D, D)        Cholesky factor
    uniforms:  (N_g, R, D)   uniform draws

    Returns:
        log_P:     (N_g,)    log choice probabilities
        dlog_P:    (N_g, K)  gradient of log_P w.r.t. beta
    """
    N_g, R, D = uniforms.shape
    K = dm_dbeta.shape[2]

    z = np.zeros((N_g, R, D))
    dz = np.zeros((N_g, R, D, K))
    log_p = np.zeros((N_g, R))
    dlog_p = np.zeros((N_g, R, K))

    for d in range(D):
        # eta_d = sum_{d'<d} L[d,d'] * z[d']
        eta = np.einsum('nrd,d->nr', z[:, :, :d], L[d, :d])
        deta = np.einsum('nrdk,d->nrk', dz[:, :, :d, :], L[d, :d])

        # upper_d = (m_d - eta_d) / L[d,d]
        upper = (m[:, d:d+1] - eta) / L[d, d]
        dupper = (dm_dbeta[:, d:d+1, :] - deta) / L[d, d]

        # Phi(upper), phi(upper)
        cdf = np.clip(norm.cdf(upper), 1e-15, 1.0)
        pdf = norm.pdf(upper)

        # log_p += log Phi(upper)
        log_p += np.log(cdf)
        dlog_p += (pdf / cdf)[:, :, None] * dupper

        # z_d = Phi^{-1}(u_d * Phi(upper))
        a = np.clip(uniforms[:, :, d] * cdf, 1e-15, 1 - 1e-15)
        z[:, :, d] = norm.ppf(a)

        # dz_d/dbeta = (u_d * phi(upper) / phi(z_d)) * dupper
        phi_z = np.clip(norm.pdf(z[:, :, d]), 1e-15, None)
        dz[:, :, d, :] = ((uniforms[:, :, d] * pdf) / phi_z)[:, :, None] * dupper

    # log P_i = logsumexp(log_p_ir) - log(R)
    log_P = logsumexp(log_p, axis=1) - np.log(R)

    # dlog P_i/dbeta = sum_r w_ir * dlog_p_ir/dbeta
    # w_ir = softmax weights = exp(log_p_ir) / sum_r' exp(log_p_ir')
    w = np.exp(log_p - logsumexp(log_p, axis=1, keepdims=True))
    dlog_P = np.einsum('nr,nrk->nk', w, dlog_p)

    return log_P, dlog_P


def estimate_probit_mle_individual(X, choices, Sigma, R=500, seed=42):
    N, J, K = X.shape
    ops = _build_diff_operators(J, Sigma)
    rng = np.random.default_rng(seed)

    Ms = [M for M, L in ops]
    Ls = [L for M, L in ops]

    groups = []
    for j in range(J + 1):
        idx = np.where(choices == j)[0]
        if len(idx) > 0:
            groups.append((j, idx, rng.uniform(size=(len(idx), R, J))))

    # Precompute dm/dbeta for each group (does not depend on beta)
    # m = X[idx] @ beta @ M.T  =>  dm/dbeta[k] = X[idx,:,k] @ M.T
    group_dm = []
    for j, idx, _ in groups:
        group_dm.append(np.einsum('njk,dj->ndk', X[idx], Ms[j]))

    def neg_ll_and_grad(beta):
        total_ll = 0.0
        total_grad = np.zeros(K)
        for (j, idx, uniforms), dm_dbeta in zip(groups, group_dm):
            m = X[idx] @ beta @ Ms[j].T
            log_P, dlog_P = _ghk_log_probs_and_grad(
                m, dm_dbeta, Ls[j], uniforms)
            total_ll += log_P.sum()
            total_grad += dlog_P.sum(axis=0)
        return -total_ll, -total_grad

    def objective(beta_np):
        v, g = neg_ll_and_grad(beta_np)
        return v, g

    result = minimize(objective, np.zeros(K), method='L-BFGS-B', jac=True)
    return result.x, result
