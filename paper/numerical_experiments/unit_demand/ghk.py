import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm as jnorm
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)


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


def _ghk_prob(m, L, uniforms):
    R, D = uniforms.shape
    z = jnp.zeros((R, D))
    log_p = jnp.zeros(R)
    for d in range(D):
        eta = z @ L[d]
        upper = (m[d] - eta) / L[d, d]
        cdf = jnp.clip(jnorm.cdf(upper), 1e-15, 1.0)
        log_p = log_p + jnp.log(cdf)
        z_d = jnorm.ppf(jnp.clip(uniforms[:, d] * cdf, 1e-15, 1 - 1e-15))
        z = z.at[:, d].set(z_d)
    return jnp.exp(log_p).mean()


def estimate_probit_mle(X, shares, Sigma, R=500, seed=42):
    J, K = X.shape
    shares_jax = jnp.array(shares)
    X_jax = jnp.array(X)

    ops = _build_diff_operators(J, Sigma)
    rng = np.random.default_rng(seed)
    Ms = jnp.stack([M for M, L in ops])
    Ls = jnp.stack([L for M, L in ops])
    all_uniforms = jnp.stack([rng.uniform(size=(R, J)) for _ in range(J + 1)])

    def neg_ll(beta):
        V = X_jax @ beta
        ll = jnp.float64(0.0)
        for j in range(J + 1):
            prob = _ghk_prob(Ms[j] @ V, Ls[j], all_uniforms[j])
            ll = ll + shares_jax[j] * jnp.log(jnp.clip(prob, 1e-300, 1.0))
        return -ll

    neg_ll_vg = jax.jit(jax.value_and_grad(neg_ll))

    def objective(beta_np):
        v, g = neg_ll_vg(jnp.array(beta_np))
        return float(v), np.array(g)

    result = minimize(objective, np.zeros(K), method='L-BFGS-B', jac=True)
    return result.x, result


def estimate_probit_mle_individual(X, choices, Sigma, R=500, seed=42):
    N, J, K = X.shape
    X_jax = jnp.array(X)

    ops = _build_diff_operators(J, Sigma)
    rng = np.random.default_rng(seed)
    Ms = jnp.stack([M for M, L in ops])
    Ls = jnp.stack([L for M, L in ops])

    groups = []
    for j in range(J + 1):
        idx = np.where(choices == j)[0]
        if len(idx) > 0:
            groups.append((
                j,
                jnp.array(idx),
                jnp.array(rng.uniform(size=(len(idx), R, J)))
            ))

    def neg_ll(beta):
        ll = jnp.float64(0.0)
        for j, idx, uniforms in groups:
            V = X_jax[idx] @ beta
            m = V @ Ms[j].T
            probs = jax.vmap(
                lambda mi, ui: _ghk_prob(mi, Ls[j], ui)
            )(m, uniforms)
            ll = ll + jnp.log(jnp.clip(probs, 1e-300, 1.0)).sum()
        return -ll

    neg_ll_vg = jax.jit(jax.value_and_grad(neg_ll))

    def objective(beta_np):
        v, g = neg_ll_vg(jnp.array(beta_np))
        return float(v), np.array(g)

    result = minimize(objective, np.zeros(K), method='L-BFGS-B', jac=True)
    return result.x, result
