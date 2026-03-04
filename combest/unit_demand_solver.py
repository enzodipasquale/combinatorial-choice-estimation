import numpy as np
import gurobipy as gp


class UnitDemandLP:

    def __init__(self, covariates, agent_ids, weights, n_agents,
                 theta_obj, theta_lb=-100, theta_ub=100):
        model = gp.Model()
        model.setParam("OutputFlag", 0)

        n_theta = covariates.shape[1]
        theta = model.addMVar(n_theta, obj=theta_obj, lb=theta_lb, ub=theta_ub)
        u = model.addMVar(n_agents, obj=weights, lb=0)
        constrs = model.addConstr(u[agent_ids] >= covariates @ theta)

        self._model = model
        self._theta = theta
        self._constrs = constrs

    def solve(self, errors, verbose=False):
        self._model.setParam("OutputFlag", int(verbose))
        self._constrs.RHS = errors
        self._model.optimize()
        if self._model.Status != gp.GRB.OPTIMAL:
            raise RuntimeError(f"Gurobi status: {self._model.Status}")
        return self._theta.X


def _build_lp(X_tiled, n_agents, J, theta_obj, kwargs):
    return UnitDemandLP(
        X_tiled,
        np.repeat(np.arange(n_agents), J),
        np.full(n_agents, 1.0 / n_agents),
        n_agents,
        theta_obj,
        **kwargs
    )


def _gumbel_errors(rng, shape):
    eps_all = rng.gumbel(size=shape[:-1] + (shape[-1] + 1,))
    return eps_all[..., 1:] - eps_all[..., 0:1]


def _theta_obj_from_shares(X, shares):
    return -(shares[1:] @ X)


def _theta_obj_from_choices(X, choices):
    N, J, K = X.shape
    obs = np.zeros((N, K))
    inside = choices > 0
    obs[inside] = X[inside, choices[inside] - 1]
    return -obs.mean(axis=0)


def estimate_probit_lp(X, shares, Sigma, n_simulations, seed=42, **kwargs):
    J, K = X.shape
    eps = np.random.default_rng(seed).multivariate_normal(
        np.zeros(J), Sigma, size=n_simulations)
    lp = _build_lp(np.tile(X, (n_simulations, 1)),
                   n_simulations, J, _theta_obj_from_shares(X, shares), kwargs)
    return lp.solve(eps.ravel())


def estimate_probit_lp_individual(X, choices, Sigma, n_simulations, seed=42, **kwargs):
    N, J, K = X.shape
    n_agents = N * n_simulations
    eps = np.random.default_rng(seed).multivariate_normal(
        np.zeros(J), Sigma, size=(N, n_simulations))
    lp = _build_lp(np.tile(X, (1, n_simulations, 1)).reshape(-1, K),
                   n_agents, J, _theta_obj_from_choices(X, choices), kwargs)
    return lp.solve(eps.ravel())


def estimate_logit_lp(X, shares, n_simulations, seed=42, **kwargs):
    J, K = X.shape
    eps = _gumbel_errors(np.random.default_rng(seed), (n_simulations, J))
    lp = _build_lp(np.tile(X, (n_simulations, 1)),
                   n_simulations, J, _theta_obj_from_shares(X, shares), kwargs)
    return lp.solve(eps.ravel())


def estimate_logit_lp_individual(X, choices, n_simulations, seed=42, **kwargs):
    N, J, K = X.shape
    n_agents = N * n_simulations
    eps = _gumbel_errors(np.random.default_rng(seed), (N, n_simulations, J))
    lp = _build_lp(np.tile(X, (1, n_simulations, 1)).reshape(-1, K),
                   n_agents, J, _theta_obj_from_choices(X, choices), kwargs)
    return lp.solve(eps.ravel())
