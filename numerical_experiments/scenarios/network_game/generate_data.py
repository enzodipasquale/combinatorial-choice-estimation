"""DGP for the network peer-effects game (Graham and Gonzalez 2023).

Single game with T players, matching the paper's Panel B setup. For player
t in a game with adjacency matrix D, covariate matrix X, and shocks U:

    v_t(y; X, U, theta) = y_t * (
        x_t' beta  +  delta * sum_{s != t} D_{ts} y_s  -  U_t
    ).

Paper's Appendix B details:

  * Positions uniform on [0, sqrt(T)]^2; edge formation follows
    D_{st} = 1(A_{st} - eps_{st} >= 0) with eps_{st} iid logistic(0, 1) and
    A_{st} = ln(3) if dist(s, t) <= r else -inf. So agents within r link
    with probability 0.75, those beyond never. r is chosen so
    E[deg] = 0.75 * pi * r^2 = avg_degree (paper: 10).
  * Covariates X_1, X_2 ~ Bern(0.5); X_3, X_4 ~ U[0, 1], held fixed.
  * Shocks U_t ~ N(0, 1) iid; fresh draws per MC rep.
  * Parameters theta_true = (-1, -0.5, -1, 0.5, 0.2).
  * Equilibrium selection: minimal NE via Tarski fixed-point iteration
    from y = 0.

Supported `selection` rules in `generate_one_rep`:
  "min"    - paper's minimal NE.
  "max"    - maximal NE (iteration from y = 1).
  "argmax" - global maximizer of V(y) = L'y + (delta/2) y'Dy via combest's
             min-cut solver. Always a NE; a.s. unique under continuous
             shocks; in general differs from min NE and max NE when the
             NE set has multiple elements.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Network and covariate construction (fixed across MC reps)
# ---------------------------------------------------------------------------

def build_geometric_graph(T: int, avg_degree: float, rng: np.random.Generator,
                          link_prob: float = 0.75):
    """Paper's Appendix B random geometric graph.

    Within-radius pairs link independently with probability `link_prob`
    (paper uses 0.75 so E[deg] = 0.75 * pi * r^2). r is chosen to hit the
    target expected degree.
    """
    side = np.sqrt(T)
    r = np.sqrt(avg_degree / (link_prob * np.pi))
    pos = rng.uniform(0.0, side, size=(T, 2))
    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    within = dist <= r
    # One logistic draw per unordered pair so D stays symmetric.
    eps_upper = rng.logistic(0.0, 1.0, size=(T, T))
    eps = np.triu(eps_upper, k=1)
    eps = eps + eps.T
    threshold = np.log(link_prob / (1.0 - link_prob))  # ln 3 at link_prob=0.75
    D = (within & (threshold >= eps)).astype(np.int8)
    np.fill_diagonal(D, 0)
    return D, pos, r


def build_covariates(T: int, rng: np.random.Generator):
    """X_1, X_2 ~ Bern(0.5); X_3, X_4 ~ U[0, 1]. Shape (T, 4)."""
    X = np.empty((T, 4))
    X[:, 0] = rng.integers(0, 2, size=T)
    X[:, 1] = rng.integers(0, 2, size=T)
    X[:, 2] = rng.uniform(0.0, 1.0, size=T)
    X[:, 3] = rng.uniform(0.0, 1.0, size=T)
    return X


def build_fixed_design(T: int, avg_degree: float, graph_seed: int):
    """Build (X, D) held fixed across MC replications.

    Returns a dict with keys "X", "D", "pos", "r".
    """
    rng = np.random.default_rng(graph_seed)
    D, pos, r = build_geometric_graph(T, avg_degree, rng)
    X = build_covariates(T, rng)
    return {"X": X, "D": D, "pos": pos, "r": r}


# ---------------------------------------------------------------------------
# Equilibrium selection
# ---------------------------------------------------------------------------

def _best_response_step(y, X, D, U, beta, delta):
    return (X @ beta + delta * (D @ y.astype(float)) - U) >= 0


def maximal_ne(X, D, U, beta, delta):
    """Max NE via best-response iteration from y = 1."""
    T = X.shape[0]
    y = np.ones(T, dtype=bool)
    for _ in range(T + 1):
        y_new = _best_response_step(y, X, D, U, beta, delta)
        if np.array_equal(y_new, y):
            return y
        y = y_new
    return y


def minimal_ne(X, D, U, beta, delta):
    """Paper's selection: min NE via best-response iteration from y = 0."""
    T = X.shape[0]
    y = np.zeros(T, dtype=bool)
    for _ in range(T + 1):
        y_new = _best_response_step(y, X, D, U, beta, delta)
        if np.array_equal(y_new, y):
            return y
        y = y_new
    return y


def argmax_potential(X, D, U, beta, delta):
    """argmax_y V(y) = (X beta - U)' y + (delta/2) y' D y via MinCutSolver.

    Passes Q = delta * triu(D, k=1) (upper-triangular, combest's convention)
    so that y' Q y = delta * sum_{i<j} D_{ij} y_i y_j = (delta/2) y' D y.
    """
    from combest.subproblems.registry.quadratic_obj.quadratic_supermodular.min_cut import (
        MinCutSolver,
    )
    T = X.shape[0]
    L = X @ beta - U
    Q = delta * np.triu(D.astype(float), k=1)
    solver = MinCutSolver(constraint_mask=None, n_items=T)
    return solver.solve(-L, -Q).astype(bool)


_SELECTION_FNS = {
    "min": minimal_ne,
    "max": maximal_ne,
    "argmax": argmax_potential,
}


# ---------------------------------------------------------------------------
# Shocks and one replication
# ---------------------------------------------------------------------------

def draw_shocks(T: int, shock_seed: int):
    """U_t ~ N(0, 1) iid, length T."""
    return np.random.default_rng(shock_seed).standard_normal(T)


def generate_one_rep(design, beta, delta, shock_seed, selection="argmax"):
    """Draw U and compute Y under the given equilibrium-selection rule.

    Returns {"Y": (T,) bool, "U": (T,) float}.
    """
    try:
        fn = _SELECTION_FNS[selection]
    except KeyError:
        raise ValueError(
            f"unknown selection={selection!r} "
            f"(expected 'min', 'max', 'argmax')") from None
    T = design["X"].shape[0]
    U = draw_shocks(T, shock_seed)
    Y = fn(design["X"], design["D"], U,
           np.asarray(beta), float(delta))
    return {"Y": Y, "U": U}
