"""Feature registry for the combinatorial auction.

Three registries:
- MODULAR: per-(obs, item) linear features, signature ctx -> (n_obs, n_items).
- QUADRATIC: per-(item, item) interaction features shared across obs,
  signature ctx -> (n_items, n_items).
- QUADRATIC_ID: per-(obs, item, item) interaction features,
  signature ctx -> (n_obs, n_items, n_items).

Column ordering in input_data mirrors the lists passed to prepare().
"""
import numpy as np

WEIGHT_ROUNDING_TICK = 1000
LB_QUADRATICS = 1e-10

MODULAR: dict = {}
QUADRATIC: dict = {}
QUADRATIC_ID: dict = {}


def _reg(store, name):
    def dec(fn):
        store[name] = fn
        return fn
    return dec


def modular(name):      return _reg(MODULAR, name)
def quadratic(name):    return _reg(QUADRATIC, name)
def quadratic_id(name): return _reg(QUADRATIC_ID, name)


def normalize_interaction_matrix(M, pop):
    """Zero diagonal, row-normalize, scale rows by pop, zero diagonal again."""
    M = M.copy()
    np.fill_diagonal(M, 0)
    row = M.sum(1)
    M[row > 0] /= row[row > 0, None]
    M *= pop[:, None]
    np.fill_diagonal(M, 0)
    return M


def _pop_centroid(ctx, delta):
    pop, geo = ctx["pop"], ctx["geo_distance"]
    pc = np.outer(pop, pop).astype(float)
    np.fill_diagonal(pc, 0)
    if delta != 0:
        m = geo > 0
        pc[m] /= geo[m] ** delta
    return normalize_interaction_matrix(pc, pop)


# ── Modular: (ctx) -> (n_obs, n_items) ────────────────────────────────

@modular("elig_constant")
def _(ctx): return ctx["elig"][:, None] * np.ones(len(ctx["pop"]))[None, :]

@modular("elig_pop")
def _(ctx): return ctx["elig"][:, None] * ctx["pop"][None, :]

@modular("elig_percapin")
def _(ctx): return ctx["elig"][:, None] * ctx["percapin"][None, :]

@modular("elig_hhinc35k")
def _(ctx): return ctx["elig"][:, None] * ctx["hhinc35k"][None, :]

@modular("elig_density")
def _(ctx): return ctx["elig"][:, None] * ctx["density"][None, :]

@modular("elig_imwl")
def _(ctx): return ctx["elig"][:, None] * ctx["imwl"][None, :]

@modular("elig_price")
def _(ctx): return ctx["elig"][:, None] * ctx["price"][None, :]

@modular("assets_pop")
def _(ctx): return ctx["assets"][:, None] * ctx["pop"][None, :]

@modular("designated_pop")
def _(ctx): return ctx["designated"][:, None] * ctx["pop"][None, :]

@modular("designated_elig_pop")
def _(ctx): return (ctx["designated"] * ctx["elig"])[:, None] * ctx["pop"][None, :]

@modular("log_dist_hq")
def _(ctx): return np.log1p(ctx["geo_distance"][ctx["hq_bta_idx"]])

@modular("elig_log_dist_hq")
def _(ctx): return ctx["elig"][:, None] * MODULAR["log_dist_hq"](ctx)

@modular("designated_log_dist_hq")
def _(ctx): return ctx["designated"][:, None] * MODULAR["log_dist_hq"](ctx)

@modular("log_dist_hq_pop")
def _(ctx): return MODULAR["log_dist_hq"](ctx) * ctx["pop"][None, :]

@modular("elig_log_dist_hq_pop")
def _(ctx): return ctx["elig"][:, None] * MODULAR["log_dist_hq_pop"](ctx)


# ── Quadratic: (ctx) -> (n_items, n_items) ────────────────────────────

@quadratic("adjacency")
def _(ctx): return normalize_interaction_matrix(ctx["bta_adjacency"], ctx["pop"])

@quadratic("pop_centroid_00")
def _(ctx): return _pop_centroid(ctx, 0)

@quadratic("pop_centroid_delta2")
def _(ctx): return _pop_centroid(ctx, 2)

@quadratic("pop_centroid_delta4")
def _(ctx): return _pop_centroid(ctx, 4)

@quadratic("travel_survey")
def _(ctx):
    M = np.where(ctx["travel_survey"] == 0, LB_QUADRATICS, ctx["travel_survey"])
    return normalize_interaction_matrix(M, ctx["pop"])

@quadratic("air_travel")
def _(ctx):
    M = np.where(ctx["air_travel"] == 0, LB_QUADRATICS, ctx["air_travel"])
    return normalize_interaction_matrix(M, ctx["pop"])


# ── Quadratic_id: (ctx) -> (n_obs, n_items, n_items) ──────────────────

def _elig_times(Q_fn):
    return lambda ctx: ctx["elig"][:, None, None] * Q_fn(ctx)[None, :, :]

QUADRATIC_ID.update({
    "elig_adjacency":            _elig_times(QUADRATIC["adjacency"]),
    "elig_pop_centroid_00":      _elig_times(QUADRATIC["pop_centroid_00"]),
    "elig_pop_centroid_delta2":  _elig_times(QUADRATIC["pop_centroid_delta2"]),
    "elig_pop_centroid_delta4":  _elig_times(QUADRATIC["pop_centroid_delta4"]),
    "elig_travel_survey":        _elig_times(QUADRATIC["travel_survey"]),
    "elig_air_travel":           _elig_times(QUADRATIC["air_travel"]),
})


def build(registry, names, ctx):
    """Stack a list of features into a (..., len(names)) float64 array.
    Returns None if `names` is empty so callers can branch on presence."""
    if not names:
        return None
    return np.stack([registry[n](ctx) for n in names], axis=-1).astype(np.float64)
