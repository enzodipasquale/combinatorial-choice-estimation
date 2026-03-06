import numpy as np

WEIGHT_ROUNDING_TICK = 1000
LB_QUADRATICS = 1e-10

MODULAR = {}
QUADRATIC = {}
QUADRATIC_ID = {}

def modular(name):
    def dec(fn): MODULAR[name] = fn; return fn
    return dec

def quadratic(name):
    def dec(fn): QUADRATIC[name] = fn; return fn
    return dec

def quadratic_id(name):
    def dec(fn): QUADRATIC_ID[name] = fn; return fn
    return dec

# helpers

def normalize_interaction_matrix(matrix, pop):
    matrix = matrix.copy()
    np.fill_diagonal(matrix, 0)
    outflow = matrix.sum(1)
    mask = outflow > 0
    matrix[mask] /= outflow[mask][:, None]
    matrix *= pop[:, None]
    np.fill_diagonal(matrix, 0)
    return matrix

def _pop_centroid(ctx, delta):
    pop, geo = ctx["pop"], ctx["geo_distance"]
    pc = (pop[:, None] * pop[None, :]).astype(float)
    np.fill_diagonal(pc, 0)
    mask = geo > 0
    pc[mask] /= geo[mask] ** delta
    return normalize_interaction_matrix(pc, pop)

# modular regressors: (ctx) -> (n_obs, n_items)

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

# quadratic regressors: (ctx) -> (n_items, n_items)

@quadratic("adjacency")
def _(ctx): return normalize_interaction_matrix(ctx["bta_adjacency"], ctx["pop"])

@quadratic("pop_centroid_delta4")
def _(ctx): return _pop_centroid(ctx, delta=4)

@quadratic("travel_survey")
def _(ctx):
    ts = np.where(ctx["travel_survey"] == 0, LB_QUADRATICS, ctx["travel_survey"])
    return normalize_interaction_matrix(ts, ctx["pop"])

@quadratic("air_travel")
def _(ctx):
    at = np.where(ctx["air_travel"] == 0, LB_QUADRATICS, ctx["air_travel"])
    return normalize_interaction_matrix(at, ctx["pop"])

# quadratic_id regressors: (ctx) -> (n_obs, n_items, n_items)

@quadratic_id("elig_adjacency")
def _(ctx): return ctx["elig"][:, None, None] * QUADRATIC["adjacency"](ctx)[None, :, :]

@quadratic_id("elig_pop_centroid_delta4")
def _(ctx): return ctx["elig"][:, None, None] * _pop_centroid(ctx, delta=4)[None, :, :]
