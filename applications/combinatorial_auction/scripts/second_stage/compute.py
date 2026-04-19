"""Per-bootstrap-draw welfare decomposition for a single spec.

For each bootstrap draw b:
    θ = bootstrap_thetas[b],   u = bootstrap_u_hat[b]
    surplus   = mean_sim(u).sum_i
    δ         = −θ_fe
    (α₀, α₁, γ) = run_2sls(δ, raw, app)
    ξ         = δ − α₀ + α₁·p − Z'γ
    entropy   = surplus − Σ θ·x̄ (named covariates) − δ.sum()

δ is additively decomposed as α₀-part + price-part + controls-part + ξ-part;
each piece is reported so Table 3 can display the structural composition.

Covariates are never re-built here — x̄ is assembled from the arrays
`data.prepare()` already produced.
"""
import json, yaml
from pathlib import Path
import numpy as np

from ...data.prepare import prepare
from ...data.loaders import load_raw
from .iv import run_2sls, compute_xi

APP_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS  = APP_ROOT / "results"
CONFIGS  = APP_ROOT / "configs"


def _xbar(input_data, b_obs):
    """x̄ from prepare()'s arrays. FE block uses b_obs.sum(axis=0) to match
    combest's internal convention (single-winner-per-item makes this identical
    to δ.sum() within the welfare decomposition)."""
    id_mod  = input_data["id_data"]["modular"]                       # (n_obs, n_bta, K_m)
    q_item  = input_data["item_data"].get("quadratic")               # (n_bta, n_bta, K_q) or None
    q_id    = input_data["id_data"].get("quadratic")                 # (n_obs, n_bta, n_bta, K_qid)

    K_m, n_bta = id_mod.shape[-1], b_obs.shape[1]
    K_qid = q_id.shape[-1]   if q_id   is not None else 0
    K_q   = q_item.shape[-1] if q_item is not None else 0

    xbar = np.zeros(K_m + n_bta + K_qid + K_q)
    xbar[:K_m]                         = np.einsum("ij,ijk->k", b_obs, id_mod)
    xbar[K_m:K_m + n_bta]              = b_obs.sum(axis=0)
    if K_qid:
        xbar[K_m + n_bta:K_m + n_bta + K_qid] = np.einsum("ij,ijlk,il->k", b_obs, q_id, b_obs)
    if K_q:
        xbar[K_m + n_bta + K_qid:]     = np.einsum("ij,jlk,il->k", b_obs, q_item, b_obs)
    return xbar


def _setup(spec_stem, configs_dir):
    """Shared setup between point-estimate and bootstrap decompositions."""
    app = yaml.safe_load(open(configs_dir / f"{spec_stem}.yaml"))["application"]
    input_data, meta = prepare(
        modular_regressors      = app.get("modular_regressors", []),
        quadratic_regressors    = app.get("quadratic_regressors", []),
        quadratic_id_regressors = app.get("quadratic_id_regressors", []),
        winners_only            = app.get("winners_only", False),
        capacity_source         = app.get("capacity_source", "initial"),
        upper_triangular_quadratic = app.get("upper_triangular_quadratic", False),
    )
    raw   = load_raw()
    price = raw["bta_data"]["bid"].to_numpy(dtype=float) / 1e9
    b_obs = input_data["id_data"]["obs_bundles"].astype(float)
    n_obs, n_btas = b_obs.shape
    n_id_mod = meta["n_id_mod"]

    mod_names   = app.get("modular_regressors", [])
    qid_names   = app.get("quadratic_id_regressors", [])
    quad_names  = app.get("quadratic_regressors", [])
    named_order = mod_names + qid_names + quad_names

    # Named covariates sit on either side of the FE block (which spans n_btas
    # positions in θ):  [ modular | FEs | quadratic_id | quadratic ].
    quad_start = n_id_mod + n_btas
    quad_end   = quad_start + meta["n_id_quad"] + meta["n_item_quad"]
    named_idx  = [*range(n_id_mod), *range(quad_start, quad_end)]

    return app, meta, raw, price, input_data, b_obs, n_obs, n_btas, n_id_mod, named_order, named_idx


def _row(th, u, xbar, *, raw, price, app, named_order, named_idx,
         n_obs, n_btas, n_id_mod, n_sim):
    """One decomposition row (shared by point-estimate and bootstrap paths)."""
    surplus  = u.reshape(n_obs, n_sim).mean(axis=1).sum()
    delta    = -th[n_id_mod:n_id_mod + n_btas]
    fe_total = delta.sum()
    contrib  = th * xbar

    named         = {n: float(th[named_idx[i]])      for i, n in enumerate(named_order)}
    contrib_named = {n: float(contrib[named_idx[i]]) for i, n in enumerate(named_order)}

    iv = run_2sls(delta, raw, app)
    xi = compute_xi(delta, price, iv["a0"], iv["a1"], iv["demand_controls"], raw["bta_data"])
    controls_part = sum(c * raw["bta_data"][v].to_numpy(dtype=float).sum()
                        for v, c in (iv["demand_controls"] or {}).items())

    return dict(
        a0=iv["a0"], a1=iv["a1"], se_a0=iv["se_a0"], se_a1=iv["se_a1"], r2=iv["r2"],
        surplus=surplus, entropy=surplus - sum(contrib_named.values()) - fe_total,
        fe_total=fe_total,
        a0_part=n_btas * iv["a0"], price_part=-iv["a1"] * price.sum(),
        controls_part=controls_part, xi_part=xi.sum(),
        **{f"theta_{k}":   v for k, v in named.items()},
        **{f"contrib_{k}": v for k, v in contrib_named.items()},
    )


def decompose_point(spec_stem, *, configs_dir=CONFIGS, results_dir=RESULTS):
    """Return (rows, named_order) for the point estimate (one row)."""
    app, meta, raw, price, input_data, b_obs, n_obs, n_btas, n_id_mod, named_order, named_idx = \
        _setup(spec_stem, configs_dir)

    r = json.load(open(results_dir / spec_stem / "point_estimate" / "result.json"))
    th = np.asarray(r["theta_hat"])
    u  = np.asarray(r["u_hat"])
    xbar = np.asarray(r["xbar"]) if r.get("xbar") is not None else _xbar(input_data, b_obs)
    n_sim = r["config"]["dimensions"]["n_simulations"]

    row = _row(th, u, xbar, raw=raw, price=price, app=app, named_order=named_order,
               named_idx=named_idx, n_obs=n_obs, n_btas=n_btas, n_id_mod=n_id_mod, n_sim=n_sim)
    return [row], named_order


def decompose(spec_stem, *, configs_dir=CONFIGS, results_dir=RESULTS):
    """Return (rows, named_order) for the bootstrap (one row per converged draw)."""
    app, meta, raw, price, input_data, b_obs, n_obs, n_btas, n_id_mod, named_order, named_idx = \
        _setup(spec_stem, configs_dir)

    r = json.load(open(results_dir / spec_stem / "bootstrap" / "bootstrap_result.json"))
    xbar = np.array(r["xbar"]) if r.get("xbar") is not None else _xbar(input_data, b_obs)
    boot_thetas = np.asarray(r["bootstrap_thetas"])
    boot_u_hats = np.asarray(r["bootstrap_u_hat"])
    converged   = r.get("converged", [True] * len(boot_thetas))
    n_sim = boot_u_hats.shape[1] // n_obs

    rows = []
    for th, u, conv in zip(boot_thetas, boot_u_hats, converged):
        if not conv:
            continue
        rows.append(_row(th, u, xbar, raw=raw, price=price, app=app,
                         named_order=named_order, named_idx=named_idx,
                         n_obs=n_obs, n_btas=n_btas, n_id_mod=n_id_mod, n_sim=n_sim))
    return rows, named_order


def decompose_all(specs):
    """Run decompose() if a bootstrap result is present, else decompose_point()
    on the point estimate. Specs with neither are skipped."""
    out = {}
    for stem in specs:
        boot = RESULTS / stem / "bootstrap" / "bootstrap_result.json"
        point = RESULTS / stem / "point_estimate" / "result.json"
        if boot.exists():
            out[stem] = decompose(stem)
        elif point.exists():
            print(f"[{stem}] using point_estimate/result.json (no bootstrap)")
            out[stem] = decompose_point(stem)
        else:
            print(f"[{stem}] no bootstrap or point_estimate result, skip")
    return out
