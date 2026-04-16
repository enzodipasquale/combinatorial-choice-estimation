"""Core computation for post-estimation analysis."""
import json, yaml
import numpy as np
from pathlib import Path

CBLOCK_DIR = Path(__file__).parent.parent

import sys
sys.path.insert(0, str(CBLOCK_DIR.parent.parent.parent.parent))

from applications.combinatorial_auction.data.loaders import load_bta_data, build_context
from applications.combinatorial_auction.data.registries import MODULAR, QUADRATIC, QUADRATIC_ID
from applications.combinatorial_auction.data.iv import (
    load_iv_instruments, second_stage, compute_xi,
)

RESULTS_DIR = CBLOCK_DIR / "results"
CONFIGS_DIR = CBLOCK_DIR / "configs"
PREFERRED = ["boot", "boot_3", "boot_pop_scaling", "boot_pop_scaling_large"]


def _load_data():
    raw = load_bta_data()
    ctx = build_context(raw)
    price = raw["bta_data"]["bid"].to_numpy().astype(float) / 1e9
    b_obs = ctx["c_obs_bundles"].astype(float)
    n_obs = len(raw["bidder_data"])
    zm, _, zh = load_iv_instruments(raw)
    return raw, ctx, price, b_obs, n_obs, zm, zh


def _build_xbar(ctx, b_obs, n_obs, n_btas, mod_names, quad_names, qid_names):
    """Fallback: build xbar from data when not saved in result JSON."""
    n_cov = len(mod_names) + n_btas + len(qid_names) + len(quad_names)
    xbar = np.zeros(n_cov)
    off = 0
    for name in mod_names:
        xbar[off] = (b_obs * MODULAR[name](ctx)).sum()
        off += 1
    xbar[off:off + n_btas] = b_obs.sum(axis=0)
    off += n_btas
    for name in qid_names:
        feat = QUADRATIC_ID[name](ctx)
        xbar[off] = sum(b_obs[i] @ feat[i] @ b_obs[i] for i in range(n_obs))
        off += 1
    for name in quad_names:
        Q = QUADRATIC[name](ctx)
        xbar[off] = sum(b_obs[i] @ Q @ b_obs[i] for i in range(n_obs))
        off += 1
    return xbar


def run_spec(config_stem, raw=None, ctx=None, price=None, b_obs=None,
             n_obs=None, zm=None, zh=None):
    """Run full second-stage analysis for one spec. Returns list of per-draw dicts."""
    if raw is None:
        raw, ctx, price, b_obs, n_obs, zm, zh = _load_data()

    cfg = yaml.safe_load(open(CONFIGS_DIR / f"{config_stem}.yaml"))
    r = json.load(open(RESULTS_DIR / config_stem / "bootstrap_result.json"))

    app = cfg["application"]
    mod_names = app.get("modular_regressors", [])
    quad_names = app.get("quadratic_regressors", [])
    qid_names = app.get("quadratic_id_regressors", [])
    all_names = mod_names + qid_names + quad_names
    n_id_mod = len(mod_names)
    n_btas = b_obs.shape[1]
    n_id_quad = len(qid_names)
    use_blp = app.get("error_scaling") == "pop"

    # xbar: use saved value or recompute
    if "xbar" in r:
        xbar = np.array(r["xbar"])
    else:
        xbar = _build_xbar(ctx, b_obs, n_obs, n_btas, mod_names, quad_names, qid_names)

    boot_thetas = [np.array(t) for t in r["bootstrap_thetas"]]
    boot_u_hats = [np.array(u) for u in r["bootstrap_u_hat"]]
    converged = r.get("converged", [True] * len(boot_thetas))
    n_sim = len(boot_u_hats[0]) // n_obs

    rows = []
    for b in range(len(boot_thetas)):
        if not converged[b]:
            continue
        th = boot_thetas[b]
        u = boot_u_hats[b]

        # Welfare decomposition (same as combest method)
        surplus = u.reshape(n_obs, n_sim).mean(axis=1).sum()
        contributions = th * xbar

        # Named covariate contributions (no sign flip needed)
        named_indices = list(range(n_id_mod)) + list(range(n_id_mod + n_btas, len(th)))
        named = {}
        for i in range(n_id_mod):
            named[mod_names[i]] = th[i]
        off = n_id_mod + n_btas
        for i, name in enumerate(qid_names):
            named[name] = th[off + i]
        off += n_id_quad
        for i, name in enumerate(quad_names):
            named[name] = th[off + i]

        # FE: sign flip for structural decomposition
        # contributions[FE] = θ_fe · x̄_fe = -δ · 1
        # delta = -θ_fe, so fe_total = δ.sum() = -contributions[FE].sum()
        delta = -th[n_id_mod:n_id_mod + n_btas]
        fe_total = delta.sum()  # = -contributions[n_id_mod:n_id_mod+n_btas].sum()

        # 2SLS on delta
        iv = second_stage(delta, price, raw, zm, zh, use_blp)
        a0, a1, dc = iv["a0"], iv["a1"], iv["demand_controls"]
        xi = compute_xi(delta, price, a0, a1, dc, raw["bta_data"])

        # Covariate contributions in utils (from θ·x̄, no sign issue for named covs)
        sys_by_cov = {}
        for name in all_names:
            idx = named_indices[all_names.index(name)]
            sys_by_cov[name] = contributions[idx]

        # Entropy (structural): surplus - covariates - delta
        entropy = surplus - sum(sys_by_cov.values()) - fe_total

        controls_part = 0.0
        if dc:
            for v, c in dc.items():
                controls_part += c * raw["bta_data"][v].to_numpy().astype(float).sum()

        rows.append(dict(
            a0=a0, a1=a1,
            se_a0=iv["se_a0"], se_a1=iv["se_a1"], r2=iv["r2"],
            surplus=surplus, entropy=entropy, fe_total=fe_total,
            a0_part=n_btas * a0, price_part=-a1 * price.sum(),
            controls_part=controls_part, xi_part=xi.sum(),
            **{f"theta_{k}": named[k] for k in named},
            **{f"contrib_{k}": sys_by_cov[k] for k in sys_by_cov},
        ))

    return rows, all_names


def run_all(specs=None):
    """Run second-stage for multiple specs, sharing data loads."""
    if specs is None:
        specs = PREFERRED
    raw, ctx, price, b_obs, n_obs, zm, zh = _load_data()
    results = {}
    for stem in specs:
        result_path = RESULTS_DIR / stem / "bootstrap_result.json"
        if not result_path.exists():
            continue
        rows, names = run_spec(stem, raw, ctx, price, b_obs, n_obs, zm, zh)
        results[stem] = (rows, names)
    return results
