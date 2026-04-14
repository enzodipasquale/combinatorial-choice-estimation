"""Core computation for post-estimation analysis."""
import json, yaml
import numpy as np
from pathlib import Path

CBLOCK_DIR = Path(__file__).parent.parent

import sys
sys.path.insert(0, str(CBLOCK_DIR.parent.parent.parent.parent))

from applications.combinatorial_auction.data.loaders import load_bta_data, build_context
from applications.combinatorial_auction.data.registries import MODULAR, QUADRATIC, QUADRATIC_ID
from applications.combinatorial_auction.data.iv import load_iv_instruments, run_2sls

RESULTS_DIR = CBLOCK_DIR / "results"
CONFIGS_DIR = CBLOCK_DIR / "configs"
PREFERRED = ["boot", "boot_3", "boot_5"]


def _load_data():
    raw = load_bta_data()
    ctx = build_context(raw)
    price = raw["bta_data"]["bid"].to_numpy().astype(float) / 1e9
    b_obs = ctx["c_obs_bundles"].astype(float)
    n_obs = len(raw["bidder_data"])
    zm, _, zh = load_iv_instruments(raw)
    return raw, ctx, price, b_obs, n_obs, zm, zh


def _xbar(ctx, b_obs, n_obs, mod_names, quad_names, qid_names):
    xbar = {}
    for name in mod_names:
        xbar[name] = (b_obs * MODULAR[name](ctx)).sum()
    for name in quad_names:
        Q = QUADRATIC[name](ctx)
        xbar[name] = sum(b_obs[i] @ Q @ b_obs[i] for i in range(n_obs))
    for name in qid_names:
        feat = QUADRATIC_ID[name](ctx)
        xbar[name] = sum(b_obs[i] @ feat[i] @ b_obs[i] for i in range(n_obs))
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
    n_btas = 480
    n_id_quad = len(qid_names)

    xbar = _xbar(ctx, b_obs, n_obs, mod_names, quad_names, qid_names)

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

        named = {mod_names[i]: th[i] for i in range(n_id_mod)}
        delta = -th[n_id_mod:n_id_mod + n_btas]
        off = n_id_mod + n_btas
        for i, name in enumerate(qid_names):
            named[name] = th[off + i]
        off += n_id_quad
        for i, name in enumerate(quad_names):
            named[name] = th[off + i]

        a0, a1, a0_se, a1_se, r2 = run_2sls(delta, price, zm, zh)
        xi = delta - a0 + a1 * price
        surplus = u.reshape(n_obs, n_sim).mean(axis=1).sum()

        sys_by_cov = {name: named[name] * xbar[name] for name in all_names}
        fe_total = delta.sum()
        entropy = surplus - sum(sys_by_cov.values()) - fe_total

        rows.append(dict(
            a0=a0, a1=a1, a0_se=a0_se, a1_se=a1_se, r2=r2,
            surplus=surplus, entropy=entropy, fe_total=fe_total,
            a0_part=n_btas * a0, price_part=-a1 * price.sum(), xi_part=xi.sum(),
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
