#!/usr/bin/env python3
"""Second-stage estimation: IV, surplus decomposition, and entropy."""
import json, sys, yaml, argparse
import numpy as np
from pathlib import Path

CBLOCK_DIR = Path(__file__).parent
sys.path.insert(0, str(CBLOCK_DIR.parent.parent.parent.parent))

from applications.combinatorial_auction.data.loaders import load_bta_data, build_context
from applications.combinatorial_auction.data.registries import MODULAR, QUADRATIC, QUADRATIC_ID
from applications.combinatorial_auction.scripts.c_block.analyze import tsls, _build_distant_stats

RESULTS_DIR = CBLOCK_DIR / "results"
CONFIGS_DIR = CBLOCK_DIR / "configs"


def _load_iv_instruments(raw):
    pop = raw["bta_data"]["pop90"].to_numpy().astype(float)
    hhinc = raw["bta_data"]["hhinc35k"].to_numpy().astype(float)
    geo = raw["geo_distance"]
    iv_pop_mean, _ = _build_distant_stats(pop, geo, [500])
    iv_hhinc_mean, _ = _build_distant_stats(hhinc, geo, [500])
    return iv_pop_mean[500], iv_hhinc_mean[500]


def _run_2sls(delta, price, zm, zh):
    n = len(delta)
    X = np.column_stack([np.ones(n), -price])
    Z = np.column_stack([zm, zh])
    b, se, r2, _ = tsls(X, delta, Z)
    return b[0], b[1], se[0], se[1], r2


def _xbar_at_obs_bundles(ctx, b_obs, mod_names, quad_names, qid_names):
    """Compute aggregate covariate x̄_k evaluated at observed bundles.

    Under BLP inversion, these equal the expected LP column coefficients.
    Returns dict: name -> scalar value.
    """
    xbar = {}

    for name in mod_names:
        feat = MODULAR[name](ctx)  # (n_obs, n_items)
        xbar[name] = (b_obs * feat).sum()

    for name in quad_names:
        Q = QUADRATIC[name](ctx)  # (n_items, n_items)
        # item quadratic contribution: sum_i b_i' Q b_i
        xbar[name] = sum(b_obs[i] @ Q @ b_obs[i] for i in range(len(b_obs)))

    for name in qid_names:
        feat = QUADRATIC_ID[name](ctx)  # (n_obs, n_items, n_items)
        xbar[name] = sum(b_obs[i] @ feat[i] @ b_obs[i] for i in range(len(b_obs)))

    return xbar


def run(config_stem):
    config_path = CONFIGS_DIR / f"{config_stem}.yaml"
    result_path = RESULTS_DIR / config_stem / "bootstrap_result.json"

    cfg = yaml.safe_load(open(config_path))
    r = json.load(open(result_path))

    app = cfg["application"]
    mod_names = app.get("modular_regressors", [])
    quad_names = app.get("quadratic_regressors", [])
    qid_names = app.get("quadratic_id_regressors", [])
    n_id_mod = len(mod_names)
    n_btas = 480
    n_id_quad = len(qid_names)
    n_item_quad = len(quad_names)

    raw = load_bta_data()
    ctx = build_context(raw)
    price = raw["bta_data"]["bid"].to_numpy().astype(float) / 1e9
    b_obs = ctx["c_obs_bundles"].astype(float)
    n_obs = len(raw["bidder_data"])

    zm, zh = _load_iv_instruments(raw)
    xbar = _xbar_at_obs_bundles(ctx, b_obs, mod_names, quad_names, qid_names)

    boot_thetas = [np.array(t) for t in r["bootstrap_thetas"]]
    boot_u_hats = [np.array(u) for u in r["bootstrap_u_hat"]]
    converged = r.get("converged", [True] * len(boot_thetas))
    n_sim = len(boot_u_hats[0]) // n_obs

    # per-draw accumulators
    rows = []
    for b in range(len(boot_thetas)):
        if not converged[b]:
            continue
        th = boot_thetas[b]
        u = boot_u_hats[b]

        # extract parameter blocks
        named = {mod_names[i]: th[i] for i in range(n_id_mod)}
        delta = -th[n_id_mod : n_id_mod + n_btas]
        off = n_id_mod + n_btas
        for i, name in enumerate(qid_names):
            named[name] = th[off + i]
        off += n_id_quad
        for i, name in enumerate(quad_names):
            named[name] = th[off + i]

        # second stage IV
        a0, a1, a0_se, a1_se, r2 = _run_2sls(delta, price, zm, zh)
        xi = delta - a0 + a1 * price

        # total surplus (avg over sims, sum over bidders)
        surplus = u.reshape(n_obs, n_sim).mean(axis=1).sum()

        # systematic part = sum_k theta_k * xbar_k
        systematic_by_cov = {name: named[name] * xbar[name] for name in named}
        # FE/delta part: under BLP inversion, sum_j delta_j
        fe_total = delta.sum()
        systematic_total = sum(systematic_by_cov.values()) + fe_total

        entropy = surplus - systematic_total

        # delta decomposition
        a0_part = n_btas * a0
        price_part = -a1 * price.sum()
        xi_part = xi.sum()

        rows.append(dict(
            a0=a0, a1=a1, a0_se=a0_se, a1_se=a1_se, r2=r2,
            surplus=surplus, entropy=entropy,
            fe_total=fe_total, a0_part=a0_part, price_part=price_part, xi_part=xi_part,
            systematic_total=systematic_total,
            **{f"theta_{k}": named[k] for k in named},
            **{f"contrib_{k}": systematic_by_cov[k] for k in systematic_by_cov},
        ))

    return rows, mod_names, quad_names, qid_names, xbar


def report(config_stem):
    rows, mod_names, quad_names, qid_names, xbar = run(config_stem)
    n = len(rows)
    all_names = mod_names + qid_names + quad_names

    def col(key):
        return np.array([r[key] for r in rows])

    a1 = col("a1")

    W = 35

    print(f"\n{'='*70}")
    print(f"  {config_stem}  [{n} bootstrap draws]")
    print(f"{'='*70}")

    # ── 1. First-stage estimates ──
    print(f"\n  1. FIRST-STAGE ESTIMATES")
    print(f"  {'':<{W}}  {'Estimate':>10}  {'Boot SE':>10}")
    print(f"  {'-'*(W+24)}")
    for k in all_names:
        v = col(f"theta_{k}")
        print(f"  {k:<{W}}  {v.mean():>10.2f}  {v.std():>10.2f}")

    # ── 2. Second-stage IV ──
    print(f"\n  2. SECOND-STAGE IV (pop+hhinc, d>500km)")
    print(f"  {'':<{W}}  {'Estimate':>10}  {'Boot SE':>10}")
    print(f"  {'-'*(W+24)}")
    a0 = col("a0")
    print(f"  {'alpha_0':<{W}}  {a0.mean():>10.4f}  {a0.std():>10.4f}")
    print(f"  {'alpha_1 (price sensitivity)':<{W}}  {a1.mean():>10.2f}  {a1.std():>10.2f}")
    print(f"  {'alpha_0/alpha_1 ($M/license)':<{W}}  {(a0/a1).mean()*1e3:>10.1f}  {(a0/a1).std()*1e3:>10.1f}")
    print(f"  {'IV R2':<{W}}  {col('r2').mean():>10.4f}")

    # ── 3. Surplus decomposition (utils) ──
    surplus = col("surplus")
    entropy = col("entropy")
    fe = col("fe_total")
    a0p = col("a0_part")
    pp = col("price_part")
    xip = col("xi_part")
    sys_total = col("systematic_total")

    print(f"\n  3. SURPLUS DECOMPOSITION (utils)")
    print(f"  {'':<{W}}  {'Mean':>10}  {'Boot SE':>10}")
    print(f"  {'-'*(W+24)}")
    print(f"  {'Total surplus (1/S sum u_si)':<{W}}  {surplus.mean():>10.2f}  {surplus.std():>10.2f}")
    print(f"  {'  Systematic (xbar theta)':<{W}}  {sys_total.mean():>10.2f}  {sys_total.std():>10.2f}")
    for k in all_names:
        c = col(f"contrib_{k}")
        print(f"  {'    ' + k:<{W}}  {c.mean():>10.2f}  {c.std():>10.2f}")
    print(f"  {'    delta (sum_j delta_j)':<{W}}  {fe.mean():>10.2f}  {fe.std():>10.2f}")
    print(f"  {'      n * alpha_0':<{W}}  {a0p.mean():>10.2f}  {a0p.std():>10.2f}")
    print(f"  {'      -alpha_1 * sum prices':<{W}}  {pp.mean():>10.2f}  {pp.std():>10.2f}")
    print(f"  {'      sum_j xi_j':<{W}}  {xip.mean():>10.2f}  {xip.std():>10.2f}")
    print(f"  {'  Entropy of choice (residual)':<{W}}  {entropy.mean():>10.2f}  {entropy.std():>10.2f}")

    # ── 4. Dollar-denominated ($B) ──
    surplus_d = surplus / a1
    entropy_d = entropy / a1
    sys_d = sys_total / a1
    fe_d = fe / a1
    a0p_d = a0p / a1
    pp_d = pp / a1
    xip_d = xip / a1
    total_revenue = col("price_part") / (-a1) * a1  # just price.sum()
    obs_revenue = -(pp / a1)  # sum prices in $B (note pp = -a1 * sum_prices)

    print(f"\n  4. SURPLUS DECOMPOSITION ($B = utils / alpha_1)")
    print(f"  {'':<{W}}  {'Mean':>10}  {'Boot SE':>10}")
    print(f"  {'-'*(W+24)}")
    print(f"  {'Total surplus':<{W}}  {surplus_d.mean():>10.4f}  {surplus_d.std():>10.4f}")
    print(f"  {'  Systematic':<{W}}  {sys_d.mean():>10.4f}  {sys_d.std():>10.4f}")
    for k in all_names:
        c = col(f"contrib_{k}") / a1
        print(f"  {'    ' + k:<{W}}  {c.mean():>10.4f}  {c.std():>10.4f}")
    print(f"  {'    delta':<{W}}  {fe_d.mean():>10.4f}  {fe_d.std():>10.4f}")
    print(f"  {'      n * alpha_0 / alpha_1':<{W}}  {a0p_d.mean():>10.4f}  {a0p_d.std():>10.4f}")
    print(f"  {'      sum prices (= revenue)':<{W}}  {obs_revenue.mean():>10.4f}  {obs_revenue.std():>10.4f}")
    print(f"  {'      sum xi / alpha_1':<{W}}  {xip_d.mean():>10.4f}  {xip_d.std():>10.4f}")
    print(f"  {'  Entropy of choice':<{W}}  {entropy_d.mean():>10.4f}  {entropy_d.std():>10.4f}")
    print()


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="*", default=["boot", "boot_3"],
                        help="Config stems to analyze (default: boot boot_3)")
    args = parser.parse_args()
    for stem in args.configs:
        report(stem)
