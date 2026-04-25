"""Surplus decomposition at the CF LP optimum.

Given the CF save artifacts:
    cf_{tag}_{label}.json              — theta_hat, alpha_0, alpha_1, prices
    cf_{tag}_{label}_bundles.npz       — BundleStore
    cf_{tag}_{label}_cut_agent_ids.npy — global agent id for each cut
    cf_{tag}_{label}_pi.npy            — LP dual for each cut

this script:

  1. Rebuilds the CF data pipeline EXACTLY as ``counterfactual/run.py::solve_cf``
     did (same ``prepare_counterfactual`` call, same error seed / scaling /
     offset), so the error oracle returns identical values to what the LP saw.
  2. Stacks the saved bundles and agent ids, feeds them to
     ``features_manager.covariates_and_errors_oracle`` to get per-cut covariate
     vectors and total errors (= aggregated random ε + deterministic offset).
  3. Splits the total error into (α₀·|m|, controls·Σpop_m, ξ_m) deterministic
     pieces and the per-(agent, bundle) stochastic remainder.
  4. Weights each cut by its LP dual π_c and aggregates.

Output mirrors the slide's surplus decomposition table: per-covariate named
contributions + α₀·N + controls + ξ + ε (stochastic) + price paid → bidder
surplus.

Usage:
    python -m applications.combinatorial_auction.scripts.counterfactual.decompose \\
        pop_scaling_large_2 [with_xi|no_xi]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import yaml

APP_ROOT  = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = APP_ROOT.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from applications.combinatorial_auction.data.loaders import load_raw
from applications.combinatorial_auction.scripts.second_stage.iv import run_2sls
from applications.combinatorial_auction.scripts.counterfactual.prepare import (
    prepare_counterfactual, freeze_bounds,
)
from applications.combinatorial_auction.scripts.counterfactual.run import (
    CF_CONFIG, CF_ERROR_SEED, _iv_tag,
)
from applications.combinatorial_auction.scripts import errors as err_mod
from combest.estimation.bundle_store import BundleStore
import combest as ce


# ── Covariate grouping (matches sync_to_slides.py conventions) ──────────
ASSORTATIVE = {"elig_pop", "assets_pop"}
COMP_LABEL = {
    "adjacency":            "Adjacency",
    "pop_centroid_delta4":  "Gravity",
    "pop_centroid_00":      "Gravity (base)",
    "air_travel":           "Air travel",
    "travel_survey":        "Commuter survey",
}


def compute_cf_decomposition(spec, label):
    """Return the π-weighted CF surplus decomposition as a dict mirroring
    `_welfare_point`'s shape:
        {
            'tag', 'label', 'alpha_1', 'a1' (alias),
            'assortative', 'license_specific', 'a0_B', 'controls_B', 'eps_B',
            'comp_total', 'comp_rows' [list of (label, $B)],
            'xi_B', 'price_paid', 'revenue', 'total', 'welfare',
            'contrib_B' [dict per named covariate],
        }
    """
    import copy

    cfg_dir = APP_ROOT / "configs"
    res_dir = APP_ROOT / "results"
    cfg_path = cfg_dir / f"{spec}.yaml"
    est_app = yaml.safe_load(open(cfg_path))["application"]
    tag = _iv_tag(est_app)

    cf_dir = res_dir / spec / "counterfactual"
    stem   = cf_dir / f"cf_{tag}_{label}"

    # --- Load CF artifacts ---------------------------------------------
    cf_json = json.load(open(stem.with_suffix(".json")))
    theta_cf = np.asarray(cf_json["theta_hat"])
    alpha_0  = float(cf_json["alpha_0"])
    alpha_1  = float(cf_json["alpha_1"])      # this is α₂ in paper notation
    prices   = np.asarray(cf_json["prices"])
    u_hat    = np.asarray(cf_json["u_hat"])

    bs          = BundleStore.load(str(stem) + "_bundles.npz")
    cut_agents  = np.load(str(stem) + "_cut_agent_ids.npy")
    pi          = np.load(str(stem) + "_pi.npy")
    n_cuts      = len(cut_agents)
    assert bs.cut_to_bundle.size == n_cuts == len(pi), "bundle/agent/pi lengths disagree"

    include_xi = (label == "with_xi")

    # --- Recompute (α₀, α₁, demand_controls) via the same 2SLS ---------
    pt = json.load(open(res_dir / spec / "point_estimate" / "result.json"))
    theta_bta = np.asarray(pt["theta_hat"])
    raw = load_raw()
    iv = run_2sls(-theta_bta[1:1 + 480], raw, est_app)
    assert abs(iv["a0"] - alpha_0) < 1e-10, "α₀ mismatch (stale CF?)"
    assert abs(iv["a1"] - alpha_1) < 1e-10, "α₁ mismatch (stale CF?)"

    # --- Rebuild CF context (same prepare as solve_cf) -----------------
    input_data, meta, cf = prepare_counterfactual(
        theta_bta, est_app, alpha_0=alpha_0, alpha_1=alpha_1,
        demand_controls=iv["demand_controls"],
    )
    offset_m = cf["offset_m"] if include_xi else cf["offset_m_no_xi"]

    # --- Build combest model identically to solve_cf -------------------
    config = copy.deepcopy(CF_CONFIG)
    freeze_bounds(config, meta, cf)
    config["dimensions"].update(
        n_obs=meta["n_obs"], n_items=meta["n_items"],
        n_covariates=meta["n_covariates"], covariate_names=meta["covariate_names"],
    )
    config["application"].update(
        n_id_mod=meta["n_id_mod"], n_item_mod=meta["n_item_mod"],
        n_id_quad=meta["n_id_quad"], n_item_quad=meta["n_item_quad"],
    )
    model = ce.Model()
    model.load_config(config)
    model.data.load_and_distribute_input_data(input_data)
    model.features.build_quadratic_covariates_from_data()
    err_mod.install_aggregated(
        model, seed=CF_ERROR_SEED, A=cf["A"], offset=offset_m,
        scaling=est_app.get("error_scaling"), pop=cf["pop"], elig=cf["elig"],
    )

    # --- Evaluate covariates + errors at the saved bundles ------------
    bundles_all = bs.get(np.arange(n_cuts))                 # (n_cuts, n_items) bool
    cov, err = model.features.covariates_and_errors_oracle(
        bundles_all.astype(bool), cut_agents.astype(np.int64))
    # cov: (n_cuts, n_covariates) ; err: (n_cuts,)  ε + offset over bundle

    # --- Deterministic vs stochastic piece of err ---------------------
    #  err(b, i) = Σ_{m ∈ b} (pop-scaled ε_{i,s,m}  +  offset_m)
    err_det   = bundles_all @ offset_m                       # (n_cuts,) deterministic
    err_stoch = err - err_det

    # Split deterministic offset into its three components (cf. prepare_counterfactual):
    #   offset_m = mta_sizes·α₀  +  A @ controls_bta  +  A @ xi   (or offset_m_no_xi w/o last)
    mta_sizes   = cf["A"].sum(1)
    A_controls  = offset_m - mta_sizes * alpha_0 - (cf["A"] @ cf["xi"] if include_xi else 0.0)
    A_xi        = cf["A"] @ cf["xi"] if include_xi else np.zeros(meta["n_items"])

    per_bundle_a0_part       = bundles_all @ (mta_sizes * alpha_0)   # α₀ · |bundle|
    per_bundle_controls_part = bundles_all @ A_controls              # controls contribution
    per_bundle_xi_part       = bundles_all @ A_xi                    # ξ contribution
    # sanity: per_bundle_a0_part + per_bundle_controls_part + per_bundle_xi_part ≈ err_det
    assert np.allclose(per_bundle_a0_part + per_bundle_controls_part + per_bundle_xi_part,
                       err_det, atol=1e-8), "offset decomposition mismatch"

    # --- Named covariate contributions (θ̂ · cov, by name) -------------
    mod_names  = est_app.get("modular_regressors", [])
    quad_names = est_app.get("quadratic_regressors", [])
    qid_names  = est_app.get("quadratic_id_regressors", [])
    named_order = mod_names + qid_names + quad_names

    n_id_mod    = meta["n_id_mod"]
    n_items_cf  = meta["n_items"]
    n_id_quad   = meta["n_id_quad"]
    # θ_cf covariate layout: [ modular | item_FE (prices) | quadratic_id | quadratic ]
    quad_start  = n_id_mod + n_items_cf
    named_idx   = [*range(n_id_mod), *range(quad_start, quad_start + n_id_quad + meta["n_item_quad"])]

    # --- Weighted aggregation ------------------------------------------
    # Per-bundle utility contribution from each named covariate k:
    #     θ_k · cov[c, idx_k]
    # Aggregate: Σ_c π_c · θ_k · cov[c, idx_k]
    contrib_named = {}
    for pos, name in enumerate(named_order):
        k = named_idx[pos]
        contrib_named[name] = float((pi * theta_cf[k] * cov[:, k]).sum())

    a0_contrib       = float((pi * per_bundle_a0_part).sum())
    controls_contrib = float((pi * per_bundle_controls_part).sum())
    xi_contrib       = float((pi * per_bundle_xi_part).sum())
    eps_contrib      = float((pi * err_stoch).sum())

    # Price paid — item-modular has -α₂ on diagonal, so utility cost per item = -α₂·price_m.
    # Σ_c π_c · [-α₂ · Σ_m price_m · b_c,m]  =  -α₂ · Σ_m price_m · (Σ_c π_c · b_c,m)
    fractional_hold_m = pi @ bundles_all.astype(float)               # (n_items,), sums to 1 per MTA at optimum
    price_util_cost   = -alpha_1 * float((prices * fractional_hold_m).sum())

    total_surplus_util = (sum(contrib_named.values())
                          + a0_contrib + controls_contrib + xi_contrib + eps_contrib
                          + price_util_cost)

    # Cross-check against u_hat (bidder surplus = Σ_{i,s} u_{i,s} = Σ_c π_c · u_c)
    n_sim = CF_CONFIG["dimensions"]["n_simulations"]
    u_cuts = cov @ theta_cf + err    # raw cut RHS (in utils)
    surplus_from_duals = float((pi * u_cuts).sum())
    print(f"\nSanity (all in utils, Σ over all (obs,sim) pairs):")
    print(f"  u_hat.sum()          = {u_hat.sum():+.4f}")
    print(f"  Σ π · u_c            = {surplus_from_duals:+.4f}")
    print(f"  decomp-components    = {total_surplus_util:+.4f}")

    # --- Convert to $B and print table ---------------------------------
    # Slide convention: surplus = (1/n_sim) · Σ u_{i,s} / α₂  → divide utility
    # components by n_sim × α₂.  Revenue is already in $B (not utils).
    inv_a = 1.0 / (n_sim * alpha_1)
    # Precompute complementarity rows (pair base quadratic with its elig_<base>)
    comp_total_B = 0.0
    comp_rows = []
    used = set()
    contrib_B = {n: contrib_named[n] * inv_a for n in named_order}
    assortative_B = sum(contrib_B[k] for k in named_order if k in ASSORTATIVE)
    for k in named_order:
        if k.startswith("elig_") or k in ASSORTATIVE:
            continue
        total = contrib_B[k]
        used.add(k)
        elig_k = f"elig_{k}"
        if elig_k in contrib_B:
            total += contrib_B[elig_k]; used.add(elig_k)
        comp_rows.append((COMP_LABEL.get(k, k), total))
        comp_total_B += total
    for k in named_order:
        if k.startswith("elig_") and k not in used and k not in ASSORTATIVE:
            comp_rows.append((f"elig × {COMP_LABEL.get(k[5:], k[5:])}", contrib_B[k]))
            comp_total_B += contrib_B[k]

    a0_B         = a0_contrib * inv_a
    controls_B   = controls_contrib * inv_a
    xi_B         = xi_contrib * inv_a
    eps_B        = eps_contrib * inv_a
    license_specific_B = a0_B + controls_B + eps_B    # ε subsumed here for "license-specific residual"
    total_B      = total_surplus_util * inv_a
    revenue_B    = float(prices.sum())
    price_paid_B = price_util_cost * inv_a

    return dict(
        tag=tag, label=label,
        alpha_1=alpha_1, a1=alpha_1,                 # alias for parity w/ _welfare_point
        assortative=assortative_B,
        a0_B=a0_B, controls_B=controls_B, eps_B=eps_B,
        license_specific=license_specific_B,
        comp_total=comp_total_B, comp_rows=comp_rows,
        xi_B=xi_B, price_paid=price_paid_B,
        revenue=revenue_B, total=total_B, welfare=total_B + revenue_B,
        contrib_B=contrib_B,
    )


def main(spec, label):
    d = compute_cf_decomposition(spec, label)
    print(f"\n── CF Welfare decomposition (π-weighted cuts)   "
          f"[{d['tag']} / {d['label']}]   α₂ = {d['alpha_1']:.4f} ──")
    print(f"  {'Component':<32}  {'$B':>10}")
    print(f"  {'-'*32}  {'-'*10}")
    print(f"  Standalone returns")
    print(f"     Assortative (λ₁)                {d['assortative']:+10.3f}")
    print(f"     License Specific (α₀+ctrls+ε)   {d['license_specific']:+10.3f}")
    print(f"        α₀ · N                        {d['a0_B']:+10.3f}")
    print(f"        controls (α₁·pop)             {d['controls_B']:+10.3f}")
    print(f"        ε (stochastic)                {d['eps_B']:+10.3f}")
    print(f"  Complementarities                  {d['comp_total']:+10.3f}")
    for lbl, v in d["comp_rows"]:
        print(f"     {lbl:<30}    {v:+10.3f}")
    print(f"  Unobservables (ξ)                  {d['xi_B']:+10.3f}")
    print(f"  Price paid (−Σp·fraction)          {d['price_paid']:+10.3f}")
    print(f"  {'-'*32}  {'-'*10}")
    print(f"  {'Total bidder surplus':<32}  {d['total']:+10.3f}")
    print(f"  {'Auction revenue':<32}  {d['revenue']:+10.3f}")
    print(f"  {'Total welfare':<32}  {d['welfare']:+10.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("label", nargs="?", default="with_xi",
                    choices=["with_xi", "no_xi"])
    args = ap.parse_args()
    main(args.spec, args.label)
