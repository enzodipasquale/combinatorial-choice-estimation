#!/usr/bin/env python3
"""
Aggregate surplus decomposition using bootstrap u_hat values.

Identity (true by construction, identifying assumption):
    (1/S) sum_{s,i} u_si = sum_i theta @ f_i(S_i^obs) + (1/S) sum_{s,i,j} S_ij^obs * eps_sij

where the LHS comes from bootstrap_u_hat, the first RHS term is computable from
observed bundles + theta_b, and the error component is backed out as the residual.

The deterministic part is split as:
    sum_i theta_b @ f_i(S_i^obs)
        = theta_b[0] * A_elig_pop
        + sum_j theta_FE_j^b * freq_j          <- BTA FEs (further split via IV)
        + theta_b[481] * A_elig_adj
        + theta_b[482] * A_adj
        + theta_b[483] * A_pcd

The BTA FE component is further decomposed via IV structural params:
    theta_FE_j^b = alpha_1 * p_j - alpha_0 - xi_j^b
    => FE_b = alpha_1 * price_paid_obs - alpha_0 * total_btas - sum_j freq_j * xi_j^b

All components expressed in dollars (divide by alpha_1).
Reported as mean +/- std across 150 bootstrap samples.
"""
import json, sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from applications.combinatorial_auction.data.loaders import load_bta_data, build_context
from applications.combinatorial_auction.data.registries import QUADRATIC

CBLOCK_DIR = Path(__file__).parent

# Structural IV params (fixed, not re-estimated per bootstrap)
ALPHA_0 = -2.511155
ALPHA_1 = 41.383851
N_SIM   = 20
N_OBS   = 252
N_BTAS  = 480


def main():
    # ── Load bootstrap results ─────────────────────────────────────────────────
    res = json.load(open(CBLOCK_DIR / "bootstrap_result_config_boot.json"))
    boot_thetas = np.array(res["bootstrap_thetas"])   # (150, 484)
    boot_u_hat  = np.array(res["bootstrap_u_hat"])    # (150, 5040)
    theta_pt    = np.array(res["theta_hat"])           # (484,)  point estimate
    n_boot = boot_thetas.shape[0]

    # ── Load data ──────────────────────────────────────────────────────────────
    raw  = load_bta_data()
    ctx  = build_context(raw)
    price = raw["bta_data"]["bid"].to_numpy().astype(float) / 1e9  # (480,)
    elig  = ctx["elig"]                                             # (252,)
    pop   = ctx["pop"]                                              # (480,)
    b_obs = ctx["c_obs_bundles"].astype(float)                     # (252, 480)

    # ── Precompute FIXED aggregate features (do not depend on theta) ───────────
    # agg_feat[k] = sum_i f_{ik}(S_i^obs)
    # so that sum_i theta @ f_i(S_i^obs) = theta @ agg_feat

    agg_feat = np.zeros(484)

    # [0] elig_pop
    elig_pop_feat = elig[:, None] * pop[None, :]          # (252, 480)
    agg_feat[0] = (b_obs * elig_pop_feat).sum()

    # [1..480] BTA FEs: freq_j = number of bidders who bid on BTA j
    freq = b_obs.sum(0)                                   # (480,)
    agg_feat[1:481] = freq

    # [481] elig_adjacency: sum_i elig_i * sum_{j,l} S_ij*S_il*adj_jl
    adj     = QUADRATIC["adjacency"](ctx)                 # (480, 480)
    quad_adj = (b_obs @ adj * b_obs).sum(1)              # (252,)
    agg_feat[481] = (elig * quad_adj).sum()

    # [482] adjacency: sum_i sum_{j,l} S_ij*S_il*adj_jl
    agg_feat[482] = quad_adj.sum()

    # [483] pop_centroid_delta4
    pcd      = QUADRATIC["pop_centroid_delta4"](ctx)
    quad_pcd = (b_obs @ pcd * b_obs).sum(1)
    agg_feat[483] = quad_pcd.sum()

    # Fixed sub-quantities for BTA FE decomposition
    total_btas      = freq.sum()                          # sum_i |S_i^obs|
    price_paid_obs  = float(freq @ price)                 # sum_i sum_j S_ij * p_j  (in $B)

    # alpha_0 and price parts of FE are FIXED (don't depend on theta_b)
    fe_alpha0_part = -ALPHA_0 * total_btas                # in utility units
    fe_price_part  =  ALPHA_1 * price_paid_obs            # in utility units

    # ── Per-bootstrap-sample decomposition ────────────────────────────────────
    T          = np.zeros(n_boot)   # total surplus (utility)
    det        = np.zeros(n_boot)   # deterministic part
    comp       = np.zeros((n_boot, 7))  # [elig_pop, fe_total, fe_a0, fe_price, fe_xi,
                                        #  elig_adj, adj, pcd]  (7 because fe split 3)
    # columns: 0=elig_pop, 1=FE_total, 2=fe_a0, 3=fe_price, 4=fe_xi,
    #          5=elig_adj, 6=adj, 7=pcd  → 8 cols
    comp = np.zeros((n_boot, 8))

    for b in range(n_boot):
        theta_b = boot_thetas[b]                          # (484,)
        u_b     = boot_u_hat[b]                           # (5040,)

        # Total surplus: (1/S) * sum_{all (i,s)} u_si
        T[b] = u_b.sum() / N_SIM

        # Deterministic: theta_b @ agg_feat
        det[b] = theta_b @ agg_feat

        # Sub-components
        comp[b, 0] = theta_b[0] * agg_feat[0]            # elig_pop

        fe_b           = theta_b[1:481] @ freq            # BTA FE total
        # xi part backed out from FE:
        # FE_b = fe_a0 + fe_price - xi_part => xi_part = fe_a0 + fe_price - FE_b
        fe_xi_b        = fe_alpha0_part + fe_price_part - fe_b

        comp[b, 1] = fe_b                                 # FE total
        comp[b, 2] = fe_alpha0_part                       # alpha_0 part (fixed)
        comp[b, 3] = fe_price_part                        # price part (fixed)
        comp[b, 4] = -fe_xi_b                             # xi contribution (note sign)
        comp[b, 5] = theta_b[481] * agg_feat[481]         # elig_adjacency
        comp[b, 6] = theta_b[482] * agg_feat[482]         # adjacency
        comp[b, 7] = theta_b[483] * agg_feat[483]         # pop_centroid_d4

    error = T - det                                       # backed-out residual

    # ── Convert to dollars: divide by ALPHA_1 ─────────────────────────────────
    T_d     = T     / ALPHA_1
    comp_d  = comp  / ALPHA_1
    error_d = error / ALPHA_1

    # Sanity: top-level components sum to det
    # cols 2,3,4 are sub-components of col 1 (FE_total), not independent terms
    top_level = comp[:, 0] + comp[:, 1] + comp[:, 5] + comp[:, 6] + comp[:, 7]
    assert np.allclose(top_level, det, atol=1e-6), "Component sum != det"
    # FE sub-decomposition: a0 + price + xi_contribution == FE_total
    assert np.allclose(comp[:, 2] + comp[:, 3] + comp[:, 4], comp[:, 1], atol=1e-6), \
        "FE sub-decomp error"

    # ── Report ─────────────────────────────────────────────────────────────────
    def fmt(arr, name, indent="  "):
        m, s = arr.mean(), arr.std()
        pct  = (arr / T_d).mean() * 100
        print(f"{indent}{name:<30} {m:>9.4f}  ±{s:.4f}   ({pct:>+6.1f}% of total)")

    print(f"\n{'='*72}")
    print(f"AGGREGATE SURPLUS DECOMPOSITION  (config_boot, 150 bootstrap samples)")
    print(f"Observed bundles: N={N_OBS} bidders, {int(total_btas)} total BTA slots")
    print(f"alpha_0={ALPHA_0}, alpha_1={ALPHA_1}  |  units: $B")
    print(f"{'='*72}")
    print(f"\n  {'Component':<30} {'Mean ($B)':>9}  {'Std':>7}   {'% of total':>11}")
    print(f"  {'─'*65}")

    fmt(T_d,           "TOTAL SURPLUS (1/S sum u_si)")
    print(f"  {'─'*65}")
    fmt(comp_d[:,0],   "(A) elig_pop")
    fmt(comp_d[:,1],   "(B) BTA FEs  [total]")
    fmt(comp_d[:,2],   "    alpha_0 intercept  [fixed]")
    fmt(comp_d[:,3],   "    price (alpha_1*p)  [fixed]")
    fmt(comp_d[:,4],   "    xi (unobs. quality)")
    fmt(comp_d[:,5],   "(C) elig_adjacency")
    fmt(comp_d[:,6],   "(D) adjacency")
    fmt(comp_d[:,7],   "(E) pop_centroid_d4")
    fmt(error_d,       "(F) errors  [residual = T - det]")
    print(f"  {'─'*65}")

    # Grouped summary
    print(f"\n  GROUPED:")
    det_nofe = comp_d[:,0] + comp_d[:,5] + comp_d[:,6] + comp_d[:,7]
    fmt(det_nofe,      "  Non-FE theta (A+C+D+E)")
    fmt(comp_d[:,1],   "  BTA FEs (B)")
    fmt(error_d,       "  Errors (F, residual)")

    # Cross-bootstrap std of each component (how stable is the decomposition?)
    print(f"\n  CROSS-BOOTSTRAP STD (stability):")
    labels = ["elig_pop","FE_total","fe_a0(fixed)","fe_price(fixed)","fe_xi",
              "elig_adj","adj","pcd","error"]
    arrs   = [comp_d[:,i] for i in range(8)] + [error_d]
    for lbl, arr in zip(labels, arrs):
        print(f"    {lbl:<25} std={arr.std():.5f}")

    # Point-estimate check (use theta_pt, no u_hat stored — just det part)
    det_pt = theta_pt @ agg_feat
    print(f"\n  POINT ESTIMATE det part = {det_pt:.4f} util = ${det_pt/ALPHA_1:.4f}B")
    print(f"  Bootstrap mean det part = {det.mean():.4f} util = ${det.mean()/ALPHA_1:.4f}B")


if __name__ == "__main__":
    main()
