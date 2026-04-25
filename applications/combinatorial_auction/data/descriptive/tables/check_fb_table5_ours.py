"""Fox-Bajari Table 5 analog using OUR 9 parameter estimates.

Same five allocations as FB Table 5 but scored with the coefficients from our
pop_scaling_large_2 point estimate (θ̂ from results/.../result.json), with the
standard normalization β_elig ≡ 1 (i.e. all other coefs divided by θ̂_elig_pop).

Our valuation for bidder i winning package J_i is

    π(i, J_i) = θ̂_elig · elig_i · Σ_{j∈J_i} pop_j
              + Σ_c θ̂_c · Σ_{j,l∈J_i} Q_c[j,l]          (item-quadratic)
              + Σ_c θ̂_{elig×c} · elig_i · Σ_{j,l∈J_i} Q_c[j,l]   (id-quadratic)

Summed over bidders and normalized by θ̂_elig gives the analog of FB Table 5.
"""
import json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))

from applications.combinatorial_auction.data.prepare import prepare
from applications.combinatorial_auction.data.loaders import load_raw, load_aggregation_matrix
from applications.combinatorial_auction.data.descriptive.maps import xwalk


QUAD_NAMES   = ["adjacency", "pop_centroid_delta4", "air_travel", "travel_survey"]
QID_NAMES    = [f"elig_{v}" for v in QUAD_NAMES]
MOD_NAMES    = ["elig_pop"]                      # single modular: elig · pop


REGION_BY_MTA = {
    # Northeast
    1:  "NE", 8:  "NE", 9:  "NE", 10: "NE", 35: "NE",
    # Midwest
    3:  "MW", 5:  "MW", 12: "MW", 16: "MW", 18: "MW", 19: "MW", 20: "MW",
    21: "MW", 26: "MW", 31: "MW", 32: "MW", 34: "MW", 38: "MW", 45: "MW", 46: "MW",
    # South
    6:  "S",  7:  "S",  11: "S",  13: "S",  14: "S",  15: "S",  17: "S",
    23: "S",  28: "S",  29: "S",  33: "S",  37: "S",  40: "S",  41: "S",
    43: "S",  44: "S",  48: "S",
    # West
    2:  "W",  4:  "W",  22: "W",  24: "W",  27: "W",  30: "W",  36: "W",
    39: "W",  42: "W",
}


def per_bidder_xbar(b, pop, elig, Q):
    """For indicator matrix b (n_bidders, n_items), return:
         xbar_mod      scalar      — Σ_i elig_i · b_i · pop
         xbar_quad     (K,)        — Σ_i Σ_{j,l} Q_c[j,l] b_ij b_il   per covariate
         xbar_quad_id  (K,)        — Σ_i elig_i · Σ_{j,l} Q_c[j,l] b_ij b_il
    """
    B = b.astype(float)
    xbar_mod     = float(np.sum(elig[:, None] * B * pop[None, :]))
    xbar_quad    = np.einsum("ij,jlk,il->k", B, Q, B, optimize=True)
    xbar_quad_id = np.einsum("i,ij,jlk,il->k", elig, B, Q, B, optimize=True)
    return xbar_mod, xbar_quad, xbar_quad_id


def load_theta_hat(spec):
    """Pull the 9 named θ̂s from the point-estimate result.json in the
    canonical combest layout:
        [ modular | BTA-FEs (n_items) | quadratic_id | quadratic ]
    """
    pt = json.load(open(REPO / "applications/combinatorial_auction/results"
                        / spec / "point_estimate/result.json"))
    theta = np.asarray(pt["theta_hat"])
    n_mod = len(MOD_NAMES); n_qid = len(QID_NAMES); n_quad = len(QUAD_NAMES); n_items = 480
    quad_start = n_mod + n_items
    th_mod  = theta[:n_mod]
    th_qid  = theta[quad_start:quad_start + n_qid]
    th_quad = theta[quad_start + n_qid:quad_start + n_qid + n_quad]
    return dict(zip(MOD_NAMES, th_mod)) | dict(zip(QID_NAMES, th_qid)) | dict(zip(QUAD_NAMES, th_quad))


def compute_fb_table5(spec="pop_scaling_large_2", theta_overrides=None):
    """Return the FB Table-5-style allocation totals.

    If ``theta_overrides`` is None, uses our 9 estimated coefficients
    (normalized so β_elig_pop = 1).  Pass a {name: value} dict to score the
    same allocations under arbitrary normalized coefs (e.g., FB's 4 numbers).

    Returns dict mapping allocation label → {covariate: contribution, total}.
    """
    raw = load_raw()
    input_data, meta = prepare(
        modular_regressors       = MOD_NAMES,
        quadratic_regressors     = QUAD_NAMES,
        quadratic_id_regressors  = QID_NAMES,
    )
    Q           = input_data["item_data"]["quadratic"]   # (n_btas, n_btas, 4)
    obs_bundles = input_data["id_data"]["obs_bundles"]
    elig        = input_data["id_data"]["elig"]
    pop         = raw["bta_data"]["pop90_share"].to_numpy(float)
    btas = raw["bta_data"]["bta"].astype(int).values
    A, mta_nums = load_aggregation_matrix(btas)
    n_btas  = A.shape[1]; n_mtas = A.shape[0]
    n_bidders = len(raw["bidder_data"])

    # θ̂ from point estimate; normalize by θ̂(elig_pop) unless overridden
    if theta_overrides is None:
        theta_hat = load_theta_hat(spec)
        θ_elig_raw = theta_hat["elig_pop"]
        θ = {k: v / θ_elig_raw for k, v in theta_hat.items()}
    else:
        θ = {**{k: 0.0 for k in [MOD_NAMES[0]] + QUAD_NAMES + QID_NAMES},
             **theta_overrides}

    # Assortative orderings
    is_winner = obs_bundles.sum(1) > 0
    n_winners = int(is_winner.sum())
    sorted_winners = np.argsort(-np.where(is_winner, elig, -np.inf))
    pop_order = np.argsort(-pop)
    pop_mta = A @ pop
    mta_order = np.argsort(-pop_mta)

    # Build the 5 allocations as (n_bidders, n_btas) indicator matrices ────
    def alloc_actual():
        return obs_bundles.astype(bool)

    def alloc_480_separate():
        # 480 singletons, assortatively matched in desc pop to eligibilities.
        # Each singleton ⇒ complementarity = 0 trivially.
        return None   # handled analytically below

    def alloc_47_mta():
        b = np.zeros((n_bidders, n_btas), bool)
        for rank, m_idx in enumerate(mta_order):
            i = int(sorted_winners[min(rank, n_winners - 1)])
            b[i] |= A[m_idx].astype(bool)
        return b

    def alloc_4_regional():
        region_bundles = {}
        for i_mta, m_num in enumerate(mta_nums):
            r = REGION_BY_MTA.get(int(m_num))
            if r is None: continue
            region_bundles.setdefault(r, np.zeros(n_btas, bool))
            region_bundles[r] |= A[i_mta].astype(bool)
        region_pops = {r: float(b @ pop) for r, b in region_bundles.items()}
        regions_desc = sorted(region_bundles, key=lambda r: -region_pops[r])
        b = np.zeros((n_bidders, n_btas), bool)
        for rank, r in enumerate(regions_desc):
            b[int(sorted_winners[rank])] = region_bundles[r]
        return b

    def alloc_nationwide():
        nw_idx = int(raw["bidder_data"].index[raw["bidder_data"]["bidder_num_fox"] == 77][0])
        b = np.zeros((n_bidders, n_btas), bool)
        b[nw_idx] = True
        return b

    b_actual     = alloc_actual()
    b_47mta      = alloc_47_mta()
    b_4regional  = alloc_4_regional()
    b_nationwide = alloc_nationwide()

    # ── Compute decompositions ────────────────────────────────────────
    def decomp(b, label_hint=None):
        mod, quad, qid = per_bidder_xbar(b, pop, elig, Q)
        contrib = {MOD_NAMES[0]: θ[MOD_NAMES[0]] * mod}
        for c, name in enumerate(QUAD_NAMES):
            contrib[name] = θ[name] * quad[c]
        for c, name in enumerate(QID_NAMES):
            contrib[name] = θ[name] * qid[c]
        return contrib

    # Allocation 2: 480 separate bidders — singletons, all quadratics = 0.
    # Only elig·Σpop matters; compute directly.
    pop_order_desc = np.argsort(-pop)
    elig_desc = np.sort(elig)[::-1]
    elig_min = float(elig.min())
    assigned_elig = np.empty(n_btas)
    for rank in range(n_btas):
        j = int(pop_order_desc[rank])
        assigned_elig[j] = elig_desc[rank] if rank < n_bidders else elig_min
    xbar_mod_480 = float((assigned_elig * pop).sum())
    contrib_480 = {MOD_NAMES[0]: θ[MOD_NAMES[0]] * xbar_mod_480}
    for n in QUAD_NAMES + QID_NAMES:
        contrib_480[n] = 0.0

    all_rows = [
        ("(1) C block, 85 winning packages", decomp(b_actual)),
        ("(2) All 480 licenses, diff bidders", contrib_480),
        ("(3) Each 47 MTAs, separate package", decomp(b_47mta)),
        ("(4) Four large regional packages",   decomp(b_4regional)),
        ("(5) Nationwide license (NextWave)",  decomp(b_nationwide)),
    ]
    return dict(coefs=θ, rows=all_rows)


def main():
    res = compute_fb_table5("pop_scaling_large_2")
    θ, all_rows = res["coefs"], res["rows"]

    # Print table: each named contribution + total
    cols_order = [MOD_NAMES[0]] + QUAD_NAMES + QID_NAMES
    headers = ["elig_pop", "adj", "gravity", "air", "comm",
               "e·adj", "e·grav", "e·air", "e·comm", "TOTAL"]
    print(f"\nNormalized coefficients (β_elig_pop ≡ 1):")
    for k, v in θ.items():
        print(f"  {k:<32}  {v:+7.4f}")
    print()
    print(f"{'Allocation':<42} " + " ".join(f"{h:>7}" for h in headers))
    print("-" * 42 + " " + "-" * (8 * 10))
    for lbl, contrib in all_rows:
        row = [contrib[c] for c in cols_order]
        total = sum(row)
        print(f"{lbl:<42} " + " ".join(f"{v:+7.3f}" for v in row + [total]))
    print(f"\n{'Allocation':<42}  {'TOTAL (β_elig=1)':>18}")
    print("-" * 62)
    nw_total = sum(all_rows[-1][1].values())
    for lbl, contrib in all_rows:
        tot = sum(contrib.values())
        print(f"{lbl:<42}  {tot:>12.4f}   ({tot/nw_total:>5.2f} × nationwide)")


if __name__ == "__main__":
    main()
