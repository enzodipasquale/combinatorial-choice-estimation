"""DGP for multi-stage production facility location.

Inspired by HMMY (2026) multi-stage production networks, adapted to a
synthetic unit-torus geometry for numerical experiments.

Geography
---------
- L1 cell-plant locations, L2 assembly locations, N markets drawn uniformly
  on [0,1]^2 torus; torus distances.
- Regions r in {0,1,2} partition locations by x-tercile.

Firms (n_firms total)
---------------------
- Models per firm nm ~ U[models_range] (default [4,8]).
- Cell groups per firm ng ~ U[cell_groups_range] (default [1,2]).
- Platforms per firm P ~ U[min_platforms, min(nm, max_platforms)].
- Quality shocks ln_xi_1 ~ N(0, 0.5) per group, ln_xi_2 per platform.

Identification design (key structural choices)
----------------------------------------------
- Firm i's HQ region: i % 3 (logical label used for focus-offset logic).
- HQ *position*: drawn from markets in hq_region, stratified so each firm
  gets a distinct market as its HQ. Markets scatter in [0,1]^2 uniformly,
  so HQs have genuine 2D spread even at small N.
- Market focus region: (hq_region + focus_offset) % 3 with focus_offset
  in {1, 2} -> focus is ALWAYS different from HQ region.
- Share boost: markets in firm's focus region get (1 + share_focus_boost)x
  shares. This forces cross-firm heterogeneity in which region each firm
  serves, decoupling delta (count of opens) from rho_HQ (sum of d_HQ
  over opens): two firms sharing a focus region but with different HQ
  regions open in the same region but at very different d_HQ.
- Per-model feasibility: two tiers (global vs regional), with >=1 market
  guaranteed from each region.
"""

import numpy as np


# ---- Geography ----

def _torus_dist(a, b):
    """Torus distance on [0,1]^2: wrap-around in both dimensions."""
    diff = a[:, None, :] - b[None, :, :]
    diff = np.minimum(np.abs(diff), 1 - np.abs(diff))
    return np.sqrt((diff ** 2).sum(-1))


def build_geography(cfg, rng):
    """Random locations on [0,1]^2 torus; torus distances; R_n ~ U(30,60)."""
    L1 = cfg['L1']
    L2 = cfg['L2']
    N = cfg['N']

    cell_locs = rng.uniform(0, 1, (L1, 2))
    asm_locs = rng.uniform(0, 1, (L2, 2))
    mkt_locs = rng.uniform(0, 1, (N, 2))

    # Torus distance matrices
    d_12 = _torus_dist(cell_locs, asm_locs)                    # (L1, L2)
    d_2m = _torus_dist(asm_locs, mkt_locs)                     # (L2, N)

    R_n = rng.uniform(30, 60, N)

    # Region partition: sort by x-coordinate into 3 thirds
    cell_order = np.argsort(cell_locs[:, 0])
    cell_region = np.zeros(L1, dtype=int)
    cell_region[cell_order[L1 // 3: 2 * L1 // 3]] = 1
    cell_region[cell_order[2 * L1 // 3:]] = 2

    asm_order = np.argsort(asm_locs[:, 0])
    asm_region = np.zeros(L2, dtype=int)
    asm_region[asm_order[L2 // 3: 2 * L2 // 3]] = 1
    asm_region[asm_order[2 * L2 // 3:]] = 2

    mkt_order = np.argsort(mkt_locs[:, 0])
    mkt_region = np.zeros(N, dtype=int)
    mkt_region[mkt_order[N // 3: 2 * N // 3]] = 1
    mkt_region[mkt_order[2 * N // 3:]] = 2

    return dict(
        L1=L1, L2=L2, n_markets=N,
        cell_locs=cell_locs, asm_locs=asm_locs, mkt_locs=mkt_locs,
        d_12=d_12, d_2m=d_2m, R_n=R_n,
        cell_region=cell_region, asm_region=asm_region, mkt_region=mkt_region,
    )


# ---- Firms ----

def _draw_feasible_markets(nm, N, mkt_region, feas_cfg, rng):
    """Draw per-model feasibility masks with two tiers and regional coverage."""
    global_prob = feas_cfg.get('global_prob', 1.0)
    gl_lo, gl_hi = feas_cfg.get('global_range', [N, N])
    rg_lo, rg_hi = feas_cfg.get('regional_range', [4, 7])

    feas = np.zeros((nm, N), dtype=bool)
    regions = [np.where(mkt_region == r)[0] for r in range(3)]

    for m in range(nm):
        is_global = rng.random() < global_prob
        k = int(rng.integers(gl_lo, gl_hi + 1)) if is_global \
            else int(rng.integers(rg_lo, rg_hi + 1))
        k = min(k, N)

        # Guarantee ≥1 market from each region
        chosen = []
        for reg_mkts in regions:
            if len(reg_mkts) > 0:
                chosen.append(rng.choice(reg_mkts))
        chosen = list(set(chosen))

        # Fill remaining slots from unchosen markets
        remaining = np.setdiff1d(np.arange(N), chosen)
        n_extra = max(0, k - len(chosen))
        if n_extra > 0 and len(remaining) > 0:
            extra = rng.choice(remaining, size=min(n_extra, len(remaining)), replace=False)
            chosen.extend(extra.tolist())

        feas[m, chosen] = True

    return feas


def build_firms(cfg, geo, rng):
    """Firms with configurable model/platform ranges and per-model feasibility."""
    nf = cfg['n_firms']
    N = geo['n_markets']
    nm_lo, nm_hi = cfg.get('models_range', [4, 8])
    ng_lo, ng_hi = cfg.get('cell_groups_range', [1, 1])
    P_max_cap = cfg.get('max_platforms', 5)
    P_min = cfg.get('min_platforms', 2)
    feas_cfg = cfg.get('feasibility', {})
    mkt_region = geo['mkt_region']
    mkt_locs = geo['mkt_locs']

    share_boost = cfg.get('share_focus_boost', 2.0)

    # Stratified HQ assignment: each firm's HQ is AT a randomly drawn market
    # in its assigned region. Markets are uniformly scattered in [0,1]^2, so
    # firms in the same region have genuinely diverse HQ positions (unlike
    # narrow x-strip placement which only allows y-variation that torus
    # geometry partially cancels).
    n_per_strip = max(1, (nf + 2) // 3)
    hq_markets_by_region = {}
    for r in range(3):
        mkts_in_r = np.where(mkt_region == r)[0]
        if len(mkts_in_r) < n_per_strip:
            # Wrap with replacement if not enough markets in a region
            choice = rng.choice(mkts_in_r, size=n_per_strip, replace=True)
        else:
            choice = rng.choice(mkts_in_r, size=n_per_strip, replace=False)
        hq_markets_by_region[r] = choice

    firms = []
    for i in range(nf):
        nm = int(rng.integers(nm_lo, nm_hi + 1))
        ng = int(rng.integers(ng_lo, ng_hi + 1))
        P = int(rng.integers(P_min, min(nm, P_max_cap) + 1))
        cg = rng.integers(0, ng, nm).astype(int)               # cell group per model
        plat = rng.integers(0, P, nm).astype(int)

        # Per-model feasibility
        feas = _draw_feasible_markets(nm, N, mkt_region, feas_cfg, rng)

        # HQ region (logical label for focus-offset logic)
        hq_region = i % 3
        firm_in_strip = i // 3

        # Market focus region: ALWAYS different from HQ region (offset in {1,2}).
        # Two firms with same focus but different HQ open in same region but
        # at very different d_HQ -> decouples delta from rho_HQ.
        focus_offset = 1 + (firm_in_strip % 2)   # in {1, 2}
        mkt_focus = (hq_region + focus_offset) % 3

        # Shares: higher in focus region (boost factor = share_focus_boost)
        shares = np.zeros((nm, N))
        focus_mask = (mkt_region == mkt_focus).astype(float)
        for m in range(nm):
            base = rng.uniform(0.5, 1.5, N) / 100.0
            shares[m] = base * (1.0 + share_boost * focus_mask)

        # HQ position: at a drawn market in hq_region. Markets scatter in
        # [0,1]^2 uniformly, giving genuine 2D spread of HQs even at small N.
        hq_mkt_idx = int(hq_markets_by_region[hq_region][firm_in_strip % n_per_strip])
        hq_coord = mkt_locs[hq_mkt_idx:hq_mkt_idx + 1]
        d_hq1 = np.log1p(_torus_dist(hq_coord, geo['cell_locs']).ravel())
        d_hq2 = np.log1p(_torus_dist(hq_coord, geo['asm_locs']).ravel())

        firms.append(dict(
            n_models=nm, n_platforms=P,
            cell_groups=cg,
            platforms=plat,
            feasible=feas,
            shares=shares,
            ln_xi_1=rng.normal(0, 0.5, ng),                    # (ng,) one per cell group
            ln_xi_2=rng.normal(0, 0.5, P),
            d_hq1=d_hq1,
            d_hq2=d_hq2,
            hq_coord=hq_coord.ravel(),
            hq_region=hq_region,
            mkt_focus=mkt_focus,
        ))
    return firms


# ---- Revenue and costs ----

def compute_pi(firm, geo, theta):
    """Per-path revenue with distance penalties (no FE — FE moved to cost side).

    pi_{m,n,l1,l2} = s_{m,n} * R_n * (1 - rho_d_1*d_{l1,l2} - rho_d_2*d_{l2,n})

    Returns shape (nm, N, L1, L2).
    """
    d12 = geo['d_12']                                           # (L1, L2)
    d2m = geo['d_2m']                                           # (L2, N)

    # rev_factor (N, L1, L2) — shared across models
    rev_factor = (1.0
                  - theta['rho_d_1'] * d12[None, :, :]
                  - theta['rho_d_2'] * d2m.T[:, None, :])

    # Per-model shares: sR[m, n] = shares[m, n] * R_n
    sR = firm['shares'] * geo['R_n'][None, :]                  # (nm, N)
    pi = sR[:, :, None, None] * rev_factor[None, :, :, :]     # (nm, N, L1, L2)
    return pi


def compute_facility_costs(firm, geo, theta):
    """Cell cost (1, L1) and assembly cost (P, L2) with 3-region FE on cost side.

    fc1[l1] = delta_1 + rho_xi_1*ln_xi_1 + rho_HQ_1*d_hq1[l1] - fe1[region[l1]]
    fc2[p,l2] = delta_2 + rho_xi_2*ln_xi_2[p] + rho_HQ_2*d_hq2[l2] - fe2[region[l2]]

    fe1 = [0, FE_1_r1, FE_1_r2], fe2 = [0, FE_2_r1, FE_2_r2]
    Region 0 normalized to zero. Positive FE = lower cost.
    """
    fe1 = np.array([0.0, theta.get('FE_1_r1', 0.0), theta.get('FE_1_r2', 0.0)])
    fe2 = np.array([0.0, theta.get('FE_2_r1', 0.0), theta.get('FE_2_r2', 0.0)])

    ng = len(firm['ln_xi_1'])
    fc1 = (theta['delta_1']
           + theta['rho_xi_1'] * firm['ln_xi_1'][:, None]     # (ng, 1)
           + theta['rho_HQ_1'] * firm['d_hq1'][None, :]       # (1, L1)
           - fe1[geo['cell_region']][None, :])                 # (1, L1) → (ng, L1)
    fc2 = (theta['delta_2']
           + theta['rho_xi_2'] * firm['ln_xi_2'][:, None]     # (P, 1)
           + theta['rho_HQ_2'] * firm['d_hq2'][None, :]       # (1, L2)
           - fe2[geo['asm_region']][None, :])                  # (1, L2) → (P, L2)
    return fc1, fc2                                            # (ng, L1), (P, L2)


# ---- Errors ----

def draw_firm_errors(firm, geo, sigma, rng):
    """Errors for a firm with ng=1, P platforms, nm models."""
    L1, L2, N = geo['L1'], geo['L2'], geo['n_markets']
    P, nm = firm['n_platforms'], firm['n_models']
    return dict(
        phi1=rng.normal(0, sigma['phi'], (1, L1)),
        phi2=rng.normal(0, sigma['phi'], (P, L2)),
        nu=rng.normal(0, sigma['nu'], (nm, N, L1, L2)),
    )


# ---- Main entry ----

def generate_synthetic_data(seed, dgp, sourcing_coefs, theta_true, sigma):
    """Generate geography, firms, and DGP errors. No Gurobi — solving is done via combest."""
    rng = np.random.default_rng(seed)
    geo = build_geography(dgp, rng)
    firms = build_firms(dgp, geo, rng)

    # Draw DGP errors per firm
    dgp_errors = []
    for firm in firms:
        dgp_errors.append(draw_firm_errors(firm, geo, sigma, rng))

    return geo, firms, dgp_errors, theta_true


if __name__ == '__main__':
    import time
    import yaml
    from pathlib import Path
    from collections import Counter

    with open(Path(__file__).parent / 'config.yaml') as f:
        cfg = yaml.safe_load(f)

    dgp = cfg['dgp']
    theta_true = cfg['theta_true']
    sigma = cfg['sigma']

    seed = cfg.get('monte_carlo', {}).get('seed', 42)

    t0 = time.perf_counter()
    geo, firms, dgp_errors, _ = generate_synthetic_data(
        seed=seed, dgp=dgp, sourcing_coefs=None,
        theta_true=theta_true, sigma=sigma)
    elapsed = time.perf_counter() - t0

    print(f"DGP generation time: {elapsed:.2f}s")
    print(f"Geography: L1={geo['L1']}, L2={geo['L2']}, N={geo['n_markets']}")
    print(f"d_12 range: [{geo['d_12'].min():.3f}, {geo['d_12'].max():.3f}]")
    print(f"d_2m range: [{geo['d_2m'].min():.3f}, {geo['d_2m'].max():.3f}]")
    print(f"R_n range:  [{geo['R_n'].min():.2f}, {geo['R_n'].max():.2f}]")
    cr, ar, mr = geo['cell_region'], geo['asm_region'], geo['mkt_region']
    print(f"Cell regions:   (r0:{(cr==0).sum()} r1:{(cr==1).sum()} r2:{(cr==2).sum()})")
    print(f"Asm regions:    (r0:{(ar==0).sum()} r1:{(ar==1).sum()} r2:{(ar==2).sum()})")
    print(f"Market regions: (r0:{(mr==0).sum()} r1:{(mr==1).sum()} r2:{(mr==2).sum()})")
    print(f"Firms: {len(firms)}")
    nm_dist = Counter(f['n_models'] for f in firms)
    P_dist = Counter(f['n_platforms'] for f in firms)
    print(f"Models dist: {dict(sorted(nm_dist.items()))}")
    print(f"Platforms dist: {dict(sorted(P_dist.items()))}")
    print()

    # Feasibility stats
    feas_sizes = []
    for firm in firms:
        for m in range(firm['n_models']):
            feas_sizes.append(firm['feasible'][m].sum())
    feas_sizes = np.array(feas_sizes)
    print(f"Feasibility (markets per model): min={feas_sizes.min()} "
          f"mean={feas_sizes.mean():.1f} max={feas_sizes.max()}")
    print(f"  Size distribution: {dict(sorted(Counter(feas_sizes).items()))}")
