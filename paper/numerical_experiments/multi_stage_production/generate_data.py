"""DGP for multi-stage production facility location — HMMY Section 5 template.

Symmetric geography on [0,1]^2 torus, torus distances, per-firm HQ, region FEs.
Firms: nm~U[1,3] models, ng=1 cell group, P~U[1,nm] platforms.
"""

import numpy as np
import gurobipy as gp
from milp import build_milp_shell


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

    return dict(
        L1=L1, L2=L2, n_markets=N,
        cell_locs=cell_locs, asm_locs=asm_locs, mkt_locs=mkt_locs,
        d_12=d_12, d_2m=d_2m, R_n=R_n,
        cell_region=cell_region, asm_region=asm_region,
    )


# ---- Firms ----

def build_firms(cfg, geo, rng):
    """Firms with configurable model/platform ranges."""
    nf = cfg['n_firms']
    N = geo['n_markets']
    nm_lo, nm_hi = cfg.get('models_range', [4, 8])
    P_max_cap = cfg.get('max_platforms', 5)
    P_min = cfg.get('min_platforms', 2)
    firms = []
    for i in range(nf):
        nm = int(rng.integers(nm_lo, nm_hi + 1))
        P = int(rng.integers(P_min, min(nm, P_max_cap) + 1))
        plat = rng.integers(0, P, nm).astype(int)              # platform assignment per model

        shares = np.zeros((nm, N))
        for m in range(nm):
            shares[m] = rng.uniform(0.5, 1.5, N) / 100.0

        # HQ in firm's assigned region strip (i % 3)
        hq_region = i % 3
        hq_x = rng.uniform(hq_region / 3.0, (hq_region + 1) / 3.0)
        hq_y = rng.uniform(0, 1)
        hq_coord = np.array([[hq_x, hq_y]])
        d_hq1 = np.log1p(_torus_dist(hq_coord, geo['cell_locs']).ravel())  # (L1,)
        d_hq2 = np.log1p(_torus_dist(hq_coord, geo['asm_locs']).ravel())  # (L2,)

        firms.append(dict(
            n_models=nm, n_platforms=P,
            cell_groups=np.zeros(nm, dtype=int),               # ng=1: all models in group 0
            platforms=plat,
            feasible=np.ones((nm, N), dtype=bool),
            shares=shares,                                     # (nm, N)
            ln_xi_1=np.array([rng.normal(0, 0.5)]),           # (1,) one cell group
            ln_xi_2=rng.normal(0, 0.5, P),                    # (P,) one per platform
            d_hq1=d_hq1,                                      # (L1,)
            d_hq2=d_hq2,                                      # (L2,)
            hq_coord=hq_coord.ravel(),                        # (2,) for plotting
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

    fc1 = (theta['delta_1']
           + theta['rho_xi_1'] * firm['ln_xi_1'][0]
           + theta['rho_HQ_1'] * firm['d_hq1']
           - fe1[geo['cell_region']])                          # (L1,)
    fc2 = (theta['delta_2']
           + theta['rho_xi_2'] * firm['ln_xi_2'][:, None]     # (P, 1)
           + theta['rho_HQ_2'] * firm['d_hq2'][None, :]       # (1, L2)
           - fe2[geo['asm_region']][None, :])                  # (1, L2) → (P, L2)
    return fc1[None, :], fc2                                   # (1, L1), (P, L2)


# ---- MILP ----

def draw_firm_errors(firm, geo, sigma, rng):
    """Errors for a firm with ng=1, P platforms, nm models."""
    L1, L2, N = geo['L1'], geo['L2'], geo['n_markets']
    P, nm = firm['n_platforms'], firm['n_models']
    return dict(
        phi1=rng.normal(0, sigma['phi'], (1, L1)),
        phi2=rng.normal(0, sigma['phi'], (P, L2)),
        nu=rng.normal(0, sigma['nu'], (nm, N, L1, L2)),
    )


def build_firm_milp(firm, geo, pi, fc1, fc2, errors, env):
    """fc1: (1, L1), fc2: (P, L2). pi: (nm, N, L1, L2)."""
    N, L1, L2 = geo['n_markets'], geo['L1'], geo['L2']
    nm, P = firm['n_models'], firm['n_platforms']

    mdl = gp.Model(env=env)
    y1, y2, z, x = build_milp_shell(
        mdl, 1, P, nm, N, L1, L2,
        firm['feasible'], firm['cell_groups'], firm['platforms'])

    mdl.setObjective(
        (pi + errors['nu']).ravel() @ x.reshape(-1)
        - (fc1 + errors['phi1']).ravel() @ y1.reshape(-1)
        - (fc2 + errors['phi2']).ravel() @ y2.reshape(-1),
        gp.GRB.MAXIMIZE)

    return mdl, y1, y2, z, x


def extract_solution(mdl, y1, y2, z, x, errors):
    return dict(
        y1=np.asarray(y1.X) > 0.5,
        y2=np.asarray(y2.X) > 0.5,
        z=np.asarray(z.X) > 0.5,
        x=np.asarray(x.X),
        obj=mdl.ObjVal,
        phi1=errors['phi1'],
        phi2=errors['phi2'],
        nu=errors['nu'],
    )


# ---- Main entry ----

def generate_synthetic_data(seed, dgp, sourcing_coefs, theta_true, sigma):
    """Public API — sourcing_coefs accepted for signature compatibility but unused."""
    rng = np.random.default_rng(seed)
    geo = build_geography(dgp, rng)
    firms = build_firms(dgp, geo, rng)

    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()

    bundles = []
    for firm in firms:
        pi = compute_pi(firm, geo, theta_true)
        fc1, fc2 = compute_facility_costs(firm, geo, theta_true)
        errors = draw_firm_errors(firm, geo, sigma, rng)
        mdl, y1, y2, z, x = build_firm_milp(
            firm, geo, pi, fc1, fc2, errors, env)
        mdl.optimize()
        bundles.append(extract_solution(mdl, y1, y2, z, x, errors))
    env.dispose()

    return geo, firms, bundles, theta_true


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
    geo, firms, bundles, _ = generate_synthetic_data(
        seed=seed, dgp=dgp, sourcing_coefs=None,
        theta_true=theta_true, sigma=sigma)
    elapsed = time.perf_counter() - t0

    print(f"DGP generation time: {elapsed:.2f}s")
    print(f"Geography: L1={geo['L1']}, L2={geo['L2']}, N={geo['n_markets']}")
    print(f"d_12 range: [{geo['d_12'].min():.3f}, {geo['d_12'].max():.3f}]")
    print(f"d_2m range: [{geo['d_2m'].min():.3f}, {geo['d_2m'].max():.3f}]")
    print(f"R_n range:  [{geo['R_n'].min():.2f}, {geo['R_n'].max():.2f}]")
    cr, ar = geo['cell_region'], geo['asm_region']
    print(f"Cell regions: {cr.tolist()}  (r0:{(cr==0).sum()} r1:{(cr==1).sum()} r2:{(cr==2).sum()})")
    print(f"Asm regions:  {ar.tolist()}  (r0:{(ar==0).sum()} r1:{(ar==1).sum()} r2:{(ar==2).sum()})")
    print(f"Firms: {len(firms)}")
    nm_dist = Counter(f['n_models'] for f in firms)
    P_dist = Counter(f['n_platforms'] for f in firms)
    print(f"Models dist: {dict(sorted(nm_dist.items()))}")
    print(f"Platforms dist: {dict(sorted(P_dist.items()))}")
    print()

    # Objective stats
    objs = [b['obj'] for b in bundles]
    print(f"Objective: min={min(objs):.2f}  mean={np.mean(objs):.2f}  max={max(objs):.2f}")

    # Market entry distribution — count distinct markets per firm (across all models)
    mkts_per_firm = []
    for bun in bundles:
        entered = set()
        for m in range(bun['z'].shape[0]):
            entered.update(np.where(bun['z'][m])[0])
        mkts_per_firm.append(len(entered))
    print(f"Distinct markets entered per firm: min={min(mkts_per_firm)}  "
          f"mean={np.mean(mkts_per_firm):.1f}  max={max(mkts_per_firm)}")

    # Per-firm summary
    n_active = sum(1 for b in bundles if b['obj'] > 0)
    avg_cells = np.mean([b['y1'].sum() for b in bundles])
    avg_asms = np.mean([b['y2'].sum() for b in bundles])
    print(f"Active firms (obj>0): {n_active}/{len(firms)}")
    print(f"Avg cells: {avg_cells:.1f}  avg asms: {avg_asms:.1f}")

    # Assembly usage per platform
    asm_counts = np.zeros(geo['L2'], dtype=int)
    for bun in bundles:
        for p in range(bun['y2'].shape[0]):
            asm_counts += bun['y2'][p].astype(int)
    print(f"\nAssembly usage (total opens across all firms/platforms):")
    for l in range(geo['L2']):
        bar = '#' * min(asm_counts[l], 80)
        print(f"  asm {l} (region {geo['asm_region'][l]}): {asm_counts[l]:3d}  {bar}")
