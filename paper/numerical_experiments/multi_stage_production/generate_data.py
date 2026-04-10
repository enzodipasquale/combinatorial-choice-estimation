"""DGP for multi-stage EV production facility location experiment."""

import numpy as np
import gurobipy as gp
from milp import build_milp_shell
from costs import compute_rev_factor, compute_facility_costs


CONT_CENTERS = np.array([[-80.0, 40.0], [110.0, 30.0], [10.0, 50.0]])


# ---- Geography ----

def _scatter(centers, counts, rng, spread):
    n = sum(counts)
    locs = np.empty((n, 2))
    conts = np.empty(n, dtype=int)
    i = 0
    for c, nc in enumerate(counts):
        locs[i:i+nc] = centers[c] + rng.normal(0, spread, (nc, 2))
        conts[i:i+nc] = c
        i += nc
    return locs, conts


def _pdist(a, b):
    return np.sqrt(((a[:, None] - b[None]) ** 2).sum(-1))


def _tariffs(cont_a, cont_b, rng):
    same = cont_a[:, None] == cont_b[None]
    return np.where(same, 0.0,
                    rng.uniform(0.05, 0.25, (len(cont_a), len(cont_b))))


def build_geography(cfg, rng):
    l1 = cfg['l1_per_continent']
    l2 = cfg['l2_per_continent']
    nc = cfg['n_continents']
    N = cfg['n_markets']

    loc1, c1 = _scatter(CONT_CENTERS, l1, rng, 10.0)
    loc2, c2 = _scatter(CONT_CENTERS, l2, rng, 10.0)
    mpc = N // nc
    loc_m, cm = _scatter(CONT_CENTERS, [mpc] * nc, rng, 15.0)
    loc_hq = CONT_CENTERS + rng.normal(0, 3, (nc, 2))

    return dict(
        L1=sum(l1), L2=sum(l2), n_markets=N,
        cont1=c1, cont2=c2, cont_m=cm,
        ln_d_12=np.log1p(_pdist(loc1, loc2)),
        ln_d_2m=np.log1p(_pdist(loc2, loc_m)),
        ln_d_hq1=np.log1p(_pdist(loc_hq, loc1)),
        ln_d_hq2=np.log1p(_pdist(loc_hq, loc2)),
        tau_12=_tariffs(c1, c2, rng),
        tau_2m=_tariffs(c2, cm, rng),
        R_n=rng.lognormal(5, 0.8, N),
    )


# ---- Firms ----

def build_firms(cfg, geo, rng):
    nf = cfg['n_firms']
    nc = cfg['n_continents']
    ng = cfg['n_groups_cells']
    np_total = cfg['n_platforms']
    lo, hi = cfg['models_range']
    fpc = max(nf // nc, 1)

    firms = []
    for f in range(nf):
        hq = min(f // fpc, nc - 1)
        nm = int(rng.integers(lo, hi + 1))
        cg = rng.integers(0, ng, nm)
        n_plat = int(rng.integers(2, min(nm, np_total) + 1))
        plat = rng.integers(0, n_plat, nm)

        # Market feasibility: home continent always, foreign ~40%
        home = geo['cont_m'] == hq
        feas = np.zeros((nm, geo['n_markets']), dtype=bool)
        feas[:, home] = True
        nf_mkt = (~home).sum()
        feas[:, ~home] = rng.random((nm, nf_mkt)) < 0.4

        # Shares: Dirichlet per market for this firm's active models
        shares = np.zeros_like(feas, dtype=float)
        for n in range(geo['n_markets']):
            k = feas[:, n].sum()
            if k > 0:
                ev_frac = (0.3 + 0.3 * rng.random()) / nf
                shares[feas[:, n], n] = rng.dirichlet(np.ones(k)) * ev_frac

        firms.append(dict(
            hq_cont=hq, n_models=nm, cell_groups=cg,
            n_platforms=n_plat, platforms=plat,
            feasible=feas, shares=shares,
            ln_xi_1=rng.normal(0, 0.5, ng),
            ln_xi_2=rng.normal(0, 0.5, n_plat),
        ))
    return firms


# ---- MILP ----

def draw_firm_errors(firm, geo, sigma, rng):
    P = firm['n_platforms']
    nm = firm['n_models']
    L1, L2, N = geo['L1'], geo['L2'], geo['n_markets']
    return dict(
        phi1=rng.normal(0, sigma['phi_1'], (len(firm['ln_xi_1']), L1)),
        phi2=rng.normal(0, sigma['phi_2'], (P, L2)),
        nu=rng.normal(0, sigma['nu'], (nm, N, L1, L2)),
    )


def build_firm_milp(firm, geo, rev_factor, fc1, fc2, errors, env):
    nm, P = firm['n_models'], firm['n_platforms']
    N, L1, L2 = geo['n_markets'], geo['L1'], geo['L2']
    ng = fc1.shape[0]

    mdl = gp.Model(env=env)
    y1, y2, z, x = build_milp_shell(
        mdl, ng, P, nm, N, L1, L2,
        firm['feasible'], firm['cell_groups'], firm['platforms'])

    rf = rev_factor.transpose(2, 0, 1)                              # (N, L1, L2)
    pi = (firm['shares'][:, :, None, None]
          * geo['R_n'][None, :, None, None]
          * rf[None, :, :, :])                                       # (nm, N, L1, L2)

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
    rng = np.random.default_rng(seed)
    geo = build_geography(dgp, rng)
    firms = build_firms(dgp, geo, rng)
    rf = compute_rev_factor(geo, theta_true, sourcing_coefs)

    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()

    bundles = []
    for firm in firms:
        fc1, fc2 = compute_facility_costs(firm, geo, theta_true)
        errors = draw_firm_errors(firm, geo, sigma, rng)
        mdl, y1, y2, z, x = build_firm_milp(
            firm, geo, rf, fc1, fc2, errors, env)
        mdl.optimize()
        bundles.append(extract_solution(mdl, y1, y2, z, x, errors))
    env.dispose()

    return geo, firms, bundles, theta_true


if __name__ == '__main__':
    import yaml
    from pathlib import Path

    with open(Path(__file__).parent / 'config.yaml') as f:
        cfg = yaml.safe_load(f)

    test_dgp = dict(
        n_firms=3, n_markets=6, n_continents=3,
        l1_per_continent=[1, 1, 1],
        l2_per_continent=[1, 1, 1],
        n_groups_cells=2, n_platforms=3,
        models_range=[2, 4],
    )

    geo, firms, bundles, theta_true = generate_synthetic_data(
        seed=42, dgp=test_dgp,
        sourcing_coefs=cfg['sourcing_coefs'],
        theta_true=cfg['theta_true'],
        sigma=cfg['sigma'],
    )

    CNAMES = ['Am', 'As', 'Eu']
    for f, (firm, bun) in enumerate(zip(firms, bundles)):
        hq = CNAMES[firm['hq_cont']]
        n_paths = (bun['x'] > 0.5).sum()
        print(f"\nFirm {f} (HQ={hq}, {firm['n_models']} models, "
              f"{firm['n_platforms']} platforms):")
        print(f"  Cells: {bun['y1'].sum():.0f}  Asm: {bun['y2'].sum():.0f}  "
              f"Markets: {bun['z'].sum():.0f}  Paths: {n_paths:.0f}  "
              f"Obj: {bun['obj']:.2f}")
        for g in range(bun['y1'].shape[0]):
            locs = np.where(bun['y1'][g])[0]
            if len(locs):
                cs = [CNAMES[geo['cont1'][l]] for l in locs]
                print(f"  Cell group {g}: {locs.tolist()} ({cs})")
        for p in range(bun['y2'].shape[0]):
            locs = np.where(bun['y2'][p])[0]
            if len(locs):
                cs = [CNAMES[geo['cont2'][l]] for l in locs]
                print(f"  Asm plat {p}:   {locs.tolist()} ({cs})")
