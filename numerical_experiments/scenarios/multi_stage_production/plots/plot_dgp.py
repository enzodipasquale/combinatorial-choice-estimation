"""Plot DGP: firm networks and facility usage. Reads config, solves MILPs independently."""

import sys
from pathlib import Path
import yaml
import numpy as np
import gurobipy as gp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PLOT_DIR = Path(__file__).resolve().parent
SCENARIO_DIR = PLOT_DIR.parent
sys.path.insert(0, str(SCENARIO_DIR))
from generate_data import generate_synthetic_data, compute_pi, compute_facility_costs, draw_firm_errors
from milp import build_milp_shell

with open(SCENARIO_DIR / 'config.yaml') as f:
    CFG = yaml.safe_load(f)

REGION_COLORS = ['tab:blue', 'tab:orange', 'tab:green']
REGION_LABELS = ['Region 0', 'Region 1', 'Region 2']


def solve_dgp(geo, firms, theta_true, sigma, seed):
    rng = np.random.default_rng(seed + 999)
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    bundles = []
    for firm in firms:
        pi = compute_pi(firm, geo, theta_true)
        fc1, fc2 = compute_facility_costs(firm, geo, theta_true)
        errors = draw_firm_errors(firm, geo, sigma, rng)
        nm, P = firm['n_models'], firm['n_platforms']
        ng = len(firm['ln_xi_1'])
        N, L1, L2 = geo['n_markets'], geo['L1'], geo['L2']
        mdl = gp.Model(env=env)
        y1, y2, z, x = build_milp_shell(
            mdl, ng, P, nm, N, L1, L2,
            firm['feasible'], firm['cell_groups'], firm['platforms'])
        mdl.setObjective(
            (pi + errors['nu']).ravel() @ x.reshape(-1)
            - (fc1 + errors['phi1']).ravel() @ y1.reshape(-1)
            - (fc2 + errors['phi2']).ravel() @ y2.reshape(-1),
            gp.GRB.MAXIMIZE)
        mdl.optimize()
        bundles.append(dict(
            y1=np.asarray(y1.X) > 0.5,
            y2=np.asarray(y2.X) > 0.5,
            z=np.asarray(z.X) > 0.5,
            x=np.asarray(x.X),
            obj=mdl.ObjVal,
        ))
    env.dispose()
    return bundles


def find_model_paths(firm, bun, geo):
    paths = []
    for m in range(firm['n_models']):
        active = bun['x'][m] > 0.5
        if not active.any():
            paths.append(None)
            continue
        x_sum = bun['x'][m].sum(axis=0)
        l1, l2 = np.unravel_index(x_sum.argmax(), x_sum.shape)
        markets = np.where(active[:, l1, l2])[0]
        paths.append(dict(cell=l1, asm=l2, markets=markets, platform=firm['platforms'][m]))
    return paths


def plot_facility_usage(geo, firms, bundles, seed, nf):
    """Per-firm stacked bar chart: cells and assemblies, colored by region."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x_pos = np.arange(nf)
    bar_w = 0.7

    # Cells: count opens per firm per region
    for ax, stage, y_key, region_key, title in [
        (ax1, 'Cells', 'y1', 'cell_region', 'Cell opens per firm'),
        (ax2, 'Assemblies', 'y2', 'asm_region', 'Assembly opens per firm'),
    ]:
        region_arr = geo[region_key]
        bottoms = np.zeros(nf)
        for r in range(3):
            counts = np.zeros(nf)
            for i, bun in enumerate(bundles):
                y = bun[y_key]
                for g in range(y.shape[0]):
                    for l in range(y.shape[1]):
                        if y[g, l] and region_arr[l] == r:
                            counts[i] += 1
            ax.bar(x_pos, counts, bar_w, bottom=bottoms,
                   color=REGION_COLORS[r], label=REGION_LABELS[r])
            bottoms += counts

        ax.set_xlabel('Firm')
        ax.set_ylabel(f'# {stage} opened')
        ax.set_title(title)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(i) for i in range(nf)], fontsize=7)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.legend(fontsize=7)

        # Mark firms with zero opens
        for i in range(nf):
            if bottoms[i] == 0:
                ax.annotate('∅', (i, 0.1), ha='center', fontsize=10, color='red', fontweight='bold')

    fig.suptitle(f'Facility usage by firm and region (N={nf}, seed={seed})', fontsize=12)
    plt.tight_layout()
    out = PLOT_DIR / f'facility_usage_seed{seed}.png'
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


def plot_networks(geo, firms, bundles, seed, nf):
    """4-panel firm network plots."""
    candidates = [(i, bundles[i]['y2'].sum(), bundles[i]['obj'])
                  for i in range(nf) if bundles[i]['obj'] > 0]
    candidates.sort(key=lambda x: (-min(x[1], 3), -x[2]))
    selected = [c[0] for c in candidates[:4]]

    cell_locs, asm_locs, mkt_locs = geo['cell_locs'], geo['asm_locs'], geo['mkt_locs']
    L1, L2 = geo['L1'], geo['L2']
    model_colors = ['tab:green', 'tab:orange', 'tab:purple', 'tab:red',
                    'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan']

    cell_xs = np.sort(cell_locs[:, 0])
    cell_b1 = (cell_xs[L1 // 3 - 1] + cell_xs[L1 // 3]) / 2
    cell_b2 = (cell_xs[2 * L1 // 3 - 1] + cell_xs[2 * L1 // 3]) / 2

    fig, axes = plt.subplots(2, 2, figsize=(13, 13))
    for panel, fidx in enumerate(selected):
        ax = axes.flat[panel]
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')

        for r, (lo, hi) in enumerate(zip([0, cell_b1, cell_b2], [cell_b1, cell_b2, 1.0])):
            ax.axvspan(lo, hi, alpha=0.15,
                       color=['#d4e6f1', '#fdebd0', '#d5f5e3'][r], zorder=0)
        for bx in [cell_b1, cell_b2]:
            ax.axvline(bx, color='gray', linewidth=0.8, linestyle='--', alpha=0.5, zorder=0)

        ax.scatter(mkt_locs[:, 0], mkt_locs[:, 1], c='gray', s=20, zorder=1, alpha=0.5)
        ax.scatter(cell_locs[:, 0], cell_locs[:, 1], c='blue', s=60, marker='s', zorder=3)
        ax.scatter(asm_locs[:, 0], asm_locs[:, 1], c='red', s=60, marker='o', zorder=3)
        for l in range(L1):
            ax.annotate(f"c{l}", cell_locs[l], fontsize=6, ha='center', va='bottom',
                        xytext=(0, 4), textcoords='offset points', color='blue')
        for l in range(L2):
            ax.annotate(f"a{l}", asm_locs[l], fontsize=6, ha='center', va='bottom',
                        xytext=(0, 4), textcoords='offset points', color='red')

        firm, bun = firms[fidx], bundles[fidx]
        nm, P = firm['n_models'], firm['n_platforms']
        paths = find_model_paths(firm, bun, geo)

        hq = firm['hq_coord']
        ax.scatter(hq[0], hq[1], marker='*', s=200, c='black', zorder=5, label='HQ')

        for m, p in enumerate(paths):
            if p is None:
                continue
            color = model_colors[m % len(model_colors)]
            cl, al = cell_locs[p['cell']], asm_locs[p['asm']]
            ax.plot([cl[0], al[0]], [cl[1], al[1]], color=color, linewidth=2.5, zorder=2,
                    label=f"m{m}(p{p['platform']})")
            for n in p['markets']:
                ax.plot([al[0], mkt_locs[n, 0]], [al[1], mkt_locs[n, 1]],
                        color=color, linewidth=0.7, alpha=0.4, zorder=1)

        ax.set_title(f"Firm {fidx}: {nm} models, {P} platforms", fontsize=11)
        ax.legend(fontsize=6, loc='lower right', ncol=2)

        print(f"Panel {panel}: Firm {fidx} ({nm}m/{P}p)")

    fig.suptitle(f"Firm facility networks (seed={seed}, N={nf}, torus geometry)\n"
                 "Stars=HQ, lines=Euclidean (actual=torus)", fontsize=11)
    plt.tight_layout()
    out = PLOT_DIR / f'networks_seed{seed}.png'
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


def plot_single_firm(geo, firms, bundles, seed, fidx=None):
    """Single-firm network, no captions/titles/legend. Picks a firm with
    decent network activity if fidx not given."""
    if fidx is None:
        candidates = [(i, bundles[i]['y2'].sum(), bundles[i]['obj'])
                      for i in range(len(firms)) if bundles[i]['obj'] > 0]
        candidates.sort(key=lambda x: (-min(x[1], 3), -x[2]))
        fidx = candidates[0][0]

    cell_locs, asm_locs, mkt_locs = geo['cell_locs'], geo['asm_locs'], geo['mkt_locs']
    L1, L2 = geo['L1'], geo['L2']
    model_colors = ['tab:green', 'tab:orange', 'tab:purple', 'tab:red',
                    'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan']

    cell_xs = np.sort(cell_locs[:, 0])
    cell_b1 = (cell_xs[L1 // 3 - 1] + cell_xs[L1 // 3]) / 2
    cell_b2 = (cell_xs[2 * L1 // 3 - 1] + cell_xs[2 * L1 // 3]) / 2

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    for r, (lo, hi) in enumerate(zip([0, cell_b1, cell_b2], [cell_b1, cell_b2, 1.0])):
        ax.axvspan(lo, hi, alpha=0.15,
                   color=['#d4e6f1', '#fdebd0', '#d5f5e3'][r], zorder=0)
    for bx in [cell_b1, cell_b2]:
        ax.axvline(bx, color='gray', linewidth=0.8, linestyle='--', alpha=0.5, zorder=0)

    ax.scatter(mkt_locs[:, 0], mkt_locs[:, 1], c='gray', s=25, zorder=1, alpha=0.5,
               label='Markets')
    ax.scatter(cell_locs[:, 0], cell_locs[:, 1], c='blue', s=70, marker='s', zorder=3,
               label='Cell plants')
    ax.scatter(asm_locs[:, 0], asm_locs[:, 1], c='red', s=70, marker='o', zorder=3,
               label='Assembly plants')

    firm, bun = firms[fidx], bundles[fidx]
    paths = find_model_paths(firm, bun, geo)

    hq = firm['hq_coord']
    ax.scatter(hq[0], hq[1], marker='*', s=260, c='black', zorder=5, label='Firm HQ')

    for m, p in enumerate(paths):
        if p is None:
            continue
        color = model_colors[m % len(model_colors)]
        cl, al = cell_locs[p['cell']], asm_locs[p['asm']]
        ax.plot([cl[0], al[0]], [cl[1], al[1]], color=color, linewidth=2.8, zorder=2)
        for n in p['markets']:
            ax.plot([al[0], mkt_locs[n, 0]], [al[1], mkt_locs[n, 1]],
                    color=color, linewidth=0.8, alpha=0.45, zorder=1)

    ax.legend(loc='upper right', fontsize=10, framealpha=0.85,
              markerscale=0.9, handletextpad=0.5)
    plt.tight_layout(pad=0.1)
    out = PLOT_DIR / f'network_single_seed{seed}_firm{fidx}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved {out}")
    return fidx


PARAM_NAMES = [
    'delta_1', 'delta_2', 'rho_xi_1', 'rho_xi_2',
    'rho_HQ_1', 'rho_HQ_2', 'FE_1_r1', 'FE_1_r2',
    'FE_2_r1', 'FE_2_r2', 'rho_d_1', 'rho_d_2',
]

PARAM_BOUNDS = {
    'lb': [0]*6 + [-5]*4 + [0]*2,
    'ub': [10]*6 + [5]*4 + [10]*2,
}


def _region_counts(bundles, y_key, region_arr, nf):
    """(nf, 3) count of opens per firm per region."""
    counts = np.zeros((nf, 3))
    for i in range(nf):
        y = bundles[i][y_key]
        for g in range(y.shape[0]):
            for l in range(y.shape[1]):
                if y[g, l]:
                    counts[i, region_arr[l]] += 1
    return counts


def _build_covariates(active, firms, bundles, geo):
    """Build (n_active, 12) covariate matrix from observed bundles."""
    L1, L2, N = geo['L1'], geo['L2'], geo['n_markets']
    cr, ar = geo['cell_region'], geo['asm_region']
    d12, d2m, R_n = geo['d_12'], geo['d_2m'], geo['R_n']
    n = len(active)
    Phi = np.zeros((n, 12))

    for idx, i in enumerate(active):
        firm = firms[i]
        bun = bundles[i]
        y1, y2, x_val = bun['y1'], bun['y2'], bun['x']
        ng, P, nm = y1.shape[0], y2.shape[0], firm['n_models']

        for g in range(ng):
            for l in range(L1):
                if y1[g, l]:
                    Phi[idx, 0] -= 1.0                          # delta_1
                    Phi[idx, 2] -= firm['ln_xi_1'][g]            # rho_xi_1
                    Phi[idx, 4] -= firm['d_hq1'][l]              # rho_HQ_1
                    if cr[l] == 1: Phi[idx, 6] += 1.0           # FE_1_r1
                    if cr[l] == 2: Phi[idx, 7] += 1.0           # FE_1_r2

        for p in range(P):
            for l in range(L2):
                if y2[p, l]:
                    Phi[idx, 1] -= 1.0                          # delta_2
                    Phi[idx, 3] -= firm['ln_xi_2'][p]            # rho_xi_2
                    Phi[idx, 5] -= firm['d_hq2'][l]              # rho_HQ_2
                    if ar[l] == 1: Phi[idx, 8] += 1.0           # FE_2_r1
                    if ar[l] == 2: Phi[idx, 9] += 1.0           # FE_2_r2

        for m in range(nm):
            sR = firm['shares'][m, :] * R_n
            Phi[idx, 10] -= (sR[:, None, None] * d12[None, :, :] * x_val[m]).sum()
            Phi[idx, 11] -= (sR[:, None, None] * d2m.T[:, None, :] * x_val[m]).sum()

    return Phi


def _compute_vif(Phi):
    """Variance inflation factors: VIF_j = 1 / (1 - R^2_j)."""
    K = Phi.shape[1]
    vif = np.full(K, np.inf)
    for j in range(K):
        y = Phi[:, j]
        X = np.delete(Phi, j, axis=1)
        if y.std() == 0:
            continue
        X = np.column_stack([np.ones(len(y)), X])
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        ss_res = (resid ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
        vif[j] = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf
    return vif


def diagnose_identification(geo, firms, bundles, theta_true):
    """Comprehensive identification diagnostics for the 12-parameter model."""
    nf = len(firms)
    L1, L2, N = geo['L1'], geo['L2'], geo['n_markets']
    cr, ar = geo['cell_region'], geo['asm_region']
    R_n = geo['R_n']
    d12, d2m = geo['d_12'], geo['d_2m']

    sep = '=' * 65
    print(f"\n{sep}")
    print('  DGP IDENTIFICATION DIAGNOSTICS')
    print(sep)

    # ---- [1] Active firms ----
    active_mask = np.array([b['obj'] > 0 for b in bundles])
    active = np.where(active_mask)[0]
    n_act = len(active)
    print(f"\n[1] Active firms: {n_act}/{nf}  (opt-out: {nf - n_act})")
    if n_act < 12:
        print('    !! CRITICAL: fewer active firms than parameters (12)')
    elif n_act < 20:
        print(f'    ! Warning: thin sample ({n_act} obs for 12 params)')

    # ---- [2] Cell region distribution ----
    cell_counts = _region_counts(bundles, 'y1', cr, nf)
    print(f'\n[2] Cell opens by region')
    for r in range(3):
        tot = int(cell_counts[active, r].sum())
        nf_r = int((cell_counts[active, r] > 0).sum())
        print(f'    r{r}: {tot:3d} opens  ({nf_r} firms)')
    total_cells = cell_counts[active].sum()
    if total_cells > 0:
        print(f'    Baseline (r0) share: {cell_counts[active, 0].sum() / total_cells:.1%}')
    if cell_counts[active, 0].sum() == 0:
        print('    !! CRITICAL: zero cells in r0 -> delta_1 collinear with FE_1_r1 + FE_1_r2')
    for r in [1, 2]:
        if cell_counts[active, r].sum() == 0:
            print(f'    !! CRITICAL: zero cells in r{r} -> FE_1_r{r} unidentified')

    # ---- [3] Assembly region distribution ----
    asm_counts = _region_counts(bundles, 'y2', ar, nf)
    print(f'\n[3] Assembly opens by region')
    for r in range(3):
        tot = int(asm_counts[active, r].sum())
        nf_r = int((asm_counts[active, r] > 0).sum())
        print(f'    r{r}: {tot:3d} opens  ({nf_r} firms)')
    total_asms = asm_counts[active].sum()
    if total_asms > 0:
        print(f'    Baseline (r0) share: {asm_counts[active, 0].sum() / total_asms:.1%}')
    if asm_counts[active, 0].sum() == 0:
        print('    !! CRITICAL: zero assemblies in r0 -> delta_2 collinear with FE_2_r1 + FE_2_r2')
    for r in [1, 2]:
        if asm_counts[active, r].sum() == 0:
            print(f'    !! CRITICAL: zero assemblies in r{r} -> FE_2_r{r} unidentified')

    # ---- [4] delta vs FE-sum collinearity ----
    print(f'\n[4] delta vs FE-sum collinearity (r0 opens per firm)')
    for label, counts in [('Cells (delta_1)', cell_counts), ('Asm (delta_2)', asm_counts)]:
        r0 = counts[active, 0]
        total = counts[active].sum(axis=1)
        frac_r0 = r0 / np.maximum(total, 1)
        print(f'    {label}:')
        print(f'      r0 opens: min={r0.min():.0f}  mean={r0.mean():.1f}  max={r0.max():.0f}  '
              f'std={r0.std():.2f}')
        print(f'      r0 fraction per firm: mean={frac_r0.mean():.2f}  std={frac_r0.std():.2f}')
        n_zero_r0 = (r0 == 0).sum()
        if n_zero_r0 == n_act:
            print(f'      !! CRITICAL: ALL firms have zero r0 opens -> exact collinearity')
        elif n_zero_r0 > n_act * 0.7:
            print(f'      ! Warning: {n_zero_r0}/{n_act} firms have zero r0 opens')

    # ---- [5] Within-firm region diversification ----
    print(f'\n[5] Within-firm region diversification')
    for label, counts in [('cell', cell_counts), ('assembly', asm_counts)]:
        n_regions = np.array([(counts[i] > 0).sum() for i in active])
        dist = {k: int((n_regions == k).sum()) for k in range(4) if (n_regions == k).sum()}
        print(f'    {label}: regions per firm = {dist}')
        if (n_regions <= 1).all():
            print(f'    ! Warning: no firm opens {label}s in >1 region')

    # ---- [6] HQ-region x facility-region cross-tab ----
    print(f'\n[6] HQ-region x facility-region cross-tab')
    for label, counts in [('cell', cell_counts), ('assembly', asm_counts)]:
        print(f'    {label} opens:  {"HQ_r0":>6}  {"HQ_r1":>6}  {"HQ_r2":>6}')
        for fac_r in range(3):
            row = []
            for hq_r in range(3):
                hq_firms = [i for i in active if i % 3 == hq_r]
                row.append(int(counts[hq_firms, fac_r].sum()))
            print(f'      fac_r{fac_r}:  {row[0]:6d}  {row[1]:6d}  {row[2]:6d}')

    # ---- [7] Covariate matrix ----
    print(f'\n[7] Covariate matrix Phi ({n_act} x 12)')
    Phi = _build_covariates(active, firms, bundles, geo)

    rank = np.linalg.matrix_rank(Phi)
    print(f'    Rank: {rank}')
    if rank < 12:
        print(f'    !! CRITICAL: rank deficient ({rank} < 12)')

    sv = np.linalg.svd(Phi, compute_uv=False)
    cond = sv[0] / sv[-1] if sv[-1] > 0 else float('inf')
    print(f'    Condition number: {cond:.1f}')
    print(f'    Singular values:')
    for j in range(len(sv)):
        flag = ' <- near-zero' if sv[j] < 1e-6 * sv[0] else ''
        print(f'      sv_{j+1:2d} = {sv[j]:.4e}{flag}')

    # ---- [8] Per-column variation ----
    print(f'\n[8] Per-parameter covariate variation')
    print(f'    {"Param":<12} {"mean":>9} {"std":>9} {"min":>9} {"max":>9}')
    print(f'    {"-"*50}')
    for j in range(12):
        col = Phi[:, j]
        print(f'    {PARAM_NAMES[j]:<12} {col.mean():>9.4f} {col.std():>9.4f} '
              f'{col.min():>9.4f} {col.max():>9.4f}')
        if col.std() < 1e-8:
            print(f'    !! CRITICAL: {PARAM_NAMES[j]} has zero variation')

    # ---- [9] Critical pairwise correlations ----
    print(f'\n[9] Critical pairwise correlations')
    pairs = [
        (0, 6, 'delta_1   vs FE_1_r1'),
        (0, 7, 'delta_1   vs FE_1_r2'),
        (1, 8, 'delta_2   vs FE_2_r1'),
        (1, 9, 'delta_2   vs FE_2_r2'),
        (0, 2, 'delta_1   vs rho_xi_1'),
        (1, 3, 'delta_2   vs rho_xi_2'),
        (0, 4, 'delta_1   vs rho_HQ_1'),
        (1, 5, 'delta_2   vs rho_HQ_2'),
        (4, 6, 'rho_HQ_1  vs FE_1_r1'),
        (4, 7, 'rho_HQ_1  vs FE_1_r2'),
        (5, 8, 'rho_HQ_2  vs FE_2_r1'),
        (5, 9, 'rho_HQ_2  vs FE_2_r2'),
        (10, 11, 'rho_d_1   vs rho_d_2'),
        (6, 7, 'FE_1_r1   vs FE_1_r2'),
        (8, 9, 'FE_2_r1   vs FE_2_r2'),
    ]
    for a, b, label in pairs:
        sa, sb = Phi[:, a].std(), Phi[:, b].std()
        if sa > 0 and sb > 0:
            r = np.corrcoef(Phi[:, a], Phi[:, b])[0, 1]
            flag = ' <- HIGH' if abs(r) > 0.9 else (' <- moderate' if abs(r) > 0.7 else '')
            print(f'    {label}  r = {r:+.3f}{flag}')
        else:
            print(f'    {label}  (degenerate)')

    # ---- [10] VIF ----
    print(f'\n[10] Variance inflation factors')
    vif = _compute_vif(Phi)
    for j in range(12):
        flag = ' <- HIGH' if vif[j] > 10 else ''
        v_str = f'{vif[j]:.1f}' if np.isfinite(vif[j]) else 'inf'
        print(f'    {PARAM_NAMES[j]:<12} VIF = {v_str:>8}{flag}')

    # ---- [11] Distance covariate diagnostics ----
    print(f'\n[11] Distance covariate diagnostics')
    for j, name in [(10, 'rho_d_1'), (11, 'rho_d_2')]:
        col = Phi[:, j]
        nz = col[col != 0]
        if len(nz) > 1:
            cv = nz.std() / abs(nz.mean()) if abs(nz.mean()) > 0 else float('inf')
            print(f'    {name}: CV = {cv:.3f}  (n_nonzero = {len(nz)})')
        else:
            print(f'    {name}: insufficient variation')
    if Phi[:, 10].std() > 0 and Phi[:, 11].std() > 0:
        r = np.corrcoef(Phi[:, 10], Phi[:, 11])[0, 1]
        print(f'    rho_d_1 vs rho_d_2 correlation: {r:+.3f}')

    # ---- [12] Quality shock diagnostics ----
    print(f'\n[12] Quality shock (xi) diagnostics')
    xi1 = np.concatenate([firms[i]['ln_xi_1'] for i in active])
    xi2 = np.concatenate([firms[i]['ln_xi_2'] for i in active])
    print(f'    xi_1 (cells):      n={len(xi1):3d}  std={xi1.std():.3f}  '
          f'range=[{xi1.min():.3f}, {xi1.max():.3f}]')
    print(f'    xi_2 (assemblies): n={len(xi2):3d}  std={xi2.std():.3f}  '
          f'range=[{xi2.min():.3f}, {xi2.max():.3f}]')

    # ---- [13] Bundle diversity ----
    print(f'\n[13] Bundle diversity')
    n_cells = cell_counts[active].sum(axis=1)
    n_asms = asm_counts[active].sum(axis=1)
    n_entries = np.array([bundles[i]['z'].sum() for i in active])
    print(f'    Cells per firm:     min={n_cells.min():.0f}  mean={n_cells.mean():.1f}  '
          f'max={n_cells.max():.0f}')
    print(f'    Asm per firm:       min={n_asms.min():.0f}  mean={n_asms.mean():.1f}  '
          f'max={n_asms.max():.0f}')
    print(f'    Market entries:     min={n_entries.min():.0f}  mean={n_entries.mean():.1f}  '
          f'max={n_entries.max():.0f}')

    # ---- [14] theta_true vs bounds ----
    print(f'\n[14] theta_true vs estimation bounds')
    lb = PARAM_BOUNDS['lb']
    ub = PARAM_BOUNDS['ub']
    for j, name in enumerate(PARAM_NAMES):
        val = theta_true[name]
        span = ub[j] - lb[j]
        near_lb = (val - lb[j]) / span < 0.10
        near_ub = (ub[j] - val) / span < 0.10
        if near_lb:
            print(f'    ! Warning: {name} = {val:.3f} near lower bound {lb[j]}')
        if near_ub:
            print(f'    ! Warning: {name} = {val:.3f} near upper bound {ub[j]}')
    else:
        # only print if no warnings fired
        pass
    if not any(
        (theta_true[name] - lb[j]) / (ub[j] - lb[j]) < 0.10 or
        (ub[j] - theta_true[name]) / (ub[j] - lb[j]) < 0.10
        for j, name in enumerate(PARAM_NAMES)
    ):
        print('    All parameters comfortably in interior')

    # ---- [15] Firm structure heterogeneity ----
    print(f'\n[15] Firm structure heterogeneity')
    nms = np.array([firms[i]['n_models'] for i in active])
    Ps = np.array([firms[i]['n_platforms'] for i in active])
    ngs = np.array([len(firms[i]['ln_xi_1']) for i in active])
    print(f'    Models per firm:    min={nms.min()}  max={nms.max()}  unique={len(np.unique(nms))}')
    print(f'    Platforms per firm: min={Ps.min()}  max={Ps.max()}  unique={len(np.unique(Ps))}')
    print(f'    Cell groups/firm:   min={ngs.min()}  max={ngs.max()}  unique={len(np.unique(ngs))}')

    print(f'\n{sep}\n')
    return Phi


def main():
    dgp = CFG['dgp']
    theta_true = CFG['theta_true']
    sigma = CFG['sigma']
    seed = CFG.get('monte_carlo', {}).get('seed', 42)
    nf = dgp['n_firms']

    geo, firms, _, _ = generate_synthetic_data(
        seed=seed, dgp=dgp, sourcing_coefs=None,
        theta_true=theta_true, sigma=sigma)
    print(f"Solving MILPs (seed={seed}, N={nf})...")
    bundles = solve_dgp(geo, firms, theta_true, sigma, seed)

    n_empty = sum(1 for b in bundles if b['obj'] <= 0)
    if n_empty:
        print(f"WARNING: {n_empty} firms chose empty bundles (obj <= 0)")

    diagnose_identification(geo, firms, bundles, theta_true)
    plot_facility_usage(geo, firms, bundles, seed, nf)
    plot_networks(geo, firms, bundles, seed, nf)
    plot_single_firm(geo, firms, bundles, seed)


if __name__ == '__main__':
    main()
