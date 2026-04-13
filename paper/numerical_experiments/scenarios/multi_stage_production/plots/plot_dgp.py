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

    plot_facility_usage(geo, firms, bundles, seed, nf)
    plot_networks(geo, firms, bundles, seed, nf)


if __name__ == '__main__':
    main()
