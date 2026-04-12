"""Plot firm facility networks and assembly usage from config. Fully independent of estimation."""

import sys
from pathlib import Path
import yaml
import numpy as np
import gurobipy as gp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_data import generate_synthetic_data, compute_pi, compute_facility_costs, draw_firm_errors
from milp import build_milp_shell

BASE = Path(__file__).resolve().parent
with open(BASE / 'config.yaml') as f:
    CFG = yaml.safe_load(f)


def solve_dgp(geo, firms, theta_true, sigma, seed):
    """Solve each firm's MILP at theta_true. Independent of combest."""
    rng = np.random.default_rng(seed + 999)  # offset to not collide with DGP rng
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()

    bundles = []
    for firm in firms:
        pi = compute_pi(firm, geo, theta_true)
        fc1, fc2 = compute_facility_costs(firm, geo, theta_true)
        errors = draw_firm_errors(firm, geo, sigma, rng)
        nm, P = firm['n_models'], firm['n_platforms']
        N, L1, L2 = geo['n_markets'], geo['L1'], geo['L2']
        ng = len(firm['ln_xi_1'])

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
    nm = firm['n_models']
    x = bun['x']
    paths = []
    for m in range(nm):
        active = x[m] > 0.5
        if not active.any():
            paths.append(None)
            continue
        x_sum_over_n = x[m].sum(axis=0)
        l1, l2 = np.unravel_index(x_sum_over_n.argmax(), x_sum_over_n.shape)
        markets = np.where(active[:, l1, l2])[0]
        paths.append(dict(cell=l1, asm=l2, markets=markets, platform=firm['platforms'][m]))
    return paths


def plot_firm(ax, firm, bun, geo, color_set):
    nm = firm['n_models']
    paths = find_model_paths(firm, bun, geo)
    hq = firm['hq_coord']
    ax.scatter(hq[0], hq[1], marker='*', s=200, c='black', zorder=5, label='HQ')

    cell_locs, asm_locs, mkt_locs = geo['cell_locs'], geo['asm_locs'], geo['mkt_locs']
    info = []
    for m, p in enumerate(paths):
        if p is None:
            info.append(f"  m{m}: inactive")
            continue
        color = color_set[m % len(color_set)]
        cl, al = cell_locs[p['cell']], asm_locs[p['asm']]
        ax.plot([cl[0], al[0]], [cl[1], al[1]], color=color, linewidth=2.5, zorder=2,
                label=f"m{m}(p{p['platform']})")
        for n in p['markets']:
            ax.plot([al[0], mkt_locs[n, 0]], [al[1], mkt_locs[n, 1]],
                    color=color, linewidth=0.7, alpha=0.4, zorder=1)
        info.append(f"  m{m}(p{p['platform']}): c{p['cell']}->a{p['asm']}->"
                    f"mkts{p['markets'].tolist()}")
    return info


def main():
    dgp = CFG['dgp']
    theta_true = CFG['theta_true']
    sigma = CFG['sigma']
    seed = CFG.get('monte_carlo', {}).get('seed', 42)
    nf = dgp['n_firms']

    # Generate and solve — fully self-contained
    geo, firms, _, _ = generate_synthetic_data(
        seed=seed, dgp=dgp, sourcing_coefs=None,
        theta_true=theta_true, sigma=sigma)
    print(f"Generating DGP and solving MILPs (seed={seed}, N={nf})...")
    bundles = solve_dgp(geo, firms, theta_true, sigma, seed)

    # Select 4 firms by assembly diversity and objective
    candidates = [(i, bundles[i]['y2'].sum(), bundles[i]['obj'])
                  for i in range(len(firms)) if bundles[i]['obj'] > 0]
    candidates.sort(key=lambda x: (-min(x[1], 3), -x[2]))
    selected = [c[0] for c in candidates[:4]]

    cell_locs, asm_locs, mkt_locs = geo['cell_locs'], geo['asm_locs'], geo['mkt_locs']
    L1, L2 = geo['L1'], geo['L2']
    colors = ['tab:green', 'tab:orange', 'tab:purple', 'tab:red',
              'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan']

    # Region boundaries
    cell_xs = np.sort(cell_locs[:, 0])
    cell_b1 = (cell_xs[L1 // 3 - 1] + cell_xs[L1 // 3]) / 2
    cell_b2 = (cell_xs[2 * L1 // 3 - 1] + cell_xs[2 * L1 // 3]) / 2

    fig, axes = plt.subplots(2, 2, figsize=(13, 13))
    for panel, fidx in enumerate(selected):
        ax = axes.flat[panel]
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')

        # Region shading
        for r, (lo, hi) in enumerate(zip([0, cell_b1, cell_b2], [cell_b1, cell_b2, 1.0])):
            ax.axvspan(lo, hi, alpha=0.15,
                       color=['#d4e6f1', '#fdebd0', '#d5f5e3'][r], zorder=0)
        for bx in [cell_b1, cell_b2]:
            ax.axvline(bx, color='gray', linewidth=0.8, linestyle='--', alpha=0.5, zorder=0)

        # Locations
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
        info = plot_firm(ax, firm, bun, geo, colors)
        ax.set_title(f"Firm {fidx}: {nm} models, {P} platforms", fontsize=11)
        ax.legend(fontsize=6, loc='lower right', ncol=2)

        print(f"Panel {panel}: Firm {fidx} ({nm}m/{P}p)")
        for line in info:
            print(line)

    fig.suptitle(f"Firm facility networks (seed={seed}, N={nf}, torus geometry)\n"
                 "Stars=HQ, lines=Euclidean (actual=torus)", fontsize=11)
    plt.tight_layout()
    fig.savefig(BASE / f'firms_seed{seed}.png', dpi=150)
    print(f"Saved to {BASE / f'firms_seed{seed}.png'}")

    # Assembly usage histogram
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    asm_counts = np.zeros(L2, dtype=int)
    for bun in bundles:
        for p in range(bun['y2'].shape[0]):
            asm_counts += bun['y2'][p].astype(int)
    region_colors = {0: 'tab:blue', 1: 'tab:orange', 2: 'tab:green'}
    ax2.bar(range(L2), asm_counts,
            color=[region_colors[geo['asm_region'][l]] for l in range(L2)])
    ax2.set_xlabel('Assembly location')
    ax2.set_ylabel('Total opens (across all firms/platforms)')
    ax2.set_title(f'Assembly usage histogram (N={nf} firms, seed={seed})')
    ax2.set_xticks(range(L2))
    ax2.set_xticklabels([f"a{l}\n(r{geo['asm_region'][l]})" for l in range(L2)])
    plt.tight_layout()
    fig2.savefig(BASE / f'asm_usage_seed{seed}.png', dpi=150)
    print(f"Saved to {BASE / f'asm_usage_seed{seed}.png'}")


if __name__ == '__main__':
    main()
