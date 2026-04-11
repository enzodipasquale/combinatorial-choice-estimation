"""HMMY Section 5-style firm facility location plots."""

import sys
from pathlib import Path
import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_data import generate_synthetic_data

BASE = Path(__file__).resolve().parent
with open(BASE / 'config.yaml') as f:
    CFG = yaml.safe_load(f)


def find_model_paths(firm, bun, geo):
    """For each model, find which (cell, asm) it uses and which markets it serves."""
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


def plot_firm(ax, firm, bun, geo, color_set, linestyle='-', alpha=1.0, label_prefix=''):
    """Plot one firm's network on an axis."""
    nm = firm['n_models']
    paths = find_model_paths(firm, bun, geo)

    # HQ
    hq = firm['hq_coord']
    ax.scatter(hq[0], hq[1], marker='*', s=150, c='black', zorder=5,
               alpha=alpha)
    # Market home
    mh = firm['market_home']
    ax.scatter(mh[0], mh[1], marker='X', s=100, c=color_set[0], zorder=5,
               alpha=alpha, edgecolors='black', linewidths=0.5)

    cell_locs = geo['cell_locs']
    asm_locs = geo['asm_locs']
    mkt_locs = geo['mkt_locs']

    info_lines = []
    for m, p in enumerate(paths):
        if p is None:
            info_lines.append(f"  m{m}: inactive")
            continue
        color = color_set[m % len(color_set)]
        cl = cell_locs[p['cell']]
        al = asm_locs[p['asm']]

        ax.plot([cl[0], al[0]], [cl[1], al[1]], color=color, linewidth=2.5,
                linestyle=linestyle, zorder=2, alpha=alpha,
                label=f"{label_prefix}m{m}(p{p['platform']})")

        for n in p['markets']:
            ml = mkt_locs[n]
            ax.plot([al[0], ml[0]], [al[1], ml[1]], color=color,
                    linewidth=0.7, alpha=0.4 * alpha, linestyle=linestyle, zorder=1)

        info_lines.append(f"  m{m}(p{p['platform']}): c{p['cell']}->a{p['asm']}->"
                          f"mkts{p['markets'].tolist()}")
    return info_lines


def main():
    dgp = dict(L1=6, L2=6, N=12, n_firms=15)
    theta_true = CFG['theta_true']
    sigma = CFG['sigma']

    geo, firms, bundles, _ = generate_synthetic_data(
        seed=42, dgp=dgp, sourcing_coefs=None,
        theta_true=theta_true, sigma=sigma)

    # Select 8 interesting firms (by objective, diverse assemblies)
    candidates = [(i, bundles[i]['y2'].sum(), bundles[i]['obj'])
                  for i in range(len(firms)) if bundles[i]['obj'] > 0]
    candidates.sort(key=lambda x: (-min(x[1], 3), -x[2]))
    selected = [c[0] for c in candidates[:8]]

    # Pair them up: 2 per panel
    pairs = [(selected[i], selected[i + 1]) for i in range(0, 8, 2)]

    cell_locs = geo['cell_locs']
    asm_locs = geo['asm_locs']
    mkt_locs = geo['mkt_locs']

    fig, axes = plt.subplots(2, 2, figsize=(13, 13))
    axes_flat = axes.ravel()

    color_sets = [
        ['tab:green', 'tab:olive', 'darkgreen'],
        ['tab:red', 'tab:orange', 'darkred'],
    ]

    for panel, (f1, f2) in enumerate(pairs):
        ax = axes_flat[panel]
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')

        # Background: locations
        ax.scatter(mkt_locs[:, 0], mkt_locs[:, 1], c='gray', s=20, zorder=1, alpha=0.5)
        ax.scatter(cell_locs[:, 0], cell_locs[:, 1], c='blue', s=80, marker='s', zorder=3)
        ax.scatter(asm_locs[:, 0], asm_locs[:, 1], c='red', s=80, marker='o', zorder=3)

        # Labels
        for l in range(geo['L1']):
            ax.annotate(f"c{l}", cell_locs[l], fontsize=7, ha='center', va='bottom',
                        xytext=(0, 5), textcoords='offset points', color='blue')
        for l in range(geo['L2']):
            ax.annotate(f"a{l}", asm_locs[l], fontsize=7, ha='center', va='bottom',
                        xytext=(0, 5), textcoords='offset points', color='red')

        # Firm 1 (solid)
        firm1, bun1 = firms[f1], bundles[f1]
        info1 = plot_firm(ax, firm1, bun1, geo, color_sets[0], '-', 1.0, f'F{f1}:')

        # Firm 2 (dashed)
        firm2, bun2 = firms[f2], bundles[f2]
        info2 = plot_firm(ax, firm2, bun2, geo, color_sets[1], '--', 0.8, f'F{f2}:')

        nm1, P1 = firm1['n_models'], firm1['n_platforms']
        nm2, P2 = firm2['n_models'], firm2['n_platforms']
        ax.set_title(f"Firm {f1} ({nm1}m/{P1}p, solid) & "
                     f"Firm {f2} ({nm2}m/{P2}p, dashed)", fontsize=10)
        ax.legend(fontsize=6, loc='lower right', ncol=2)

        print(f"Panel {panel}: Firm {f1} ({nm1}m/{P1}p)")
        for line in info1:
            print(line)
        print(f"  Firm {f2} ({nm2}m/{P2}p)")
        for line in info2:
            print(line)
        print()

    fig.suptitle("Firm facility networks (seed=42, N=15, torus geometry)\n"
                 "Stars=HQ, X=market home, lines=Euclidean (actual=torus)",
                 fontsize=11)
    plt.tight_layout()

    out_path = BASE / 'firms_seed42.png'
    fig.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")

    # Assembly usage histogram
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    asm_counts = np.zeros(geo['L2'], dtype=int)
    for bun in bundles:
        for p in range(bun['y2'].shape[0]):
            asm_counts += bun['y2'][p].astype(int)

    colors = ['tab:blue' if geo['asm_region'][l] == 0 else 'tab:orange'
              for l in range(geo['L2'])]
    bars = ax2.bar(range(geo['L2']), asm_counts, color=colors)
    ax2.set_xlabel('Assembly location')
    ax2.set_ylabel('Total opens (across all firms/platforms)')
    ax2.set_title(f'Assembly usage histogram (N=15 firms, seed=42)')
    ax2.set_xticks(range(geo['L2']))
    ax2.set_xticklabels([f"a{l}\n(r{geo['asm_region'][l]})" for l in range(geo['L2'])])
    plt.tight_layout()

    out_path2 = BASE / 'asm_usage_seed42.png'
    fig2.savefig(out_path2, dpi=150)
    print(f"Saved to {out_path2}")


if __name__ == '__main__':
    main()
