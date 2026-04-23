"""Plot airline network choices: cities as nodes, selected routes as edges."""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_data import generate_data


def plot_airline_network(ax, locations, populations, hubs_i, bundle,
                         endpoints_a, endpoints_b, C, airline_idx,
                         edges_color='C0'):
    """Plot one airline's network on an axes (undirected edges)."""
    # City sizes proportional to population
    pop_sizes = 30 + 200 * (populations / populations.max())

    # Draw all cities as grey dots
    ax.scatter(locations[:, 0], locations[:, 1], s=pop_sizes,
               c='lightgrey', edgecolors='grey', zorder=3, linewidths=0.5)

    # Draw selected routes as undirected lines
    active = np.where(bundle)[0]
    for j in active:
        a, b = endpoints_a[j], endpoints_b[j]
        ax.plot([locations[a, 0], locations[b, 0]],
                [locations[a, 1], locations[b, 1]],
                color=edges_color, alpha=0.35, lw=0.7)

    # Highlight hub cities
    hub_list = sorted(hubs_i)
    for h in hub_list:
        ax.scatter(locations[h, 0], locations[h, 1], s=pop_sizes[h] * 1.8,
                   marker='*', c='red', edgecolors='darkred', zorder=5,
                   linewidths=0.5)

    # City labels
    for c in range(C):
        ax.annotate(str(c), locations[c], fontsize=5, ha='center', va='bottom',
                    xytext=(0, 4), textcoords='offset points', color='dimgrey')

    n_routes = bundle.sum()
    n_hubs = len(hub_list)
    ax.set_title(f'Airline {airline_idx}: {n_routes} routes, {n_hubs} hubs',
                 fontsize=9)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=6)


def main():
    import yaml
    with open(Path(__file__).resolve().parent / 'config.yaml') as f:
        cfg = yaml.safe_load(f)

    theta_star, obs_bundles, dgp_data, diag = generate_data(
        cfg['dgp'], cfg['healthy_dgp'], cfg['seeds'], verbose=False)
    C = cfg['dgp']['C']
    N = cfg['dgp']['N']

    locations = dgp_data['locations']
    populations = dgp_data['populations']
    hubs = dgp_data['hubs']
    endpoints_a = dgp_data['endpoints_a']
    endpoints_b = dgp_data['endpoints_b']
    M = dgp_data['M']

    # Pick 4 airlines with varied bundle sizes for visual interest
    sizes = obs_bundles.sum(axis=1)
    sorted_idx = np.argsort(sizes)
    picks = [sorted_idx[0], sorted_idx[N // 3], sorted_idx[2 * N // 3], sorted_idx[-1]]

    colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        f'Airline route networks (C={C}, M={M}, N={N})\n'
        f'Stars = hubs, dot size ~ population, arrows = selected routes',
        fontsize=12, y=0.98)

    for idx, (airline, color) in enumerate(zip(picks, colors)):
        ax = axes[idx // 2, idx % 2]
        plot_airline_network(ax, locations, populations, hubs[airline],
                             obs_bundles[airline], endpoints_a, endpoints_b, C,
                             airline, edges_color=color)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = Path(__file__).resolve().parent / 'networks_seed42.png'
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    print(f"Saved to {out_path}")
    plt.close()


if __name__ == '__main__':
    main()
