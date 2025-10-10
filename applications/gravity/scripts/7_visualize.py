"""
Visualize bilateral trade flows from simulation.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
import networkx as nx

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


def load_simulation():
    """Load simulation results."""
    sim = np.load('data/simulation/obs_bundles.npz')
    features = pd.read_csv('data/features/country_features.csv', index_col=0)
    
    return sim, features


def plot_bilateral_heatmap(bilateral_flows, countries, top_n=20):
    """Heatmap of bilateral trade flows."""
    # Select top N countries by total trade
    total_trade = bilateral_flows.sum(axis=0) + bilateral_flows.sum(axis=1)
    top_idx = np.argsort(total_trade)[::-1][:top_n]
    
    # Subset matrix
    subset = bilateral_flows[np.ix_(top_idx, top_idx)]
    labels = [countries[i] for i in top_idx]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(subset, xticklabels=labels, yticklabels=labels,
                cmap='YlOrRd', annot=False, fmt='d', cbar_kws={'label': 'Number of Exporters'})
    plt.title(f'Bilateral Export Flows (Top {top_n} Countries)', fontsize=16, fontweight='bold')
    plt.xlabel('Destination Country', fontsize=12)
    plt.ylabel('Origin Country', fontsize=12)
    plt.tight_layout()
    plt.savefig('data/plots/bilateral_heatmap.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: bilateral_heatmap.png")


def plot_network(bilateral_flows, countries, features, threshold_pct=95):
    """Network diagram of major trade flows."""
    # Only show top X percentile of flows
    threshold = np.percentile(bilateral_flows[bilateral_flows > 0], threshold_pct)
    
    # Create network
    G = nx.DiGraph()
    
    # Add nodes (countries)
    gdp = features['gdp_billions'].fillna(0).values
    for i, country in enumerate(countries):
        G.add_node(country, size=gdp[i])
    
    # Add edges (major flows only)
    for i in range(len(countries)):
        for j in range(len(countries)):
            if i != j and bilateral_flows[i, j] >= threshold:
                G.add_edge(countries[i], countries[j], weight=bilateral_flows[i, j])
    
    # Layout
    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Node sizes proportional to GDP
    node_sizes = [G.nodes[n]['size'] * 20 for n in G.nodes()]
    
    # Draw
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue',
                          alpha=0.8, edgecolors='navy', linewidths=2)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')
    
    # Edge widths proportional to flow
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    if weights:
        max_weight = max(weights)
        edge_widths = [w/max_weight * 5 for w in weights]
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.4, 
                              edge_color='red', arrows=True, arrowsize=15)
    
    plt.title(f'Major Trade Flows (Top {100-threshold_pct}% of Bilateral Relationships)', 
              fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('data/plots/trade_network.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: trade_network.png")


def plot_gravity_patterns(bilateral_flows, countries, features, obs_bundles):
    """Check gravity equation patterns."""
    distances = pd.read_csv('data/features/distances.csv', index_col=0).values
    gdp_dest = features['gdp_billions'].fillna(0).values
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Distance decay
    pairs_dist = []
    pairs_flow = []
    for i in range(len(countries)):
        for j in range(len(countries)):
            if i != j and bilateral_flows[i, j] > 0:
                pairs_dist.append(distances[i, j])
                pairs_flow.append(bilateral_flows[i, j])
    
    axes[0].scatter(pairs_dist, pairs_flow, alpha=0.3, s=20)
    axes[0].set_xlabel('Distance (km)', fontsize=11)
    axes[0].set_ylabel('Number of Exporters', fontsize=11)
    axes[0].set_title('Distance Decay Effect', fontsize=13, fontweight='bold')
    axes[0].set_xscale('log')
    if len(pairs_flow) > 0:
        axes[0].set_yscale('log')
    axes[0].grid(alpha=0.3)
    
    # 2. Market size effect (GDP)
    inflows = bilateral_flows.sum(axis=0)
    axes[1].scatter(gdp_dest, inflows, s=100, alpha=0.6, c='green')
    for i, country in enumerate(countries[:15]):  # Label top 15
        if gdp_dest[i] > 1000:
            axes[1].annotate(country, (gdp_dest[i], inflows[i]), fontsize=8)
    axes[1].set_xlabel('Destination GDP (billions $)', fontsize=11)
    axes[1].set_ylabel('Total Exporters', fontsize=11)
    axes[1].set_title('Market Size Effect', fontsize=13, fontweight='bold')
    axes[1].set_xscale('log')
    axes[1].grid(alpha=0.3)
    
    # 3. Export intensity distribution (FIRM-LEVEL)
    destinations_per_firm = obs_bundles.sum(axis=1)  # Fixed: use obs_bundles directly!
    axes[2].hist(destinations_per_firm, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    axes[2].set_xlabel('Number of Export Destinations', fontsize=11)
    axes[2].set_ylabel('Number of Firms', fontsize=11)
    axes[2].set_title('Export Intensity Distribution', fontsize=13, fontweight='bold')
    axes[2].axvline(destinations_per_firm.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {destinations_per_firm.mean():.1f}')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/plots/gravity_patterns.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: gravity_patterns.png")


def plot_regional_flows(bilateral_flows, countries, features):
    """Regional trade integration visualization."""
    region_cols = [c for c in features.columns if c in ['Europe', 'Asia', 'Africa', 'America', 'Americas', 'Oceania']]
    
    if not region_cols:
        print("  ! No region data available")
        return
    
    # Assign each country to a region
    country_regions = {}
    for i, country in enumerate(countries):
        for region in region_cols:
            if features.loc[country, region] == 1:
                country_regions[country] = region
                break
    
    # Compute intra vs inter-regional trade
    regions = list(set(country_regions.values()))
    region_matrix = np.zeros((len(regions), len(regions)))
    
    for i, c1 in enumerate(countries):
        for j, c2 in enumerate(countries):
            if c1 in country_regions and c2 in country_regions:
                r1 = regions.index(country_regions[c1])
                r2 = regions.index(country_regions[c2])
                region_matrix[r1, r2] += bilateral_flows[i, j]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(region_matrix, xticklabels=regions, yticklabels=regions,
                annot=True, fmt='.0f', cmap='Blues', cbar_kws={'label': 'Number of Exporters'})
    plt.title('Regional Trade Integration', fontsize=16, fontweight='bold')
    plt.xlabel('Destination Region', fontsize=12)
    plt.ylabel('Origin Region', fontsize=12)
    plt.tight_layout()
    plt.savefig('data/plots/regional_integration.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: regional_integration.png")


def main():
    print("="*60)
    print("VISUALIZING BILATERAL TRADE FLOWS")
    print("="*60)
    
    # Load
    sim, features = load_simulation()
    obs_bundles = sim['obs_bundles']
    home_countries = sim['home_countries']
    countries = sim['country_names']
    
    print(f"\n✓ Loaded simulation: {len(obs_bundles)} firms × {len(countries)} countries")
    
    # Compute bilateral flows
    n = len(countries)
    bilateral = np.zeros((n, n))
    for i in range(len(obs_bundles)):
        home = home_countries[i]
        for dest in np.where(obs_bundles[i])[0]:
            bilateral[home, dest] += 1
    
    print(f"\nGenerating visualizations...")
    
    # 1. Heatmap
    plot_bilateral_heatmap(bilateral, countries, top_n=20)
    
    # 2. Network
    plot_network(bilateral, countries, features, threshold_pct=95)
    
    # 3. Gravity patterns
    plot_gravity_patterns(bilateral, countries, features, obs_bundles)
    
    # 4. Regional integration
    plot_regional_flows(bilateral, countries, features)
    
    print(f"\n{'='*60}")
    print("✅ ALL VISUALIZATIONS SAVED TO data/")
    print(f"{'='*60}")
    print("  - bilateral_heatmap.png")
    print("  - trade_network.png")
    print("  - gravity_patterns.png")
    print("  - regional_integration.png")


if __name__ == '__main__':
    main()
