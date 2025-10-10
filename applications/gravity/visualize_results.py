"""
Visualize simulated trade flows.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

sns.set_style("whitegrid")


def load_data():
    sim = np.load('datasets/simulated_choices.npz')
    features = pd.read_csv('datasets/country_features.csv', index_col=0)
    distances = pd.read_csv('datasets/distances.csv', index_col=0)
    
    return sim, features, distances


def plot_comprehensive_analysis(sim, features, distances):
    """6-panel comprehensive visualization."""
    obs_bundles = sim['bundles']
    home_countries = sim['home_countries']
    countries = sim['country_names']
    
    # Compute bilateral flows
    n = len(countries)
    bilateral = np.zeros((n, n))
    for i in range(len(obs_bundles)):
        home = home_countries[i]
        for dest in np.where(obs_bundles[i])[0]:
            bilateral[home, dest] += 1
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 1. Partner distribution
    partners = obs_bundles.sum(axis=1)
    axes[0, 0].hist(partners, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(partners.mean(), color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {partners.mean():.1f}')
    axes[0, 0].set_xlabel('Export Partners per Firm', fontsize=12)
    axes[0, 0].set_ylabel('Number of Firms', fontsize=12)
    axes[0, 0].set_title('Export Intensity Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Top destinations
    inflows = bilateral.sum(axis=0)
    top_idx = np.argsort(inflows)[::-1][:15]
    axes[0, 1].barh(range(15), [inflows[i] for i in top_idx], color='coral', alpha=0.7)
    axes[0, 1].set_yticks(range(15))
    axes[0, 1].set_yticklabels([countries[i] for i in top_idx])
    axes[0, 1].set_xlabel('Number of Exporters', fontsize=12)
    axes[0, 1].set_title('Top 15 Export Destinations', fontsize=14, fontweight='bold')
    axes[0, 1].invert_yaxis()
    axes[0, 1].grid(alpha=0.3, axis='x')
    
    # 3. Bilateral heatmap (top 15)
    bilateral_top = bilateral[np.ix_(top_idx[:15], top_idx[:15])]
    sns.heatmap(bilateral_top, 
                xticklabels=[countries[i] for i in top_idx[:15]],
                yticklabels=[countries[i] for i in top_idx[:15]],
                cmap='YlOrRd', ax=axes[0, 2], cbar_kws={'label': 'Firms'})
    axes[0, 2].set_title('Bilateral Flows (Top 15×15)', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('Destination')
    axes[0, 2].set_ylabel('Origin')
    
    # 4. Distance decay
    pairs_dist = []
    pairs_flow = []
    dist_vals = distances.values
    for i in range(n):
        for j in range(n):
            if i != j and bilateral[i, j] > 0:
                pairs_dist.append(dist_vals[i, j])
                pairs_flow.append(bilateral[i, j])
    
    axes[1, 0].scatter(pairs_dist, pairs_flow, alpha=0.4, s=20, color='purple')
    axes[1, 0].set_xlabel('Distance (km)', fontsize=12)
    axes[1, 0].set_ylabel('Number of Firms', fontsize=12)
    axes[1, 0].set_title('Distance Decay Effect', fontsize=14, fontweight='bold')
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(alpha=0.3)
    
    # 5. GDP effect
    gdp = features['gdp_billions'].fillna(0).values
    axes[1, 1].scatter(gdp, inflows, s=100, alpha=0.6, c='green')
    # Label top economies
    for idx in top_idx[:10]:
        if gdp[idx] > 1000:
            axes[1, 1].annotate(countries[idx], (gdp[idx], inflows[idx]), fontsize=9)
    
    axes[1, 1].set_xlabel('GDP (billions $)', fontsize=12)
    axes[1, 1].set_ylabel('Number of Exporters', fontsize=12)
    axes[1, 1].set_title('Market Size Effect', fontsize=14, fontweight='bold')
    axes[1, 1].set_xscale('log')
    axes[1, 1].grid(alpha=0.3)
    
    # Compute correlation
    corr, pval = spearmanr(gdp, inflows)
    axes[1, 1].text(0.05, 0.95, f'Correlation: {corr:.3f}\n(p={pval:.3f})', 
                     transform=axes[1, 1].transAxes, fontsize=11,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                     verticalalignment='top')
    
    # 6. Top corridors
    corridors = []
    for i in range(n):
        for j in range(n):
            if i != j and bilateral[i, j] > 0:
                corridors.append((countries[i], countries[j], bilateral[i, j]))
    corridors.sort(key=lambda x: x[2], reverse=True)
    
    top_corr = corridors[:12]
    labels = [f"{o}→{d}" for o, d, _ in top_corr]
    values = [v for _, _, v in top_corr]
    
    axes[1, 2].barh(range(12), values, color='orange', alpha=0.7)
    axes[1, 2].set_yticks(range(12))
    axes[1, 2].set_yticklabels(labels, fontsize=10)
    axes[1, 2].set_xlabel('Number of Firms', fontsize=12)
    axes[1, 2].set_title('Top 12 Bilateral Corridors', fontsize=14, fontweight='bold')
    axes[1, 2].invert_yaxis()
    axes[1, 2].grid(alpha=0.3, axis='x')
    
    plt.suptitle(f'Gravity Model Simulation: {len(obs_bundles):,} Firms × {n} Countries',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('datasets/trade_flow_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✅ Saved: datasets/trade_flow_analysis.png")


def main():
    print("="*60)
    print("VISUALIZING TRADE FLOWS")
    print("="*60)
    
    sim, features, distances = load_data()
    plot_comprehensive_analysis(sim, features, distances)
    
    # Print summary
    obs = sim['bundles']
    partners = obs.sum(axis=1)
    
    print(f"\nSummary:")
    print(f"  Firms: {len(obs):,}")
    print(f"  Countries: {obs.shape[1]}")
    print(f"  Avg partners: {partners.mean():.1f}")
    print(f"  Partner range: {partners.min()}-{partners.max()}")
    print(f"  Total flows: {int(obs.sum()):,}")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    main()

