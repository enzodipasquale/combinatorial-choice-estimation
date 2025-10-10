"""
Enhanced visualizations comparing simulated vs real trade flows.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150


def load_all_data():
    """Load simulation and real trade data."""
    sim = np.load('data/simulation/obs_bundles.npz')
    real = np.load('data/features/real_trade_data.npz', allow_pickle=True)
    features = pd.read_csv('data/features/country_features.csv', index_col=0)
    distances = pd.read_csv('data/features/distances.csv', index_col=0)
    
    return sim, real, features, distances


def compute_bilateral_flows(obs_bundles, home_countries):
    """Convert bundles to bilateral flow matrix."""
    n = obs_bundles.shape[1]
    bilateral = np.zeros((n, n))
    
    for i in range(len(obs_bundles)):
        home = home_countries[i]
        for dest in np.where(obs_bundles[i])[0]:
            bilateral[home, dest] += 1
    
    return bilateral


def plot_simulated_vs_real_comparison(sim_bilateral, real_bilateral, countries):
    """Compare simulated vs real trade patterns."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Normalize both to same scale for comparison
    sim_norm = sim_bilateral / sim_bilateral.sum() if sim_bilateral.sum() > 0 else sim_bilateral
    real_norm = real_bilateral / real_bilateral.sum() if real_bilateral.sum() > 0 else real_bilateral
    
    # 1. Scatter: Simulated vs Real
    sim_flat = sim_norm.flatten()
    real_flat = real_norm.flatten()
    
    # Remove zeros for log scale
    nonzero = (sim_flat > 0) & (real_flat > 0)
    if nonzero.sum() > 0:
        corr, pval = pearsonr(sim_flat[nonzero], real_flat[nonzero])
        
        axes[0, 0].scatter(real_flat[nonzero], sim_flat[nonzero], alpha=0.5, s=20)
        axes[0, 0].plot([0, real_flat.max()], [0, real_flat.max()], 'r--', label='45° line', linewidth=2)
        axes[0, 0].set_xlabel('Real Trade Flow Intensity', fontsize=12)
        axes[0, 0].set_ylabel('Simulated Flow Intensity', fontsize=12)
        axes[0, 0].set_title(f'Simulated vs Real Trade Flows\nCorrelation: {corr:.3f} (p={pval:.3f})', 
                            fontsize=13, fontweight='bold')
        axes[0, 0].set_xscale('log')
        axes[0, 0].set_yscale('log')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
    
    # 2. Inflow distribution: Simulated vs Real
    sim_inflows = sim_bilateral.sum(axis=0)
    real_inflows = real_bilateral.sum(axis=0)
    
    # Normalize
    sim_inflows = sim_inflows / sim_inflows.sum() * 100 if sim_inflows.sum() > 0 else sim_inflows
    real_inflows = real_inflows / real_inflows.sum() * 100 if real_inflows.sum() > 0 else real_inflows
    
    # Sort by real inflows
    sorted_idx = np.argsort(real_inflows)[::-1][:20]
    
    x = np.arange(len(sorted_idx))
    width = 0.35
    
    axes[0, 1].bar(x - width/2, real_inflows[sorted_idx], width, label='Real', alpha=0.7, color='steelblue')
    axes[0, 1].bar(x + width/2, sim_inflows[sorted_idx], width, label='Simulated', alpha=0.7, color='coral')
    axes[0, 1].set_xlabel('Country (sorted by real inflows)', fontsize=12)
    axes[0, 1].set_ylabel('% of Total Inflows', fontsize=12)
    axes[0, 1].set_title('Top 20 Destinations: Simulated vs Real', fontsize=13, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([countries[i] for i in sorted_idx], rotation=45, ha='right')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3, axis='y')
    
    # 3. Distance elasticity check
    distances_df = pd.read_csv('data/features/distances.csv', index_col=0)
    
    sim_pairs_dist = []
    sim_pairs_flow = []
    real_pairs_dist = []
    real_pairs_flow = []
    
    for i in range(len(countries)):
        for j in range(len(countries)):
            if i != j:
                dist = distances_df.iloc[i, j]
                if sim_bilateral[i, j] > 0:
                    sim_pairs_dist.append(dist)
                    sim_pairs_flow.append(sim_bilateral[i, j])
                if real_bilateral[i, j] > 0:
                    real_pairs_dist.append(dist)
                    real_pairs_flow.append(real_bilateral[i, j])
    
    axes[1, 0].scatter(sim_pairs_dist, sim_pairs_flow, alpha=0.4, s=15, label='Simulated', color='coral')
    axes[1, 0].scatter(real_pairs_dist, real_pairs_flow, alpha=0.4, s=15, label='Real', color='steelblue')
    axes[1, 0].set_xlabel('Distance (km)', fontsize=12)
    axes[1, 0].set_ylabel('Trade Flow', fontsize=12)
    axes[1, 0].set_title('Distance Decay: Simulated vs Real', fontsize=13, fontweight='bold')
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # 4. Trade concentration (Lorenz curve)
    sim_sorted = np.sort(sim_inflows)[::-1]
    real_sorted = np.sort(real_inflows)[::-1]
    
    sim_cumsum = np.cumsum(sim_sorted) / sim_sorted.sum() if sim_sorted.sum() > 0 else sim_sorted
    real_cumsum = np.cumsum(real_sorted) / real_sorted.sum() if real_sorted.sum() > 0 else real_sorted
    
    x_pct = np.arange(1, len(sim_cumsum)+1) / len(sim_cumsum) * 100
    
    axes[1, 1].plot(x_pct, sim_cumsum * 100, label='Simulated', linewidth=2, color='coral')
    axes[1, 1].plot(x_pct, real_cumsum * 100, label='Real', linewidth=2, color='steelblue')
    axes[1, 1].plot([0, 100], [0, 100], 'k--', alpha=0.3, label='Perfect equality')
    axes[1, 1].set_xlabel('Cumulative % of Countries', fontsize=12)
    axes[1, 1].set_ylabel('Cumulative % of Inflows', fontsize=12)
    axes[1, 1].set_title('Trade Concentration (Lorenz Curve)', fontsize=13, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/plots/comparison_simulated_vs_real.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: comparison_simulated_vs_real.png")


def plot_trade_intensity_map(bilateral, countries, features, sim):
    """Heatmap showing export/import intensity by region and GDP."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Sort countries by GDP
    gdp = features['gdp_billions'].fillna(0).values
    sorted_idx = np.argsort(gdp)[::-1]
    
    # Reorder matrix
    bilateral_sorted = bilateral[np.ix_(sorted_idx, sorted_idx)]
    sorted_countries = [countries[i] for i in sorted_idx]
    
    # Plot 1: Full matrix (top 30)
    n_show = min(30, len(sorted_countries))
    sns.heatmap(bilateral_sorted[:n_show, :n_show], 
                xticklabels=sorted_countries[:n_show],
                yticklabels=sorted_countries[:n_show],
                cmap='YlOrRd', ax=axes[0], cbar_kws={'label': 'Export Flow'})
    axes[0].set_title('Bilateral Flows (Sorted by GDP)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Destination', fontsize=11)
    axes[0].set_ylabel('Origin', fontsize=11)
    
    # Plot 2: Export vs Import intensity
    exports_per_country = bilateral.sum(axis=1)  # Total outflows
    imports_per_country = bilateral.sum(axis=0)  # Total inflows
    
    # Normalize by firm count in each country
    home_counts = np.bincount([sim['home_countries'][i] for i in range(len(sim['obs_bundles']))], 
                               minlength=len(countries))
    export_intensity = np.divide(exports_per_country, home_counts, 
                                 out=np.zeros_like(exports_per_country, dtype=float), 
                                 where=home_counts>0)
    
    axes[1].scatter(gdp, export_intensity, s=100, alpha=0.6, c='green', label='Export Intensity')
    axes[1].scatter(gdp, imports_per_country, s=100, alpha=0.6, c='blue', marker='s', label='Import Attractiveness')
    
    # Label top countries
    for i in range(min(15, len(countries))):
        if gdp[i] > 1000:
            axes[1].annotate(countries[i], (gdp[i], export_intensity[i]), fontsize=8, alpha=0.7)
    
    axes[1].set_xlabel('GDP (billions $)', fontsize=12)
    axes[1].set_ylabel('Trade Intensity', fontsize=12)
    axes[1].set_title('Export vs Import Intensity by GDP', fontsize=14, fontweight='bold')
    axes[1].set_xscale('log')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/plots/trade_intensity_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: trade_intensity_analysis.png")


def plot_gravity_equation_fit(bilateral, countries, features, distances):
    """Check if simulated data fits gravity equation."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    gdp_origin = features['gdp_billions'].fillna(0).values
    gdp_dest = features['gdp_billions'].fillna(0).values
    dist_matrix = distances.values
    
    # Prepare data for regression
    flows = []
    gdp_o = []
    gdp_d = []
    dists = []
    
    for i in range(len(countries)):
        for j in range(len(countries)):
            if i != j and bilateral[i, j] > 0:
                flows.append(bilateral[i, j])
                gdp_o.append(gdp_origin[i])
                gdp_d.append(gdp_dest[j])
                dists.append(dist_matrix[i, j])
    
    flows = np.array(flows)
    gdp_o = np.array(gdp_o)
    gdp_d = np.array(gdp_d)
    dists = np.array(dists)
    
    # Log-log plots (gravity equation)
    # 1. Origin GDP effect
    axes[0, 0].scatter(gdp_o, flows, alpha=0.4, s=20)
    axes[0, 0].set_xlabel('Origin GDP (billions $)', fontsize=11)
    axes[0, 0].set_ylabel('Export Flow', fontsize=11)
    axes[0, 0].set_title('Origin Market Size Effect', fontsize=13, fontweight='bold')
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Destination GDP effect
    axes[0, 1].scatter(gdp_d, flows, alpha=0.4, s=20, color='green')
    axes[0, 1].set_xlabel('Destination GDP (billions $)', fontsize=11)
    axes[0, 1].set_ylabel('Export Flow', fontsize=11)
    axes[0, 1].set_title('Destination Market Size Effect', fontsize=13, fontweight='bold')
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(alpha=0.3)
    
    # 3. Distance effect (negative)
    axes[1, 0].scatter(dists, flows, alpha=0.4, s=20, color='red')
    axes[1, 0].set_xlabel('Distance (km)', fontsize=11)
    axes[1, 0].set_ylabel('Export Flow', fontsize=11)
    axes[1, 0].set_title('Distance Decay Effect (Gravity)', fontsize=13, fontweight='bold')
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(alpha=0.3)
    
    # 4. Gravity equation fit: log(Flow) vs log(GDP_i * GDP_j / Distance)
    gravity_index = (gdp_o * gdp_d) / (dists ** 1.1)  # Classic gravity
    
    axes[1, 1].scatter(gravity_index, flows, alpha=0.4, s=20, color='purple')
    
    # Fit line
    log_g = np.log(gravity_index + 1)
    log_f = np.log(flows + 1)
    z = np.polyfit(log_g, log_f, 1)
    p = np.poly1d(z)
    
    x_fit = np.sort(gravity_index)
    y_fit = np.exp(p(np.log(x_fit + 1)))
    axes[1, 1].plot(x_fit, y_fit, "r-", linewidth=2, label=f'Fit: slope={z[0]:.2f}')
    
    axes[1, 1].set_xlabel('Gravity Index: GDP_i × GDP_j / Distance^1.1', fontsize=11)
    axes[1, 1].set_ylabel('Export Flow', fontsize=11)
    axes[1, 1].set_title('Gravity Equation Fit', fontsize=13, fontweight='bold')
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/plots/gravity_equation_fit.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: gravity_equation_fit.png")


def plot_trade_statistics(sim_bilateral, real_bilateral, countries, features, sim):
    """Interesting trade statistics."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Trade partner distribution
    partners_per_country_sim = (sim_bilateral > 0).sum(axis=1)
    partners_per_country_real = (real_bilateral > 0).sum(axis=1)
    
    axes[0, 0].hist(partners_per_country_sim, bins=20, alpha=0.6, label='Simulated', color='coral')
    axes[0, 0].hist(partners_per_country_real, bins=20, alpha=0.6, label='Real', color='steelblue')
    axes[0, 0].set_xlabel('Number of Export Partners', fontsize=11)
    axes[0, 0].set_ylabel('Number of Countries', fontsize=11)
    axes[0, 0].set_title('Export Partner Distribution', fontsize=13, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Trade reciprocity
    reciprocal_sim = ((sim_bilateral > 0) & (sim_bilateral.T > 0)).sum() / (sim_bilateral > 0).sum()
    reciprocal_real = ((real_bilateral > 0) & (real_bilateral.T > 0)).sum() / (real_bilateral > 0).sum() if (real_bilateral > 0).sum() > 0 else 0
    
    axes[0, 1].bar(['Simulated', 'Real'], [reciprocal_sim * 100, reciprocal_real * 100], 
                   color=['coral', 'steelblue'], alpha=0.7)
    axes[0, 1].set_ylabel('Reciprocal Trade %', fontsize=11)
    axes[0, 1].set_title('Trade Reciprocity\n(A→B and B→A both exist)', fontsize=13, fontweight='bold')
    axes[0, 1].grid(alpha=0.3, axis='y')
    
    # 3. Largest exporters
    sim_outflows = sim_bilateral.sum(axis=1)
    real_outflows = real_bilateral.sum(axis=0)  # Use inflows as proxy
    
    top_sim = np.argsort(sim_outflows)[::-1][:10]
    top_real = np.argsort(real_outflows)[::-1][:10]
    
    y_pos = np.arange(10)
    axes[0, 2].barh(y_pos, [sim_outflows[i] for i in top_sim], alpha=0.7, color='coral')
    axes[0, 2].set_yticks(y_pos)
    axes[0, 2].set_yticklabels([countries[i] for i in top_sim])
    axes[0, 2].set_xlabel('Total Exports', fontsize=11)
    axes[0, 2].set_title('Top 10 Exporting Countries (Simulated)', fontsize=13, fontweight='bold')
    axes[0, 2].grid(alpha=0.3, axis='x')
    axes[0, 2].invert_yaxis()
    
    # 4. Trade balance
    sim_balance = sim_bilateral.sum(axis=1) - sim_bilateral.sum(axis=0)
    axes[1, 0].hist(sim_balance, bins=30, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Trade Balance (Exports - Imports)', fontsize=11)
    axes[1, 0].set_ylabel('Number of Countries', fontsize=11)
    axes[1, 0].set_title('Trade Balance Distribution', fontsize=13, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    
    # 5. Network density by region
    region_cols = [c for c in features.columns if c in ['Europe', 'Asia', 'Africa', 'America', 'Americas', 'Oceania']]
    if region_cols:
        regions = list(set([c for c in region_cols]))
        densities = []
        region_names = []
        
        for region in regions:
            in_region = features[region] == 1
            indices = np.where(in_region)[0]
            if len(indices) > 1:
                submatrix = sim_bilateral[np.ix_(indices, indices)]
                density = (submatrix > 0).sum() / (len(indices) * (len(indices) - 1)) if len(indices) > 1 else 0
                densities.append(density * 100)
                region_names.append(region)
        
        if region_names:
            axes[1, 1].bar(region_names, densities, alpha=0.7, color='teal')
            axes[1, 1].set_ylabel('Network Density (%)', fontsize=11)
            axes[1, 1].set_title('Intra-Regional Trade Density', fontsize=13, fontweight='bold')
            axes[1, 1].grid(alpha=0.3, axis='y')
    
    # 6. Top 5 bilateral corridors
    top_corridors = []
    for i in range(len(countries)):
        for j in range(len(countries)):
            if i != j and sim_bilateral[i, j] > 0:
                top_corridors.append((countries[i], countries[j], sim_bilateral[i, j]))
    
    top_corridors.sort(key=lambda x: x[2], reverse=True)
    
    corridor_labels = [f"{o}→{d}" for o, d, _ in top_corridors[:10]]
    corridor_values = [v for _, _, v in top_corridors[:10]]
    
    y_pos = np.arange(10)
    axes[1, 2].barh(y_pos, corridor_values, alpha=0.7, color='orange')
    axes[1, 2].set_yticks(y_pos)
    axes[1, 2].set_yticklabels(corridor_labels, fontsize=9)
    axes[1, 2].set_xlabel('Number of Firms', fontsize=11)
    axes[1, 2].set_title('Top 10 Bilateral Trade Corridors', fontsize=13, fontweight='bold')
    axes[1, 2].grid(alpha=0.3, axis='x')
    axes[1, 2].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('data/plots/trade_statistics.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: trade_statistics.png")


def main():
    print("="*60)
    print("ENHANCED TRADE FLOW VISUALIZATIONS")
    print("="*60)
    
    # Load data
    sim, real, features, distances = load_all_data()
    
    obs_bundles = sim['obs_bundles']
    home_countries = sim['home_countries']
    countries = sim['country_names']
    real_bilateral = real['bilateral_flows']
    
    print(f"\n✓ Loaded:")
    print(f"  Simulated: {len(obs_bundles)} firms × {len(countries)} countries")
    print(f"  Real trade data: {len(countries)} countries")
    
    # Compute bilateral flows
    sim_bilateral = compute_bilateral_flows(obs_bundles, home_countries)
    
    print(f"\nGenerating visualizations...")
    
    # 1. Simulated vs Real comparison
    plot_simulated_vs_real_comparison(sim_bilateral, real_bilateral, countries)
    
    # 2. Gravity equation validation
    plot_gravity_equation_fit(sim_bilateral, countries, features, distances)
    
    # 3. Trade statistics
    plot_trade_statistics(sim_bilateral, real_bilateral, countries, features, sim)
    
    # 4. Intensity analysis
    plot_trade_intensity_map(sim_bilateral, countries, features, sim)
    
    print(f"\n{'='*60}")
    print("✅ ALL VISUALIZATIONS COMPLETE")
    print(f"{'='*60}")
    print("\nCreated:")
    print("  1. comparison_simulated_vs_real.png - 4-panel comparison")
    print("  2. gravity_equation_fit.png - Gravity model validation")
    print("  3. trade_statistics.png - 6-panel statistics")
    print("  4. trade_intensity_analysis.png - Export/import patterns")
    print(f"\n{'='*60}")


if __name__ == '__main__':
    main()
