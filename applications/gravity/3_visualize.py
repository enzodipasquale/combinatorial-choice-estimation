"""
Step 3: Visualize simulation results.

Creates comprehensive plots analyzing trade flows.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

sns.set_style('whitegrid')


def load_results():
    """Load simulation results."""
    sim = np.load('datasets/quad_simulation.npz')
    features = pd.read_csv('datasets/country_features.csv', index_col=0)
    distances = pd.read_csv('datasets/distances.csv', index_col=0)
    
    return {
        'bundles': sim['bundles'],
        'home_countries': sim['home_countries'],
        'country_names': sim['country_names'],
        'theta_true': sim['theta_true']
    }, features, distances


def plot_comprehensive_analysis(sim, features, distances):
    """Create 6-panel comprehensive visualization."""
    bundles = sim['bundles']
    home_countries = sim['home_countries']
    countries = sim['country_names']
    
    num_firms = len(bundles)
    num_countries = len(countries)
    
    # Compute statistics
    bilateral_flows = bundles.sum(axis=0)
    partners_per_firm = bundles.sum(axis=1)
    
    # Check self-trade
    self_trade = sum([bundles[i, home_countries[i]] for i in range(num_firms)])
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Gravity Model: Trade Flow Analysis', fontsize=16, fontweight='bold')
    
    # 1. Export intensity distribution
    ax = axes[0, 0]
    ax.hist(partners_per_firm, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(partners_per_firm.mean(), color='red', linestyle='--', 
               label=f'Mean: {partners_per_firm.mean():.1f}')
    ax.set_xlabel('Number of Export Partners')
    ax.set_ylabel('Number of Firms')
    ax.set_title('Export Intensity Distribution')
    ax.legend()
    ax.text(0.05, 0.95, f'Self-trade: {self_trade/num_firms*100:.1f}%',
            transform=ax.transAxes, va='top', bbox=dict(boxstyle='round', facecolor='wheat'))
    
    # 2. Top destinations
    ax = axes[0, 1]
    top_idx = np.argsort(bilateral_flows)[::-1][:15]
    top_countries = [countries[i] for i in top_idx]
    top_flows = bilateral_flows[top_idx]
    
    ax.barh(range(len(top_countries)), top_flows)
    ax.set_yticks(range(len(top_countries)))
    ax.set_yticklabels(top_countries)
    ax.set_xlabel('Number of Exporting Firms')
    ax.set_title('Top 15 Export Destinations')
    ax.invert_yaxis()
    
    # 3. Market size vs inflows (gravity check)
    ax = axes[0, 2]
    gdp = features['gdp_billions'].values
    ax.scatter(gdp, bilateral_flows, alpha=0.6, s=100)
    
    # Add country labels for top destinations
    for idx in top_idx[:10]:
        ax.annotate(countries[idx], (gdp[idx], bilateral_flows[idx]), 
                   fontsize=8, alpha=0.7)
    
    ax.set_xlabel('GDP (billions USD)')
    ax.set_ylabel('Number of Exporting Firms')
    ax.set_title('Market Size Effect')
    ax.set_xscale('log')
    
    # Add correlation
    corr = np.corrcoef(np.log(gdp + 1), bilateral_flows)[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}',
            transform=ax.transAxes, va='top', bbox=dict(boxstyle='round', facecolor='lightblue'))
    
    # 4. Distance decay
    ax = axes[1, 0]
    avg_distances = []
    avg_flows = []
    
    dist_bins = np.logspace(2, 5, 20)
    dist_array = distances.values
    for i in range(len(dist_bins) - 1):
        # Find all country pairs in this distance bin
        flows_in_bin = []
        for j in range(num_countries):
            mask = (dist_array[:, j] >= dist_bins[i]) & (dist_array[:, j] < dist_bins[i + 1])
            mask = mask & (dist_array[:, j] > 0)
            if mask.any():
                flows_in_bin.append(bundles[:, j].mean())
        
        if flows_in_bin:
            avg_distances.append(np.sqrt(dist_bins[i] * dist_bins[i + 1]))
            avg_flows.append(np.mean(flows_in_bin))
    
    ax.plot(avg_distances, avg_flows, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Avg Export Probability')
    ax.set_title('Distance Decay Effect')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # 5. Trade network (top 30 flows)
    ax = axes[1, 1]
    G = nx.DiGraph()
    
    # Add nodes
    for i, country in enumerate(countries):
        G.add_node(country, size=gdp[i])
    
    # Add top edges
    edges = []
    for i in range(num_firms):
        home = countries[home_countries[i]]
        for j in range(num_countries):
            if bundles[i, j]:
                edges.append((home, countries[j]))
    
    edge_counts = pd.Series(edges).value_counts()
    top_edges = edge_counts.head(30)
    
    for (source, target), weight in top_edges.items():
        G.add_edge(source, target, weight=weight)
    
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Draw
    node_sizes = [np.log(G.nodes[node].get('size', 1) + 1) * 100 for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', 
                          alpha=0.7, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=6, ax=ax)
    
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_edges(G, pos, width=[w/50 for w in weights], alpha=0.3,
                          arrows=True, arrowsize=10, ax=ax)
    
    ax.set_title('Trade Network (Top 30 Flows)')
    ax.axis('off')
    
    # 6. Sparsity statistics
    ax = axes[1, 2]
    stats = {
        'Total firms': num_firms,
        'Countries': num_countries,
        'Avg partners': f'{partners_per_firm.mean():.1f}',
        'Partner range': f'{partners_per_firm.min()}-{partners_per_firm.max()}',
        'Sparsity': f'{(1 - bundles.sum()/(num_firms*num_countries))*100:.1f}%',
        'Self-trade': f'{self_trade/num_firms*100:.1f}%',
        'Total exports': int(bundles.sum())
    }
    
    text = '\n'.join([f'{k}: {v}' for k, v in stats.items()])
    ax.text(0.1, 0.5, text, transform=ax.transAxes, fontsize=14,
           verticalalignment='center', family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.set_title('Summary Statistics')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('datasets/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: datasets/comprehensive_analysis.png")
    plt.close()


def plot_parameter_sensitivity(sim, features):
    """Show which countries benefit from each parameter."""
    bundles = sim['bundles']
    countries = sim['country_names']
    theta = sim['theta_true']
    
    inflows = bundles.sum(axis=0)
    gdp = features['gdp_billions'].values
    pop = features['population_millions'].values
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Parameter Effects on Trade Flows', fontsize=14, fontweight='bold')
    
    # GDP effect
    ax = axes[0]
    ax.scatter(gdp, inflows, alpha=0.6, s=100)
    ax.set_xlabel('GDP (billions USD)')
    ax.set_ylabel('Exporting Firms')
    ax.set_title(f'GDP Effect (θ={theta[1]:.1f})')
    ax.set_xscale('log')
    
    # Population effect
    ax = axes[1]
    ax.scatter(pop, inflows, alpha=0.6, s=100, color='green')
    ax.set_xlabel('Population (millions)')
    ax.set_ylabel('Exporting Firms')
    ax.set_title(f'Population Effect (θ={theta[2]:.1f})')
    ax.set_xscale('log')
    
    # Combined
    ax = axes[2]
    expected_inflows = theta[1] * np.log(gdp + 1) + theta[2] * np.log(pop + 1)
    ax.scatter(expected_inflows, inflows, alpha=0.6, s=100, color='red')
    ax.plot([expected_inflows.min(), expected_inflows.max()],
           [expected_inflows.min(), expected_inflows.max()],
           'k--', alpha=0.3, label='Perfect fit')
    ax.set_xlabel('Expected Inflows (linear combination)')
    ax.set_ylabel('Actual Inflows')
    ax.set_title('Model Fit')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('datasets/parameter_effects.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: datasets/parameter_effects.png")
    plt.close()


def main():
    print("\n" + "="*60)
    print("VISUALIZING SIMULATION RESULTS")
    print("="*60 + "\n")
    
    sim, features, distances = load_results()
    
    print(f"Loaded simulation:")
    print(f"  Firms: {len(sim['bundles'])}")
    print(f"  Countries: {len(sim['country_names'])}")
    print(f"  Parameters: {sim['theta_true']}")
    
    print("\nCreating plots...")
    plot_comprehensive_analysis(sim, features, distances)
    plot_parameter_sensitivity(sim, features)
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()

