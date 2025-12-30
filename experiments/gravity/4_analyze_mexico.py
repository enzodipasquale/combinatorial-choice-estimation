"""
Analyze Mexico firms from gravity simulation.
Creates plots similar to firms_export application.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def load_mexico_data():
    """Load simulation data and extract Mexico firms."""
    sim = np.load('datasets/quad_simulation.npz')
    bundles = sim['bundles']
    home_countries = sim['home_countries']
    country_names = list(sim['country_names'])
    
    # Find Mexico
    mex_idx = country_names.index('MEX')
    mex_firms = np.where(home_countries == mex_idx)[0]
    
    # Extract Mexico firm choices
    mex_bundles = bundles[mex_firms]
    
    # Remove Mexico column (no self-trade)
    mex_export_choices = np.delete(mex_bundles, mex_idx, axis=1)
    dest_countries = [c for i, c in enumerate(country_names) if i != mex_idx]
    
    print(f"✓ Loaded Mexico data:")
    print(f"  Mexico firms: {len(mex_firms)}")
    print(f"  Destination countries: {len(dest_countries)}")
    print(f"  Total exports: {mex_export_choices.sum()}")
    
    return mex_export_choices, dest_countries, mex_firms


def plot_marginals(mex_bundles, dest_countries):
    """
    Plot fraction of firms exporting to each destination.
    Similar to firms_export marginals.png
    """
    # Calculate demand (fraction of firms exporting to each destination)
    demand = mex_bundles.mean(axis=0)  # Fraction of firms exporting to each country
    
    # Sort by demand
    sorted_idx = np.argsort(demand)[::-1]
    sorted_demand = demand[sorted_idx]
    sorted_countries = [dest_countries[i] for i in sorted_idx]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(sorted_demand))
    ax.plot(x, sorted_demand, 'o-', linewidth=2, markersize=6, 
            color='steelblue', label='Simulated Demand')
    
    ax.set_xlabel('Destinations (sorted by demand)', fontsize=12)
    ax.set_ylabel('Fraction of Mexico Firms Exporting', fontsize=12)
    ax.set_title('Mexico Export Destinations: Fraction of Firms per Country', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Annotate top destinations
    for i in range(min(5, len(sorted_countries))):
        ax.annotate(sorted_countries[i], 
                   xy=(i, sorted_demand[i]), 
                   xytext=(5, 5), 
                   textcoords='offset points',
                   fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('datasets/mexico_marginals.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: datasets/mexico_marginals.png")
    
    return sorted_countries[:10], sorted_demand[:10]


def plot_destination_distribution(mex_bundles):
    """
    Plot histogram of number of destinations per firm.
    Similar to firms_export count_distribution.png
    """
    # Count destinations per firm
    destinations_per_firm = mex_bundles.sum(axis=1)
    
    # Create histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bins = np.arange(0, destinations_per_firm.max() + 2) - 0.5
    ax.hist(destinations_per_firm, bins=bins, alpha=0.7, 
            color='steelblue', edgecolor='black', linewidth=1.2)
    
    ax.set_xlabel('Number of Export Destinations', fontsize=12)
    ax.set_ylabel('Number of Firms', fontsize=12)
    ax.set_title('Mexico Firms: Distribution of Export Destinations', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    stats_text = f"Total Firms: {len(destinations_per_firm)}\n"
    stats_text += f"Mean: {destinations_per_firm.mean():.1f}\n"
    stats_text += f"Median: {np.median(destinations_per_firm):.0f}\n"
    stats_text += f"Range: {destinations_per_firm.min()}-{destinations_per_firm.max()}"
    
    ax.text(0.97, 0.97, stats_text, 
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('datasets/mexico_count_distribution.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: datasets/mexico_count_distribution.png")
    
    return destinations_per_firm


def plot_top_destinations(mex_bundles, dest_countries):
    """Plot top export destinations with bar chart."""
    demand = mex_bundles.mean(axis=0)
    sorted_idx = np.argsort(demand)[::-1][:15]
    
    top_countries = [dest_countries[i] for i in sorted_idx]
    top_demand = demand[sorted_idx]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_countries)))
    bars = ax.barh(range(len(top_countries)), top_demand, color=colors, edgecolor='black', linewidth=0.8)
    
    ax.set_yticks(range(len(top_countries)))
    ax.set_yticklabels(top_countries, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Fraction of Mexico Firms', fontsize=12)
    ax.set_title('Top 15 Export Destinations for Mexican Firms', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add percentages
    for i, (bar, val) in enumerate(zip(bars, top_demand)):
        ax.text(val + 0.01, i, f'{val*100:.1f}%', 
                va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('datasets/mexico_top_destinations.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: datasets/mexico_top_destinations.png")


def create_comprehensive_plot(mex_bundles, dest_countries):
    """Create a comprehensive 4-panel plot."""
    fig = plt.figure(figsize=(14, 10))
    
    # Panel 1: Marginals
    ax1 = plt.subplot(2, 2, 1)
    demand = mex_bundles.mean(axis=0)
    sorted_idx = np.argsort(demand)[::-1]
    sorted_demand = demand[sorted_idx]
    ax1.plot(sorted_demand, 'o-', linewidth=2, markersize=4, color='steelblue')
    ax1.set_xlabel('Destinations (sorted by demand)', fontsize=10)
    ax1.set_ylabel('Fraction of Firms', fontsize=10)
    ax1.set_title('Export Demand by Destination', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Destination distribution
    ax2 = plt.subplot(2, 2, 2)
    destinations_per_firm = mex_bundles.sum(axis=1)
    bins = np.arange(0, destinations_per_firm.max() + 2) - 0.5
    ax2.hist(destinations_per_firm, bins=bins, alpha=0.7, 
            color='coral', edgecolor='black', linewidth=1)
    ax2.set_xlabel('Number of Destinations', fontsize=10)
    ax2.set_ylabel('Number of Firms', fontsize=10)
    ax2.set_title('Distribution of Firm Export Intensity', fontsize=11, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Panel 3: Top destinations bar
    ax3 = plt.subplot(2, 2, 3)
    top_idx = sorted_idx[:10]
    top_countries = [dest_countries[i] for i in top_idx]
    top_demand = demand[top_idx]
    bars = ax3.barh(range(len(top_countries)), top_demand, 
                    color=plt.cm.viridis(np.linspace(0.3, 0.9, len(top_countries))))
    ax3.set_yticks(range(len(top_countries)))
    ax3.set_yticklabels(top_countries, fontsize=9)
    ax3.invert_yaxis()
    ax3.set_xlabel('Fraction of Firms', fontsize=10)
    ax3.set_title('Top 10 Destinations', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Panel 4: Summary statistics
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    stats = [
        "MEXICO EXPORT SIMULATION",
        "=" * 35,
        f"Total Firms: {len(mex_bundles):,}",
        f"Destination Countries: {len(dest_countries)}",
        f"Total Exports: {mex_bundles.sum():,}",
        "",
        "DESTINATIONS PER FIRM:",
        f"  Mean: {destinations_per_firm.mean():.1f}",
        f"  Median: {np.median(destinations_per_firm):.0f}",
        f"  Range: {destinations_per_firm.min()}-{destinations_per_firm.max()}",
        f"  Std Dev: {destinations_per_firm.std():.1f}",
        "",
        "TOP 3 DESTINATIONS:",
    ]
    
    for i in range(3):
        stats.append(f"  {i+1}. {top_countries[i]}: {top_demand[i]*100:.1f}%")
    
    stats.extend([
        "",
        "SPARSITY:",
        f"  {(1 - mex_bundles.mean())*100:.1f}% of pairs unused"
    ])
    
    ax4.text(0.1, 0.95, '\n'.join(stats), 
            transform=ax4.transAxes,
            fontsize=10,
            verticalalignment='top',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.suptitle('Mexico Firm Export Analysis (Gravity Simulation)', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig('datasets/mexico_comprehensive.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: datasets/mexico_comprehensive.png")


def main():
    print("="*60)
    print("MEXICO EXPORT ANALYSIS")
    print("="*60)
    
    # Load data
    mex_bundles, dest_countries, mex_firms = load_mexico_data()
    
    print("\nCreating visualizations...")
    
    # 1. Marginals plot (like firms_export)
    top_countries, top_demand = plot_marginals(mex_bundles, dest_countries)
    
    # 2. Destination distribution (like firms_export)
    dest_per_firm = plot_destination_distribution(mex_bundles)
    
    # 3. Top destinations bar chart
    plot_top_destinations(mex_bundles, dest_countries)
    
    # 4. Comprehensive plot
    create_comprehensive_plot(mex_bundles, dest_countries)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nMexico firms: {len(mex_bundles)}")
    print(f"Avg destinations: {dest_per_firm.mean():.1f}")
    print(f"Total exports: {mex_bundles.sum()}")
    
    print(f"\nTop 10 destinations:")
    for i, (country, demand) in enumerate(zip(top_countries, top_demand)):
        print(f"  {i+1}. {country}: {demand*100:.1f}% of firms")
    
    print("\n" + "="*60)
    print("✅ MEXICO ANALYSIS COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()

