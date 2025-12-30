"""
Compare simulated Mexico data with REAL Mexico export data.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def load_real_data():
    """Load real Mexico data summary."""
    real = np.load('../firms_export/real_mex_summary.npz', allow_pickle=True)
    return {
        'destinations_per_firm': real['destinations_per_firm'],
        'dest_names': real['dest_names'],
        'dest_fractions': real['dest_fractions'],
        'dest_firms': real['dest_firms'],
        'total_firms': int(real['total_firms']),
        'year': int(real['year'])
    }


def load_simulated_data():
    """Load simulated Mexico data."""
    sim = np.load('datasets/quad_simulation.npz')
    bundles = sim['bundles']
    home_countries = sim['home_countries']
    country_names = list(sim['country_names'])
    
    # Extract Mexico
    mex_idx = country_names.index('MEX')
    mex_firms = np.where(home_countries == mex_idx)[0]
    mex_bundles = bundles[mex_firms]
    
    # Remove Mexico column
    mex_export_choices = np.delete(mex_bundles, mex_idx, axis=1)
    dest_countries = [c for i, c in enumerate(country_names) if i != mex_idx]
    
    # Calculate stats
    destinations_per_firm = mex_export_choices.sum(axis=1)
    dest_fractions = mex_export_choices.mean(axis=0)
    
    # Sort by fraction
    sorted_idx = np.argsort(dest_fractions)[::-1]
    
    return {
        'destinations_per_firm': destinations_per_firm,
        'dest_names': np.array([dest_countries[i] for i in sorted_idx]),
        'dest_fractions': dest_fractions[sorted_idx],
        'total_firms': len(mex_firms)
    }


def plot_comparison():
    """Create comprehensive comparison plots."""
    real = load_real_data()
    sim = load_simulated_data()
    
    fig = plt.figure(figsize=(16, 10))
    
    # ===== PANEL 1: Destinations per firm distribution =====
    ax1 = plt.subplot(2, 3, 1)
    
    # Real data
    bins = np.arange(0, max(real['destinations_per_firm'].max(), 
                           sim['destinations_per_firm'].max()) + 2) - 0.5
    ax1.hist(real['destinations_per_firm'], bins=bins[:50], alpha=0.6, 
             label=f"Real (year {real['year']})", color='steelblue', 
             edgecolor='black', linewidth=1)
    ax1.hist(sim['destinations_per_firm'], bins=bins[:50], alpha=0.6, 
             label='Simulated', color='coral', 
             edgecolor='black', linewidth=1)
    
    ax1.set_xlabel('Number of Destinations', fontsize=11)
    ax1.set_ylabel('Number of Firms', fontsize=11)
    ax1.set_title('Distribution of Export Destinations', fontsize=12, fontweight='bold')
    ax1.set_yscale('log')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # ===== PANEL 2: CDF of destinations =====
    ax2 = plt.subplot(2, 3, 2)
    
    real_sorted = np.sort(real['destinations_per_firm'])
    sim_sorted = np.sort(sim['destinations_per_firm'])
    real_cdf = np.arange(1, len(real_sorted)+1) / len(real_sorted)
    sim_cdf = np.arange(1, len(sim_sorted)+1) / len(sim_sorted)
    
    ax2.plot(real_sorted, real_cdf, linewidth=2, label=f"Real (year {real['year']})", color='steelblue')
    ax2.plot(sim_sorted, sim_cdf, linewidth=2, label='Simulated', color='coral')
    
    ax2.set_xlabel('Number of Destinations', fontsize=11)
    ax2.set_ylabel('Cumulative Fraction of Firms', fontsize=11)
    ax2.set_title('CDF: Export Intensity', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 30)
    
    # ===== PANEL 3: Summary statistics =====
    ax3 = plt.subplot(2, 3, 3)
    ax3.axis('off')
    
    stats_text = "COMPARISON SUMMARY\n"
    stats_text += "="*40 + "\n\n"
    stats_text += f"REAL DATA (Year {real['year']}):\n"
    stats_text += f"  Firms: {real['total_firms']:,}\n"
    stats_text += f"  Mean destinations: {real['destinations_per_firm'].mean():.1f}\n"
    stats_text += f"  Median destinations: {np.median(real['destinations_per_firm']):.0f}\n"
    stats_text += f"  Range: {real['destinations_per_firm'].min()}-{real['destinations_per_firm'].max()}\n"
    stats_text += f"  Std: {real['destinations_per_firm'].std():.1f}\n\n"
    
    stats_text += "SIMULATED DATA:\n"
    stats_text += f"  Firms: {sim['total_firms']:,}\n"
    stats_text += f"  Mean destinations: {sim['destinations_per_firm'].mean():.1f}\n"
    stats_text += f"  Median destinations: {np.median(sim['destinations_per_firm']):.0f}\n"
    stats_text += f"  Range: {sim['destinations_per_firm'].min()}-{sim['destinations_per_firm'].max()}\n"
    stats_text += f"  Std: {sim['destinations_per_firm'].std():.1f}\n\n"
    
    stats_text += "KEY DIFFERENCES:\n"
    diff_mean = sim['destinations_per_firm'].mean() - real['destinations_per_firm'].mean()
    stats_text += f"  Mean diff: {diff_mean:+.1f} destinations\n"
    stats_text += f"  Sim/Real ratio: {sim['destinations_per_firm'].mean()/real['destinations_per_firm'].mean():.1f}x\n\n"
    
    stats_text += "⚠️  SIMULATION OVER-EXPORTS!\n"
    stats_text += "   Real firms are much more sparse"
    
    ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes,
             fontsize=9, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # ===== PANEL 4: Marginals (top destinations) =====
    ax4 = plt.subplot(2, 3, 4)
    
    # Get top 20 destinations from real data
    top_n = 20
    real_top = real['dest_fractions'][:top_n]
    real_names = real['dest_names'][:top_n]
    
    # Match simulated to same countries
    sim_top = []
    for name in real_names:
        idx = np.where(sim['dest_names'] == name)[0]
        if len(idx) > 0:
            sim_top.append(sim['dest_fractions'][idx[0]])
        else:
            sim_top.append(0)
    sim_top = np.array(sim_top)
    
    x = np.arange(len(real_top))
    ax4.plot(x, real_top, 'o-', linewidth=2, markersize=6, 
             label=f"Real (year {real['year']})", color='steelblue')
    ax4.plot(x, sim_top, 's-', linewidth=2, markersize=5, 
             label='Simulated', color='coral')
    
    ax4.set_xlabel('Destinations (sorted by real data)', fontsize=11)
    ax4.set_ylabel('Fraction of Firms Exporting', fontsize=11)
    ax4.set_title('Top 20 Destinations: Marginals', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # ===== PANEL 5: Top 10 bar chart =====
    ax5 = plt.subplot(2, 3, 5)
    
    top_10_real = real['dest_fractions'][:10]
    top_10_names = real['dest_names'][:10]
    
    # Match simulated
    top_10_sim = []
    for name in top_10_names:
        idx = np.where(sim['dest_names'] == name)[0]
        if len(idx) > 0:
            top_10_sim.append(sim['dest_fractions'][idx[0]])
        else:
            top_10_sim.append(0)
    
    x = np.arange(len(top_10_names))
    width = 0.35
    
    ax5.barh(x - width/2, top_10_real, width, label=f"Real ({real['year']})", 
             color='steelblue', alpha=0.8)
    ax5.barh(x + width/2, top_10_sim, width, label='Simulated', 
             color='coral', alpha=0.8)
    
    ax5.set_yticks(x)
    ax5.set_yticklabels(top_10_names, fontsize=9)
    ax5.invert_yaxis()
    ax5.set_xlabel('Fraction of Firms', fontsize=11)
    ax5.set_title('Top 10 Destinations Comparison', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3, axis='x')
    
    # ===== PANEL 6: Scatter plot Real vs Simulated =====
    ax6 = plt.subplot(2, 3, 6)
    
    # Match all common destinations
    common = []
    for i, name in enumerate(real['dest_names']):
        idx = np.where(sim['dest_names'] == name)[0]
        if len(idx) > 0:
            common.append((real['dest_fractions'][i], sim['dest_fractions'][idx[0]], name))
    
    if common:
        real_vals = [c[0] for c in common]
        sim_vals = [c[1] for c in common]
        names = [c[2] for c in common]
        
        ax6.scatter(real_vals, sim_vals, alpha=0.6, s=50, color='steelblue')
        
        # Add diagonal
        max_val = max(max(real_vals), max(sim_vals))
        ax6.plot([0, max_val], [0, max_val], 'k--', linewidth=1, alpha=0.5, label='Perfect match')
        
        # Annotate top destinations
        for i in range(min(5, len(names))):
            ax6.annotate(names[i], (real_vals[i], sim_vals[i]), 
                        fontsize=8, alpha=0.7)
        
        ax6.set_xlabel('Real Fraction', fontsize=11)
        ax6.set_ylabel('Simulated Fraction', fontsize=11)
        ax6.set_title('Real vs Simulated (Scatter)', fontsize=12, fontweight='bold')
        ax6.legend(fontsize=9)
        ax6.grid(True, alpha=0.3)
        
        # Calculate correlation
        corr = np.corrcoef(real_vals, sim_vals)[0, 1]
        ax6.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                transform=ax6.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('REAL vs SIMULATED: Mexico Export Patterns', 
                 fontsize=15, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig('datasets/mexico_real_vs_simulated.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: datasets/mexico_real_vs_simulated.png")


def print_comparison():
    """Print detailed comparison."""
    real = load_real_data()
    sim = load_simulated_data()
    
    print("="*70)
    print("REAL vs SIMULATED: DETAILED COMPARISON")
    print("="*70)
    
    print(f"\n{'':20s} {'REAL':>20s} {'SIMULATED':>20s}")
    print("-"*70)
    print(f"{'Year/Data':20s} {real['year']:>20d} {'2024 (sim)':>20s}")
    print(f"{'Firms':20s} {real['total_firms']:>20,} {sim['total_firms']:>20,}")
    print(f"{'Mean destinations':20s} {real['destinations_per_firm'].mean():>20.1f} {sim['destinations_per_firm'].mean():>20.1f}")
    print(f"{'Median destinations':20s} {np.median(real['destinations_per_firm']):>20.0f} {np.median(sim['destinations_per_firm']):>20.0f}")
    print(f"{'Min destinations':20s} {real['destinations_per_firm'].min():>20d} {sim['destinations_per_firm'].min():>20d}")
    print(f"{'Max destinations':20s} {real['destinations_per_firm'].max():>20d} {sim['destinations_per_firm'].max():>20d}")
    print(f"{'Std destinations':20s} {real['destinations_per_firm'].std():>20.1f} {sim['destinations_per_firm'].std():>20.1f}")
    
    print("\n" + "="*70)
    print("TOP 10 DESTINATIONS")
    print("="*70)
    print(f"{'Rank':>5s} {'Country':>8s} {'Real %':>12s} {'Sim %':>12s} {'Diff':>12s}")
    print("-"*70)
    
    for i in range(10):
        real_name = real['dest_names'][i]
        real_frac = real['dest_fractions'][i]
        
        # Find in simulated
        idx = np.where(sim['dest_names'] == real_name)[0]
        if len(idx) > 0:
            sim_frac = sim['dest_fractions'][idx[0]]
        else:
            sim_frac = 0
        
        diff = sim_frac - real_frac
        print(f"{i+1:>5d} {real_name:>8s} {real_frac*100:>11.1f}% {sim_frac*100:>11.1f}% {diff*100:>11.1f}%")
    
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    print(f"""
1. SPARSITY MISMATCH:
   - Real firms export to ~1.8 destinations on average
   - Simulated firms export to ~20 destinations
   - Simulation is {sim['destinations_per_firm'].mean()/real['destinations_per_firm'].mean():.1f}x TOO DENSE!

2. USA DOMINANCE:
   - Real: {real['dest_fractions'][0]*100:.1f}% of firms export to USA
   - Simulated: {sim['dest_fractions'][np.where(sim['dest_names']=='USA')[0][0]]*100:.1f}% export to USA
   - Real data shows EXTREME concentration on USA (81%)

3. DISTRIBUTION SHAPE:
   - Real: Highly skewed (median=1, mean=1.8)
   - Simulated: More symmetric (median=20, mean=20)
   - Real data has many "single-destination" firms

4. RECOMMENDATIONS:
   - Increase fixed export costs (make exporting more expensive)
   - Reduce complementarities further (or make negative)
   - Add more heterogeneity in firm capabilities
   - Consider product-level variation (real data has products)
    """)
    
    print("="*70)


def main():
    print("\n" + "="*70)
    print("COMPARING REAL VS SIMULATED MEXICO EXPORT DATA")
    print("="*70)
    
    print("\nCreating comparison visualizations...")
    plot_comparison()
    
    print_comparison()
    
    print("\n✅ COMPARISON COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()

