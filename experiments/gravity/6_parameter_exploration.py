"""
Explore different parameter combinations to match real Mexico data.
"""
import numpy as np
import pandas as pd
from bundlechoice.core import BundleChoice
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def load_data():
    features = pd.read_csv('datasets/country_features.csv', index_col=0)
    distances = pd.read_csv('datasets/distances.csv', index_col=0)
    return features, distances


def prepare_data(features, distances, home_countries, error_std=1.5, 
                 agent_std=0.5, fixed_cost=0.0):
    """Prepare data with configurable parameters."""
    num_firms = len(home_countries)
    num_countries = len(features)
    
    # Modular features
    gdp = features['gdp_billions'].fillna(0).values
    pop = features['population_millions'].fillna(0).values
    
    gdp_std = (gdp - gdp.mean()) / (gdp.std() + 1e-8)
    pop_std = (pop - pop.mean()) / (pop.std() + 1e-8)
    
    item_modular = np.column_stack([gdp_std, pop_std])
    
    # Quadratic: proximity
    log_dist = np.log(distances.values + 1)
    proximity = log_dist.max() - log_dist
    proximity = proximity / proximity.max()
    np.fill_diagonal(proximity, 0)
    
    item_quadratic = proximity[:, :, np.newaxis]
    
    # Agent modular (heterogeneity) + fixed cost
    agent_modular = np.random.normal(0, agent_std, (num_firms, num_countries, 1))
    
    # Add fixed cost to agent_modular (same across all countries)
    if fixed_cost != 0:
        agent_modular = agent_modular + fixed_cost
    
    # Errors
    errors = np.random.normal(0, error_std, (num_firms, num_countries))
    
    # Constraint masks
    constraint_mask = []
    for i in range(num_firms):
        home = home_countries[i]
        mask = np.ones(num_countries, dtype=bool)
        mask[home] = False
        constraint_mask.append(mask)
    
    input_data = {
        "item_data": {
            "modular": item_modular,
            "quadratic": item_quadratic
        },
        "agent_data": {
            "modular": agent_modular,
            "constraint_mask": np.array(constraint_mask)
        },
        "errors": errors,
    }
    
    return input_data


def assign_firms(features, num_firms=500, seed=42):
    """Assign firms to home countries."""
    np.random.seed(seed)
    gdp = features['gdp_billions'].fillna(0).values
    weights = np.power(gdp, 0.8)
    weights = weights / weights.sum()
    home_countries = np.random.choice(len(features), size=num_firms, p=weights)
    return home_countries


def simulate_with_params(features, distances, theta, error_std=1.5, 
                        agent_std=0.5, fixed_cost=0.0, num_firms=500, seed=42):
    """Run simulation with given parameters."""
    np.random.seed(seed)
    
    home_countries = assign_firms(features, num_firms, seed)
    input_data = prepare_data(features, distances, home_countries, 
                              error_std, agent_std, fixed_cost)
    
    cfg = {
        "dimensions": {
            "num_agents": num_firms,
            "num_items": len(features),
            "num_features": 4,
            "num_simulations": 1
        },
        "subproblem": {
            "name": "QuadSupermodularNetwork",
            "settings": {}
        }
    }
    
    bc = BundleChoice()
    bc.load_config(cfg)
    bc.data.load_and_scatter(input_data)
    bc.oracles.build_from_data()
    
    bundles = bc.subproblems.init_and_solve(theta)
    
    if bc.rank == 0:
        # Extract Mexico
        country_names = list(features.index)
        mex_idx = country_names.index('MEX')
        mex_firms = np.where(home_countries == mex_idx)[0]
        
        if len(mex_firms) > 0:
            mex_bundles = bundles[mex_firms]
            mex_export = np.delete(mex_bundles, mex_idx, axis=1)
            
            destinations_per_firm = mex_export.sum(axis=1)
            
            return {
                'mean': destinations_per_firm.mean(),
                'median': np.median(destinations_per_firm),
                'min': destinations_per_firm.min(),
                'max': destinations_per_firm.max(),
                'std': destinations_per_firm.std(),
                'destinations_per_firm': destinations_per_firm,
                'bundles': mex_export,
                'num_firms': len(mex_firms)
            }
    
    return None


def explore_parameters():
    """Systematically explore parameter space."""
    print("="*70)
    print("PARAMETER EXPLORATION")
    print("="*70)
    
    # Load data
    features, distances = load_data()
    
    # Real data target
    real_mean = 1.8
    real_median = 1.0
    
    print(f"\n🎯 TARGET (Real Mexico 2000):")
    print(f"   Mean destinations: {real_mean}")
    print(f"   Median destinations: {real_median}")
    
    # Parameter grid
    experiments = []
    
    # Base parameters
    base_theta = np.array([0.3, 2.0, 0.5, 0.0])
    
    print(f"\n📊 Running experiments...")
    
    # Experiment 1: Vary error std (increase randomness)
    print(f"\n--- Experiment 1: Varying error std ---")
    for error_std in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        result = simulate_with_params(features, distances, base_theta, 
                                     error_std=error_std, num_firms=500, seed=42)
        if result:
            experiments.append({
                'name': f'error_std={error_std}',
                'error_std': error_std,
                'agent_std': 0.5,
                'fixed_cost': 0.0,
                'theta_agent': 0.3,
                'theta_gdp': 2.0,
                'theta_pop': 0.5,
                'theta_prox': 0.0,
                **result
            })
            print(f"  error_std={error_std:4.1f} → mean={result['mean']:5.1f}, median={result['median']:4.0f}")
    
    # Experiment 2: Vary agent heterogeneity
    print(f"\n--- Experiment 2: Varying agent heterogeneity ---")
    for agent_std in [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]:
        result = simulate_with_params(features, distances, base_theta, 
                                     agent_std=agent_std, num_firms=500, seed=42)
        if result:
            experiments.append({
                'name': f'agent_std={agent_std}',
                'error_std': 1.5,
                'agent_std': agent_std,
                'fixed_cost': 0.0,
                'theta_agent': 0.3,
                'theta_gdp': 2.0,
                'theta_pop': 0.5,
                'theta_prox': 0.0,
                **result
            })
            print(f"  agent_std={agent_std:4.1f} → mean={result['mean']:5.1f}, median={result['median']:4.0f}")
    
    # Experiment 3: Add fixed costs
    print(f"\n--- Experiment 3: Adding fixed export costs ---")
    for fixed_cost in [0, -1, -2, -5, -10, -20]:
        result = simulate_with_params(features, distances, base_theta, 
                                     fixed_cost=fixed_cost, num_firms=500, seed=42)
        if result:
            experiments.append({
                'name': f'fixed_cost={fixed_cost}',
                'error_std': 1.5,
                'agent_std': 0.5,
                'fixed_cost': fixed_cost,
                'theta_agent': 0.3,
                'theta_gdp': 2.0,
                'theta_pop': 0.5,
                'theta_prox': 0.0,
                **result
            })
            print(f"  fixed_cost={fixed_cost:4.0f} → mean={result['mean']:5.1f}, median={result['median']:4.0f}")
    
    # Experiment 4: Reduce GDP coefficient (less gravity)
    print(f"\n--- Experiment 4: Varying GDP coefficient ---")
    for theta_gdp in [0.5, 1.0, 1.5, 2.0, 3.0]:
        theta = np.array([0.3, theta_gdp, 0.5, 0.0])
        result = simulate_with_params(features, distances, theta, num_firms=500, seed=42)
        if result:
            experiments.append({
                'name': f'theta_gdp={theta_gdp}',
                'error_std': 1.5,
                'agent_std': 0.5,
                'fixed_cost': 0.0,
                'theta_agent': 0.3,
                'theta_gdp': theta_gdp,
                'theta_pop': 0.5,
                'theta_prox': 0.0,
                **result
            })
            print(f"  theta_gdp={theta_gdp:4.1f} → mean={result['mean']:5.1f}, median={result['median']:4.0f}")
    
    # Experiment 5: Fine-tune fixed costs (zooming in)
    print(f"\n--- Experiment 5: Fine-tuning fixed costs ---")
    for fixed_cost in [-15, -18, -22, -25, -30]:
        result = simulate_with_params(features, distances, base_theta, 
                                     fixed_cost=fixed_cost, num_firms=500, seed=42)
        if result:
            experiments.append({
                'name': f'fixed_cost={fixed_cost}',
                'error_std': 1.5,
                'agent_std': 0.5,
                'fixed_cost': fixed_cost,
                'theta_agent': 0.3,
                'theta_gdp': 2.0,
                'theta_pop': 0.5,
                'theta_prox': 0.0,
                **result
            })
            print(f"  fixed_cost={fixed_cost:4.0f} → mean={result['mean']:5.1f}, median={result['median']:4.0f}")
    
    # Experiment 6: Combinations
    print(f"\n--- Experiment 6: Best combinations ---")
    combos = [
        {'error_std': 2.0, 'agent_std': 2.0, 'fixed_cost': -18, 'theta': [0.5, 1.5, 0.3, 0.0]},
        {'error_std': 3.0, 'agent_std': 3.0, 'fixed_cost': -20, 'theta': [0.5, 1.0, 0.2, 0.0]},
        {'error_std': 1.5, 'agent_std': 1.0, 'fixed_cost': -22, 'theta': [0.3, 2.0, 0.4, 0.0]},
    ]
    
    for i, combo in enumerate(combos):
        theta = np.array(combo['theta'])
        result = simulate_with_params(features, distances, theta, 
                                     error_std=combo['error_std'],
                                     agent_std=combo['agent_std'],
                                     fixed_cost=combo['fixed_cost'],
                                     num_firms=500, seed=42)
        if result:
            experiments.append({
                'name': f'combo_{i+1}',
                'error_std': combo['error_std'],
                'agent_std': combo['agent_std'],
                'fixed_cost': combo['fixed_cost'],
                'theta_agent': theta[0],
                'theta_gdp': theta[1],
                'theta_pop': theta[2],
                'theta_prox': theta[3],
                **result
            })
            print(f"  combo_{i+1} → mean={result['mean']:5.1f}, median={result['median']:4.0f}")
    
    return pd.DataFrame(experiments)


def visualize_results(df):
    """Visualize parameter exploration results."""
    fig = plt.figure(figsize=(16, 10))
    
    # Real target
    real_mean = 1.8
    real_median = 1.0
    
    # Calculate distance from target
    df['error_mean'] = np.abs(df['mean'] - real_mean)
    df['error_median'] = np.abs(df['median'] - real_median)
    df['total_error'] = df['error_mean'] + df['error_median']
    
    # Sort by total error
    df_sorted = df.sort_values('total_error')
    
    # Panel 1: Mean destinations
    ax1 = plt.subplot(2, 3, 1)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(df)))
    ax1.scatter(range(len(df)), df_sorted['mean'], c=colors, s=50, alpha=0.7)
    ax1.axhline(real_mean, color='red', linestyle='--', linewidth=2, label='Real (1.8)')
    ax1.set_xlabel('Experiment (sorted by error)', fontsize=10)
    ax1.set_ylabel('Mean Destinations', fontsize=10)
    ax1.set_title('Mean Destinations per Firm', fontsize=11, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Median destinations
    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(range(len(df)), df_sorted['median'], c=colors, s=50, alpha=0.7)
    ax2.axhline(real_median, color='red', linestyle='--', linewidth=2, label='Real (1.0)')
    ax2.set_xlabel('Experiment (sorted by error)', fontsize=10)
    ax2.set_ylabel('Median Destinations', fontsize=10)
    ax2.set_title('Median Destinations per Firm', fontsize=11, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Best parameters
    ax3 = plt.subplot(2, 3, 3)
    ax3.axis('off')
    
    top5 = df_sorted.head(5)
    text = "TOP 5 PARAMETER SETS\n"
    text += "="*45 + "\n\n"
    
    for i, (idx, row) in enumerate(top5.iterrows()):
        text += f"#{i+1}: {row['name']}\n"
        text += f"  Mean: {row['mean']:.1f} (Δ={row['error_mean']:.1f})\n"
        text += f"  Median: {row['median']:.0f} (Δ={row['error_median']:.1f})\n"
        text += f"  Params: error_std={row['error_std']:.1f}\n"
        text += f"          agent_std={row['agent_std']:.1f}\n"
        text += f"          fixed_cost={row['fixed_cost']:.1f}\n"
        text += f"          θ_GDP={row['theta_gdp']:.1f}\n\n"
    
    ax3.text(0.05, 0.95, text, transform=ax3.transAxes,
             fontsize=8, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    # Panel 4: Error vs fixed cost
    ax4 = plt.subplot(2, 3, 4)
    fixed_cost_data = df[df['agent_std'] == 0.5]
    ax4.scatter(fixed_cost_data['fixed_cost'], fixed_cost_data['mean'], 
               s=80, alpha=0.7, c=fixed_cost_data['total_error'], cmap='RdYlGn_r')
    ax4.axhline(real_mean, color='red', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Fixed Cost', fontsize=10)
    ax4.set_ylabel('Mean Destinations', fontsize=10)
    ax4.set_title('Effect of Fixed Export Cost', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Panel 5: Error vs error_std
    ax5 = plt.subplot(2, 3, 5)
    error_std_data = df[df['fixed_cost'] == 0.0]
    ax5.scatter(error_std_data['error_std'], error_std_data['mean'], 
               s=80, alpha=0.7, c=error_std_data['total_error'], cmap='RdYlGn_r')
    ax5.axhline(real_mean, color='red', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Error Std Dev', fontsize=10)
    ax5.set_ylabel('Mean Destinations', fontsize=10)
    ax5.set_title('Effect of Error Variance', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Panel 6: Parameter heatmap
    ax6 = plt.subplot(2, 3, 6)
    
    # Create parameter importance scores
    param_effects = {
        'Fixed Cost': df.groupby('fixed_cost')['total_error'].mean().std() if len(df['fixed_cost'].unique()) > 1 else 0,
        'Error Std': df.groupby('error_std')['total_error'].mean().std() if len(df['error_std'].unique()) > 1 else 0,
        'Agent Std': df.groupby('agent_std')['total_error'].mean().std() if len(df['agent_std'].unique()) > 1 else 0,
        'θ GDP': df.groupby('theta_gdp')['total_error'].mean().std() if len(df['theta_gdp'].unique()) > 1 else 0,
    }
    
    params = list(param_effects.keys())
    values = list(param_effects.values())
    
    colors_bar = plt.cm.plasma(np.linspace(0.3, 0.9, len(params)))
    ax6.barh(params, values, color=colors_bar, alpha=0.8)
    ax6.set_xlabel('Impact on Error (std)', fontsize=10)
    ax6.set_title('Parameter Importance', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Parameter Exploration: Matching Real Mexico Data', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig('datasets/parameter_exploration.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: datasets/parameter_exploration.png")
    
    return df_sorted


def main():
    df = explore_parameters()
    df_sorted = visualize_results(df)
    
    # Save results
    df_sorted.to_csv('datasets/parameter_exploration.csv', index=False)
    print(f"✓ Saved: datasets/parameter_exploration.csv")
    
    print("\n" + "="*70)
    print("BEST PARAMETER SET")
    print("="*70)
    best = df_sorted.iloc[0]
    print(f"\nName: {best['name']}")
    print(f"Mean destinations: {best['mean']:.1f} (real: 1.8, error: {best['error_mean']:.1f})")
    print(f"Median destinations: {best['median']:.0f} (real: 1.0, error: {best['error_median']:.1f})")
    print(f"\nParameters:")
    print(f"  error_std = {best['error_std']}")
    print(f"  agent_std = {best['agent_std']}")
    print(f"  fixed_cost = {best['fixed_cost']}")
    print(f"  theta = [{best['theta_agent']}, {best['theta_gdp']}, {best['theta_pop']}, 0.0]")
    print("="*70)


if __name__ == '__main__':
    main()

