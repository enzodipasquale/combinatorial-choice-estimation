"""
Simple simulation without full MPI - just generate choices for analysis.
"""
import numpy as np
import pandas as pd
import argparse

def load_data():
    """Load gravity data."""
    features = pd.read_csv('datasets/country_features.csv', index_col=0)
    distances = pd.read_csv('datasets/distances.csv', index_col=0)
    
    # Create simple language/region matrices (10% common language, 15% common region)
    n = len(features)
    lang = pd.DataFrame(np.random.binomial(1, 0.1, (n, n)), 
                        index=features.index, columns=features.index)
    np.fill_diagonal(lang.values, 1)
    
    region = pd.DataFrame(np.random.binomial(1, 0.15, (n, n)),
                          index=features.index, columns=features.index)
    np.fill_diagonal(region.values, 1)
    
    return features, distances, lang, region


def assign_firms(features, num_firms, seed=42):
    """Assign firms to home countries proportionally to real firm counts."""
    np.random.seed(seed)
    
    # Use GDP^0.8 for firm distribution
    gdp = features['gdp_billions'].fillna(0).values
    weights = np.power(gdp, 0.8)
    weights = weights / weights.sum()
    
    home_countries = np.random.choice(len(features), size=num_firms, p=weights)
    
    print(f"\nFirm distribution (top 10):")
    unique, counts = np.unique(home_countries, return_counts=True)
    sorted_idx = np.argsort(counts)[::-1][:10]
    for idx in sorted_idx:
        country_idx = unique[idx]
        print(f"  {features.index[country_idx]}: {counts[idx]} firms ({counts[idx]/num_firms*100:.1f}%)")
    
    return home_countries


def compute_utilities(features, distances, lang, region, home_countries, theta):
    """
    Compute utility for each firm-destination pair.
    
    theta = [firm_heterog, gdp, pop, gdppc, trade, proximity, language, region]
    """
    num_firms = len(home_countries)
    num_countries = len(features)
    
    # Extract features
    gdp = features['gdp_billions'].fillna(features['gdp_billions'].mean()).values
    pop = features['population_millions'].fillna(features['population_millions'].mean()).values
    gdppc = features['gdp_per_capita'].fillna(features['gdp_per_capita'].mean()).values
    trade = features['trade_openness_pct'].fillna(features['trade_openness_pct'].mean()).values
    
    # Normalize
    gdp = (gdp - gdp.mean()) / gdp.std()
    pop = (pop - pop.mean()) / pop.std()
    gdppc = (gdppc - gdppc.mean()) / gdppc.std()
    trade = (trade - trade.mean()) / trade.std()
    
    # Proximity (inverse log distance)
    log_dist = np.log(distances.values + 1)
    proximity = log_dist.max() - log_dist
    proximity = proximity / proximity.max()
    
    # Initialize utilities
    utilities = np.zeros((num_firms, num_countries))
    
    # Modular: firm heterogeneity + home boost
    firm_effects = np.random.normal(0, 0.3, (num_firms, num_countries))
    for i, home_idx in enumerate(home_countries):
        firm_effects[i, home_idx] += 5.0  # Strong home boost
    utilities += theta[0] * firm_effects
    
    # Modular: country characteristics
    utilities += theta[1] * gdp[None, :]
    utilities += theta[2] * pop[None, :]
    utilities += theta[3] * gdppc[None, :]
    utilities += theta[4] * trade[None, :]
    
    # Add FIXED EXPORT COST (shift utilities down to create sparsity!)
    utilities -= 3.0  # Each export has a base cost
    
    # Add errors
    utilities += np.random.normal(0, 2.0, (num_firms, num_countries))
    
    return utilities, proximity, lang.values, region.values


def greedy_solve(utilities, proximity, lang, region, theta, max_destinations=30):
    """
    Greedy with realistic heterogeneous stopping.
    Each firm stops when marginal utility becomes negative (not worth exporting).
    """
    num_firms, num_countries = utilities.shape
    bundles = np.zeros((num_firms, num_countries), dtype=bool)
    
    print(f"\nSolving via greedy (heterogeneous stopping, max {max_destinations})...")
    
    for i in range(num_firms):
        if i % 1000 == 0:
            print(f"  Firm {i}/{num_firms}...")
        
        base_utility = utilities[i, :].copy()
        selected = []
        
        # Greedy: add destinations while marginal utility is positive
        for step in range(min(max_destinations, num_countries)):
            current_utility = base_utility.copy()
            
            # Add complementarity bonuses from existing selections
            # Use REDUCED complementarities to avoid "all export to all"
            if selected:
                for j in selected:
                    current_utility += theta[5] * 0.01 * proximity[j, :]  # Reduced!
                    current_utility += theta[6] * 0.01 * lang[j, :]      # Reduced!
                    current_utility += theta[7] * 0.01 * region[j, :]    # Reduced!
            
            # Set already selected to -inf
            current_utility[selected] = -np.inf
            
            # Find best unselected
            best = np.argmax(current_utility)
            marginal_utility = current_utility[best]
            
            # Stop if marginal utility is negative (not worth it!)
            # This creates heterogeneity: firms with different utilities stop at different points
            if marginal_utility < 0:
                break
            
            selected.append(best)
            bundles[i, best] = True
    
    return bundles


def analyze_flows(bundles, home_countries, country_names):
    """Analyze trade flows."""
    print(f"\n{'='*60}")
    print("EXPORT FLOW ANALYSIS")
    print(f"{'='*60}")
    
    num_firms, num_countries = bundles.shape
    
    # Aggregate statistics
    destinations_per_firm = bundles.sum(axis=1)
    print(f"\nFirm-level statistics:")
    print(f"  Mean destinations: {destinations_per_firm.mean():.1f}")
    print(f"  Median: {np.median(destinations_per_firm):.0f}")
    print(f"  Range: {destinations_per_firm.min():.0f} - {destinations_per_firm.max():.0f}")
    
    # Home market
    home_exports = np.array([bundles[i, home_countries[i]] for i in range(num_firms)])
    print(f"  Exporting to home country: {home_exports.mean()*100:.1f}%")
    
    # Destination popularity
    print(f"\nTop 10 export destinations:")
    dest_counts = bundles.sum(axis=0)
    top_idx = np.argsort(dest_counts)[::-1][:10]
    for idx in top_idx:
        print(f"  {country_names[idx]}: {dest_counts[idx]} exporters ({dest_counts[idx]/num_firms*100:.1f}%)")
    
    # Bilateral flows
    print(f"\nTop 10 bilateral export relationships:")
    bilateral = np.zeros((num_countries, num_countries))
    for i in range(num_firms):
        home = home_countries[i]
        for dest in np.where(bundles[i])[0]:
            bilateral[home, dest] += 1
    
    # Get top pairs
    flows = []
    for i in range(num_countries):
        for j in range(num_countries):
            if i != j and bilateral[i,j] > 0:
                flows.append((country_names[i], country_names[j], bilateral[i,j]))
    flows.sort(key=lambda x: x[2], reverse=True)
    
    for origin, dest, count in flows[:10]:
        print(f"  {origin} → {dest}: {count:.0f} firms")
    
    return bundles, bilateral


def main():
    parser = argparse.ArgumentParser(description='Simulate and analyze export flows')
    parser.add_argument('--num_firms', type=int, default=10000, help='Number of firms to simulate')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--analyze', action='store_true', help='Run flow analysis after simulation')
    args = parser.parse_args()
    
    print("="*60)
    print(f"SIMPLE GRAVITY SIMULATION: {args.num_firms} firms")
    print("="*60)
    
    # Load
    features, distances, lang, region = load_data()
    print(f"\n✓ Loaded {len(features)} countries")
    
    # Assign firms
    home_countries = assign_firms(features, args.num_firms, args.seed)
    
    # Compute utilities
    print(f"\nComputing utilities...")
    # Calibrated params
    theta = np.array([0.5, 0.85, 0.5, 0.3, 0.2, 1.1, 0.35, 0.65])
    utilities, proximity, lang_mat, region_mat = compute_utilities(
        features, distances, lang, region, home_countries, theta
    )
    print(f"  ✓ Utility matrix: {utilities.shape}")
    
    # Solve
    bundles = greedy_solve(utilities, proximity, lang_mat, region_mat, theta)
    
    # Analyze
    bundles, bilateral = analyze_flows(bundles, home_countries, features.index.tolist())
    
    # Save
    np.savez('datasets/simulated_choices.npz',
             bundles=bundles,
             home_countries=home_countries,
             bilateral_flows=bilateral,
             country_names=features.index.tolist(),
             theta_true=theta,
             num_firms=args.num_firms,
             seed=args.seed)
    
    df = pd.DataFrame(bundles, columns=features.index.tolist())
    df.insert(0, 'home_country', [features.index[i] for i in home_countries])
    df.to_csv('datasets/simulated_choices.csv')
    
    print(f"\n✓ Saved to datasets/simulated_choices.npz and .csv")
    
    # Optional: run detailed flow analysis
    if args.analyze:
        print(f"\n{'='*60}")
        print("RUNNING DETAILED FLOW ANALYSIS")
        print(f"{'='*60}")
        import subprocess
        subprocess.run(['python', 'analyze_flows.py'])


if __name__ == '__main__':
    main()
