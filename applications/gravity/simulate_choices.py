"""
Simulate firm export destination choices using gravity model data.
Uses quadratic supermodular subproblem from bundlechoice.
"""
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import sys
import traceback

# Install custom exception handler to catch errors before MPI aborts
def exception_handler(exc_type, exc_value, exc_traceback):
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"PYTHON EXCEPTION CAUGHT:", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)
    sys.exit(1)

sys.excepthook = exception_handler

from bundlechoice.core import BundleChoice


def load_gravity_data():
    """Load generated gravity model data."""
    print("Loading gravity data...")
    
    # Load country features
    features = pd.read_csv('datasets/country_features.csv', index_col=0)
    distances = pd.read_csv('datasets/distances.csv', index_col=0)
    
    # Load pairwise features
    pairwise_files = ['common_language.csv', 'common_region.csv']
    pairwise_data = {}
    for filename in pairwise_files:
        filepath = f'datasets/{filename}'
        if Path(filepath).exists():
            pairwise_data[filename.replace('.csv', '')] = pd.read_csv(filepath, index_col=0)
    
    print(f"  Loaded {len(features)} countries")
    print(f"  Loaded {len(features.columns)} country features")
    print(f"  Loaded {len(pairwise_data) + 1} pairwise features")
    
    return features, distances, pairwise_data


def select_modular_features(features, feature_names=None):
    """Select which modular features to use."""
    if feature_names is None:
        # Default: use key economic indicators
        feature_names = [
            'gdp_billions',
            'population_millions', 
            'gdp_per_capita',
            'trade_openness_pct',
        ]
    
    # Filter to available features
    available = [f for f in feature_names if f in features.columns]
    
    if not available:
        raise ValueError(f"None of {feature_names} found in data. Available: {list(features.columns)}")
    
    return features[available].values


def assign_firm_home_countries(features, num_firms, seed=None, use_calibration=True):
    """
    Assign each firm to a home country based on realistic distribution.
    
    Uses GDP^0.8 scaling (empirically realistic for firm counts).
    
    Returns:
        home_countries: Array of country indices for each firm
        country_names: List of country names
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Try to load calibrated weights
    if use_calibration and Path('datasets/calibration_data.npz').exists():
        print("  Using calibrated firm distribution")
        calib = np.load('datasets/calibration_data.npz', allow_pickle=True)
        weights = calib['firm_weights']
    else:
        # Use GDP^0.8 (firm count scales sub-linearly with GDP)
        # This is empirically realistic: larger economies have proportionally fewer firms
        if 'gdp_billions' in features.columns:
            weights = features['gdp_billions'].values ** 0.8
        elif 'population_millions' in features.columns:
            weights = features['population_millions'].values
        else:
            weights = np.ones(len(features))
        
        # Ensure non-negative and normalize
        weights = np.maximum(weights, 0)
        weights = weights / weights.sum()
    
    # Sample firm home countries
    num_countries = len(features)
    
    # For small samples, ensure each country gets at least 1 firm
    # (prevents solver issues when a country has 0 firms)
    if num_firms < num_countries * 20:  # If fewer than 20 firms per country on average
        print(f"  ! Small sample ({num_firms} firms): ensuring each country gets ≥1 firm")
        
        # Assign 1 firm to each country first
        home_countries = list(range(num_countries))
        
        # Then sample the remaining firms proportionally
        remaining = num_firms - num_countries
        if remaining > 0:
            additional = np.random.choice(num_countries, size=remaining, p=weights)
            home_countries.extend(additional.tolist())
        
        # Convert to array and shuffle
        home_countries = np.array(home_countries)
        np.random.shuffle(home_countries)
    else:
        # Normal sampling for large samples
        home_countries = np.random.choice(num_countries, size=num_firms, p=weights)
    
    # Count firms per country
    unique, counts = np.unique(home_countries, return_counts=True)
    
    print(f"\nFirm distribution across home countries:")
    for idx, count in zip(unique, counts):
        print(f"  {features.index[idx]}: {count} firms ({count/num_firms*100:.1f}%)")
    
    # Check for missing countries
    missing = set(range(num_countries)) - set(unique)
    if missing:
        print(f"  ⚠ Warning: {len(missing)} countries with 0 firms: {[features.index[i] for i in missing]}")
    
    return home_countries, features.index.tolist()


def prepare_bundlechoice_data(features, distances, pairwise_data, 
                              num_firms, theta_true, sigma=1.0,
                              modular_features=None, seed=None):
    """
    Convert gravity data to bundlechoice format.
    
    Args:
        features: Country-level features DataFrame
        distances: Distance matrix DataFrame  
        pairwise_data: Dict of pairwise feature DataFrames
        num_firms: Number of firms to simulate
        theta_true: True parameter vector
        sigma: Error standard deviation
        modular_features: List of modular feature names to use
        seed: Random seed for firm assignment
    """
    num_countries = len(features)
    
    # Assign firms to home countries
    home_countries, country_names = assign_firm_home_countries(features, num_firms, seed)
    
    print(f"\nPreparing data for {num_firms} firms in {num_countries} countries...")
    
    # Item (country) modular features
    item_modular = select_modular_features(features, modular_features)
    num_item_modular = item_modular.shape[1]
    print(f"  Item modular features: {num_item_modular}")
    
    # Handle NaN values (some countries missing data)
    # Fill with mean (common imputation for missing economic data)
    for j in range(item_modular.shape[1]):
        col = item_modular[:, j]
        if np.isnan(col).any():
            mean_val = np.nanmean(col)
            item_modular[np.isnan(col), j] = mean_val
            print(f"    ! Imputed {np.isnan(col).sum()} NaN values in feature {j} with mean {mean_val:.2f}")
    
    # Normalize modular features (important for numerical stability)
    item_modular = (item_modular - item_modular.mean(axis=0)) / (item_modular.std(axis=0) + 1e-8)
    
    # Item (country-pair) quadratic features
    # Stack: [distances, common_language, common_region, ...]
    quad_features = [distances.values]
    for name, df in pairwise_data.items():
        quad_features.append(df.values)
    
    item_quadratic = np.stack(quad_features, axis=2)
    num_item_quadratic = item_quadratic.shape[2]
    
    # Normalize distance (log transform often used in gravity models)
    # For supermodular: need non-negative values, use inverse (closer = higher value)
    # Transform: max_dist - log(dist) so that closer countries have higher values
    log_dist = np.log(item_quadratic[:, :, 0] + 1)
    item_quadratic[:, :, 0] = log_dist.max() - log_dist  # Invert so close = high value
    item_quadratic[:, :, 0] = item_quadratic[:, :, 0] / item_quadratic[:, :, 0].max()  # Scale to [0,1]
    
    # Ensure diagonal is zero and all values non-negative for supermodularity
    for k in range(num_item_quadratic):
        np.fill_diagonal(item_quadratic[:, :, k], 0)
        # Ensure non-negative (should already be, but safety check)
        item_quadratic[:, :, k] = np.maximum(item_quadratic[:, :, k], 0)
    
    print(f"  Item quadratic features: {num_item_quadratic} (proximity [inverse log dist], {', '.join(pairwise_data.keys())})")
    
    # Agent (firm) modular features - firm-destination specific
    # Combine home indicator with firm heterogeneity into single feature
    # Home indicator: 1 for home country, 0 otherwise
    # Add firm-specific random effects
    agent_modular = np.random.normal(0, 0.5, (num_firms, num_countries, 1))
    
    # Add strong boost for home country
    for i, home_idx in enumerate(home_countries):
        agent_modular[i, home_idx, 0] += 3.0  # Home market boost (will be scaled by theta)
    
    num_agent_modular = 1
    print(f"  Agent modular features: {num_agent_modular} (firm heterogeneity + home boost)")
    
    # Generate errors
    errors = sigma * np.random.normal(0, 1, size=(num_firms, num_countries))
    
    print(f"  Error std: {sigma}")
    
    # Prepare bundlechoice input
    input_data = {
        "item_data": {
            "modular": item_modular,
            "quadratic": item_quadratic
        },
        "agent_data": {
            "modular": agent_modular,
        },
        "errors": errors,
    }
    
    num_features = num_agent_modular + num_item_modular + num_item_quadratic
    
    return input_data, num_features, home_countries, country_names


def generate_choices(input_data, num_features, theta_true, home_countries, country_names, num_simuls=1):
    """
    Generate firm export choices using quadratic supermodular solver.
    
    Args:
        input_data: BundleChoice formatted data
        num_features: Total number of features
        theta_true: True parameter vector
        home_countries: Array of home country indices for each firm
        country_names: List of country names
        num_simuls: Number of simulation draws for estimation
    """
    num_firms = input_data["errors"].shape[0]
    num_countries = input_data["errors"].shape[1]
    
    print(f"\nGenerating export choices...")
    print(f"  True parameters: {theta_true}")
    print(f"  Parameter breakdown:")
    print(f"    - Firm/home heterogeneity: {theta_true[0]:.2f}")
    print(f"    - Item modular (GDP, pop, etc.): {theta_true[1:5]}")
    print(f"    - Proximity/language/region: {theta_true[-3:]}")
    
    # Configuration
    cfg = {
        "dimensions": {
            "num_agents": num_firms,
            "num_items": num_countries,
            "num_features": num_features,
            "num_simuls": 1  # For generation, use 1
        },
        "subproblem": {
            "name": "QuadSupermodularNetwork",
            "settings": {}
        }
    }
    
    # Validate input data before passing to BundleChoice
    print("  Validating input data...")
    print(f"    item_modular: {input_data['item_data']['modular'].shape}")
    print(f"    item_quadratic: {input_data['item_data']['quadratic'].shape}")
    print(f"    agent_modular: {input_data['agent_data']['modular'].shape}")
    print(f"    errors: {input_data['errors'].shape}")
    print(f"    theta_true length: {len(theta_true)}, num_features: {num_features}")
    
    assert input_data['item_data']['modular'].shape == (num_countries, 4), f"Item modular shape mismatch"
    assert input_data['item_data']['quadratic'].shape == (num_countries, num_countries, 3), f"Item quadratic shape mismatch"
    assert input_data['agent_data']['modular'].shape == (num_firms, num_countries, 1), f"Agent modular shape mismatch"
    assert input_data['errors'].shape == (num_firms, num_countries), f"Errors shape mismatch"
    assert len(theta_true) == num_features, f"Theta length mismatch"
    
    # Check for NaN or Inf
    for key in ['item_data', 'agent_data']:
        for subkey, arr in input_data[key].items():
            assert not np.isnan(arr).any(), f"NaN found in {key}/{subkey}"
            assert not np.isinf(arr).any(), f"Inf found in {key}/{subkey}"
    
    # Check quadratic features are non-negative
    qmin = input_data['item_data']['quadratic'].min()
    qmax = input_data['item_data']['quadratic'].max()
    print(f"    Quadratic range: [{qmin:.3f}, {qmax:.3f}]")
    assert qmin >= 0, f"Quadratic features have negative values: min={qmin}"
    
    print("  ✓ All validation checks passed")
    
    # Initialize BundleChoice  
    try:
        print("  Step 1: Creating BundleChoice instance...")
        bc = BundleChoice()
        
        print("  Step 2: Loading config...")
        bc.load_config(cfg)
        
        print("  Step 3: Loading and scattering data...")
        bc.data.load_and_scatter(input_data)
        
        print("  Step 4: Building features...")
        bc.features.build_from_data()
        
        print("  Step 5: Solving optimization problems...")
        obs_bundles = bc.subproblems.init_and_solve(theta_true)
        
        print("  ✓ Solving complete!")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    if bc.rank == 0:
        assert obs_bundles is not None
        
        # Statistics
        num_destinations = obs_bundles.sum(axis=1)
        num_exporters = (obs_bundles.sum(axis=0) > 0).sum()
        
        # Check home market participation
        home_exports = np.array([obs_bundles[i, home_countries[i]] for i in range(num_firms)])
        pct_home = home_exports.mean() * 100
        
        # Export destinations by home country
        print(f"\n✓ Generated {num_firms} firm choices")
        print(f"  Destinations per firm: {num_destinations.min():.0f} - {num_destinations.max():.0f} (mean: {num_destinations.mean():.1f})")
        print(f"  Firms exporting to home country: {pct_home:.1f}%")
        print(f"  Countries receiving exports: {num_exporters}/{num_countries}")
        print(f"  Total export relationships: {obs_bundles.sum():.0f}")
        print(f"  Sparsity: {(1 - obs_bundles.sum() / (num_firms * num_countries)) * 100:.1f}%")
        
        # Show sample export patterns
        print(f"\n  Sample export patterns (first 5 firms):")
        for i in range(min(5, num_firms)):
            home_name = country_names[home_countries[i]]
            destinations = [country_names[j] for j in range(num_countries) if obs_bundles[i, j]]
            print(f"    Firm {i} (from {home_name}): {', '.join(destinations)}")
        
        return obs_bundles, home_countries
    
    return None, None


def save_choices(obs_bundles, home_countries, country_names, output_file='datasets/firm_choices.csv'):
    """Save observed choices to CSV."""
    if obs_bundles is None:
        return
    
    df = pd.DataFrame(obs_bundles, columns=country_names)
    df.insert(0, 'home_country', [country_names[i] for i in home_countries])
    df.index.name = 'firm_id'
    df.to_csv(output_file)
    
    print(f"\n✓ Saved choices to {output_file}")


def load_calibrated_parameters():
    """Load literature-calibrated gravity parameters if available."""
    if Path('datasets/calibration_data.npz').exists():
        calib = np.load('datasets/calibration_data.npz', allow_pickle=True)
        params = calib['gravity_params'].item()
        return params
    return None


def main():
    parser = argparse.ArgumentParser(description='Simulate firm export choices')
    parser.add_argument('--num_firms', type=int, default=100, help='Number of firms')
    parser.add_argument('--sigma', type=float, default=2.0, help='Error standard deviation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use_calibration', action='store_true', help='Use literature-calibrated parameters')
    parser.add_argument('--theta_home', type=float, default=None, help='Home country coefficient (default: calibrated 2.5)')
    parser.add_argument('--theta_firm', type=float, default=0.5, help='Firm heterogeneity coefficient')
    parser.add_argument('--theta_proximity', type=float, default=None, help='Proximity coefficient (default: calibrated 1.1)')
    parser.add_argument('--theta_language', type=float, default=None, help='Common language coefficient (default: calibrated 0.35)')
    parser.add_argument('--theta_region', type=float, default=None, help='Common region coefficient (default: calibrated 0.65)')
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    print("=" * 60)
    print("GRAVITY MODEL: SIMULATE FIRM EXPORT CHOICES")
    print("=" * 60)
    
    # Load data
    features, distances, pairwise_data = load_gravity_data()
    
    # Load calibrated parameters if requested
    calibrated_params = None
    if args.use_calibration or any([args.theta_home is None, args.theta_proximity is None, 
                                     args.theta_language is None, args.theta_region is None]):
        calibrated_params = load_calibrated_parameters()
        if calibrated_params:
            print("\n✓ Using literature-calibrated parameters (Head & Mayer 2014)")
    
    # Define true parameters with calibrated defaults
    # Order: [agent_modular (combined), item_modular..., item_quadratic...]
    num_item_modular = 4  # Default features
    
    # Use calibrated or user-specified values
    theta_proximity = args.theta_proximity if args.theta_proximity is not None else \
                     (calibrated_params['theta_proximity'] if calibrated_params else 0.5)
    theta_language = args.theta_language if args.theta_language is not None else \
                    (calibrated_params['theta_language'] if calibrated_params else 0.3)
    theta_region = args.theta_region if args.theta_region is not None else \
                  (calibrated_params['theta_region'] if calibrated_params else 0.2)
    theta_gdp = calibrated_params['theta_gdp'] if calibrated_params else 1.0
    
    # Note: theta_home and theta_firm now combined into single agent_modular coefficient
    # (home boost is baked into the features, scaled by this single parameter)
    theta_true = np.array([
        args.theta_firm,     # Firm/home heterogeneity (combined)
        theta_gdp, 0.5, 0.3, 0.2,  # GDP, pop, GDP/capita, trade openness
        theta_proximity,     # Proximity (positive, inverse of distance)
        theta_language,      # Common language
        theta_region,        # Common region
    ])
    
    print(f"\nSimulation parameters:")
    print(f"  Firms: {args.num_firms}")
    print(f"  Error std: {args.sigma}")
    print(f"  Random seed: {args.seed}")
    
    # Prepare data
    print("\n[DEBUG] Calling prepare_bundlechoice_data...")
    input_data, num_features, home_countries, country_names = prepare_bundlechoice_data(
        features, distances, pairwise_data,
        num_firms=args.num_firms,
        theta_true=theta_true,
        sigma=args.sigma,
        seed=args.seed
    )
    print(f"[DEBUG] Data prepared. num_features={num_features}")
    
    # Generate choices
    print("[DEBUG] Calling generate_choices...")
    obs_bundles, home_countries_result = generate_choices(
        input_data, num_features, theta_true, home_countries, country_names
    )
    print("[DEBUG] Choices generated.")
    
    # Save results
    if obs_bundles is not None:
        save_choices(obs_bundles, home_countries, country_names)
        
        # Save data for estimation
        np.savez(
            'datasets/simulation_data.npz',
            obs_bundles=obs_bundles,
            home_countries=home_countries,
            item_modular=input_data['item_data']['modular'],
            item_quadratic=input_data['item_data']['quadratic'],
            agent_modular=input_data['agent_data']['modular'],
            theta_true=theta_true,
            country_names=country_names,
            sigma=args.sigma
        )
        print(f"✓ Saved full simulation data to datasets/simulation_data.npz")
    
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
