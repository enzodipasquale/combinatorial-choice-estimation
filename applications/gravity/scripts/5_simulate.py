"""
Generate firm export choices using quadratic supermodular solver.
Flexible covariate selection, constraint support, real firm distribution.
"""
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from bundlechoice.core import BundleChoice


def load_all_data():
    """Load all generated gravity data."""
    features = pd.read_csv('data/features/country_features.csv', index_col=0)
    
    pairwise_files = {
        'distances': 'data/features/distances.csv',
        'common_language': 'data/features/common_language.csv',
        'common_region': 'data/features/common_region.csv',
        'contiguity': 'data/features/contiguity.csv',
        'timezone_difference': 'data/features/timezone_difference.csv',
        'colonial_ties': 'data/features/colonial_ties.csv',
        'legal_origin_similarity': 'data/features/legal_origin_similarity.csv',
        'bilateral_tariffs': 'data/features/bilateral_tariffs.csv',
    }
    
    pairwise = {}
    for name, path in pairwise_files.items():
        if Path(path).exists():
            pairwise[name] = pd.read_csv(path, index_col=0)
    
    return features, pairwise


def assign_firms(features, num_firms, seed):
    """Assign firms to countries based on real firm distribution."""
    np.random.seed(seed)
    
    # Load calibrated weights
    calib = np.load('data/features/calibration_data.npz', allow_pickle=True)
    weights = calib['firm_weights']
    
    home_countries = np.random.choice(len(features), size=num_firms, p=weights)
    
    # Show distribution
    unique, counts = np.unique(home_countries, return_counts=True)
    print(f"\nFirm distribution:")
    sorted_idx = np.argsort(counts)[::-1][:10]
    for idx in sorted_idx:
        country_idx = unique[idx]
        print(f"  {features.index[country_idx]}: {counts[idx]} ({counts[idx]/num_firms*100:.1f}%)")
    
    return home_countries


def prepare_features(features, pairwise, modular_vars, quadratic_vars):
    """
    Flexibly select and prepare features.
    
    Args:
        modular_vars: List of column names from country_features.csv
        quadratic_vars: List of keys from pairwise dict
    """
    print(f"\nPreparing features...")
    print(f"  Modular: {modular_vars}")
    print(f"  Quadratic: {quadratic_vars}")
    
    # Item modular
    item_modular = features[modular_vars].fillna(features[modular_vars].mean()).values
    item_modular = (item_modular - item_modular.mean(axis=0)) / (item_modular.std(axis=0) + 1e-8)
    
    # Item quadratic
    quad_matrices = []
    for var in quadratic_vars:
        if var == 'distances':
            # Transform to proximity (inverse log distance)
            dist = pairwise[var].values
            log_dist = np.log(dist + 1)
            proximity = log_dist.max() - log_dist
            proximity = proximity / proximity.max()
            np.fill_diagonal(proximity, 0)
            quad_matrices.append(proximity)
        elif var == 'bilateral_tariffs':
            # Invert tariffs: lower tariff = higher value (complementarity)
            tariff = pairwise[var].values
            tariff_penalty = 15 - tariff  # Max tariff ~14%, so this gives [1, 15]
            tariff_penalty = tariff_penalty / tariff_penalty.max()
            np.fill_diagonal(tariff_penalty, 0)
            quad_matrices.append(tariff_penalty)
        else:
            # Binary features (language, region, etc.)
            mat = pairwise[var].values.copy()
            np.fill_diagonal(mat, 0)
            quad_matrices.append(mat)
    
    item_quadratic = np.stack(quad_matrices, axis=2)
    
    # Verify non-negative
    assert np.all(item_quadratic >= 0), "Quadratic features must be non-negative for supermodular"
    
    print(f"  Item modular: {item_modular.shape}")
    print(f"  Item quadratic: {item_quadratic.shape}")
    
    return item_modular, item_quadratic


def create_input_data(features, item_modular, item_quadratic, home_countries, 
                      exclude_home, sigma, seed):
    """Create BundleChoice input data with optional home country exclusion."""
    np.random.seed(seed)
    
    num_firms = len(home_countries)
    num_countries = len(features)
    
    # Agent modular: firm heterogeneity
    agent_modular = np.random.normal(0, 0.3, (num_firms, num_countries, 1))
    
    # Implement home exclusion via very negative utility (soft constraint)
    if exclude_home:
        print(f"\n  ⚠️  Excluding home country via large negative utility")
        for i in range(num_firms):
            agent_modular[i, home_countries[i], 0] = -1000.0  # Very negative
    
    # Errors
    errors = sigma * np.random.normal(0, 1, (num_firms, num_countries))
    
    input_data = {
        "item_data": {
            "modular": item_modular,
            "quadratic": item_quadratic
        },
        "agent_data": {
            "modular": agent_modular
        },
        "errors": errors
    }
    
    return input_data


def solve_bundlechoice(input_data, num_firms, num_countries, num_features, theta_true):
    """Solve using BundleChoice quadratic supermodular solver."""
    
    cfg = {
        "dimensions": {
            "num_agents": num_firms,
            "num_items": num_countries,
            "num_features": num_features,
            "num_simuls": 1
        },
        "subproblem": {
            "name": "QuadSupermodularNetwork",
            "settings": {}
        }
    }
    
    bc = BundleChoice()
    bc.load_config(cfg)
    bc.data.load_and_scatter(input_data)
    bc.features.build_from_data()
    bc.subproblems.load()  # Initialize subproblem solver
    
    print(f"\n  Solving with QuadSupermodularNetwork...")
    obs_bundles = bc.subproblems.init_and_solve(theta_true)
    
    if bc.rank == 0:
        return obs_bundles, bc.rank
    return None, bc.rank


def analyze_results(obs_bundles, home_countries, country_names, exclude_home):
    """Quick analysis of results."""
    if obs_bundles is None:
        return
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    
    num_firms, num_countries = obs_bundles.shape
    destinations_per_firm = obs_bundles.sum(axis=1)
    
    print(f"\n✓ Generated {num_firms} firm choices")
    print(f"  Destinations per firm: {destinations_per_firm.min():.0f}-{destinations_per_firm.max():.0f} (mean: {destinations_per_firm.mean():.1f})")
    print(f"  Total export relationships: {obs_bundles.sum():.0f}")
    print(f"  Sparsity: {(1 - obs_bundles.sum()/(num_firms*num_countries))*100:.1f}%")
    
    # Check home constraint
    home_exports = sum([obs_bundles[i, home_countries[i]] for i in range(num_firms)])
    print(f"  Firms exporting to home: {home_exports}/{num_firms} ({home_exports/num_firms*100:.1f}%)")
    
    if exclude_home and home_exports > 0:
        print(f"  ⚠️  WARNING: {home_exports} firms violated home exclusion constraint!")
    elif exclude_home:
        print(f"  ✓ Home exclusion constraint satisfied")
    
    # Top destinations
    inflows = obs_bundles.sum(axis=0)
    top_idx = np.argsort(inflows)[::-1][:10]
    print(f"\n  Top 10 destinations:")
    for idx in top_idx:
        print(f"    {country_names[idx]}: {inflows[idx]} firms ({inflows[idx]/num_firms*100:.1f}%)")


def save_results(obs_bundles, home_countries, country_names, theta_true, args):
    """Save simulation results."""
    if obs_bundles is None:
        return
    
    # CSV format
    df = pd.DataFrame(obs_bundles, columns=country_names)
    df.insert(0, 'home_country', [country_names[i] for i in home_countries])
    df.to_csv('data/simulation/obs_bundles.csv')
    
    # NPZ format (for estimation)
    np.savez('data/simulation/obs_bundles.npz',
             obs_bundles=obs_bundles,
             home_countries=home_countries,
             theta_true=theta_true,
             country_names=country_names,
             **vars(args))
    
    print(f"\n✓ Saved: data/obs_bundles.csv and .npz")


def main():
    parser = argparse.ArgumentParser(description='Simulate firm export choices')
    parser.add_argument('--num_firms', type=int, default=1000, help='Number of firms')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--sigma', type=float, default=2.0, help='Error std dev')
    parser.add_argument('--exclude_home', action='store_true', help='Firms cannot export to home country')
    
    # Flexible covariate selection
    parser.add_argument('--modular', nargs='+', 
                       default=['gdp_billions', 'population_millions', 'gdp_per_capita', 'trade_openness_pct'],
                       help='Modular features from country_features.csv')
    parser.add_argument('--quadratic', nargs='+',
                       default=['distances', 'common_language', 'common_region'],
                       help='Quadratic features: distances, common_language, common_region, bilateral_tariffs, etc.')
    
    # Parameters
    parser.add_argument('--theta_agent', type=float, default=0.5, help='Agent heterogeneity')
    parser.add_argument('--theta_modular', nargs='+', type=float, default=None,
                       help='Modular coefficients (one per modular var)')
    parser.add_argument('--theta_quadratic', nargs='+', type=float, default=None,
                       help='Quadratic coefficients (one per quadratic var)')
    
    args = parser.parse_args()
    
    print("="*60)
    print(f"GRAVITY MODEL SIMULATION")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Firms: {args.num_firms}")
    print(f"  Modular vars: {args.modular}")
    print(f"  Quadratic vars: {args.quadratic}")
    print(f"  Exclude home: {args.exclude_home}")
    
    # Load data
    features, pairwise = load_all_data()
    print(f"\n✓ Loaded {len(features)} countries, {len(pairwise)} pairwise matrices")
    
    # Assign firms
    home_countries = assign_firms(features, args.num_firms, args.seed)
    
    # Prepare features
    item_modular, item_quadratic = prepare_features(features, pairwise, 
                                                     args.modular, args.quadratic)
    
    # Create input data
    input_data = create_input_data(features, item_modular, item_quadratic, 
                                   home_countries, args.exclude_home, 
                                   args.sigma, args.seed)
    
    # Set parameters
    num_modular = len(args.modular)
    num_quadratic = len(args.quadratic)
    num_features = 1 + num_modular + num_quadratic  # agent + item_mod + item_quad
    
    theta_modular = args.theta_modular if args.theta_modular else [0.8] * num_modular
    theta_quadratic = args.theta_quadratic if args.theta_quadratic else [0.5] * num_quadratic
    
    theta_true = np.array([args.theta_agent] + theta_modular + theta_quadratic)
    
    print(f"\nParameters (θ):")
    print(f"  Agent: {args.theta_agent}")
    print(f"  Modular: {theta_modular}")
    print(f"  Quadratic: {theta_quadratic}")
    
    # Solve
    print(f"\nSolving...")
    obs_bundles, rank = solve_bundlechoice(input_data, args.num_firms, len(features), 
                                           num_features, theta_true)
    
    # Analyze and save (rank 0 only)
    if rank == 0:
        analyze_results(obs_bundles, home_countries, features.index.tolist(), args.exclude_home)
        save_results(obs_bundles, home_countries, features.index.tolist(), theta_true, args)
        
        print(f"\n{'='*60}")
        print("✅ SIMULATION COMPLETE")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
