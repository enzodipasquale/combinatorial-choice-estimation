"""
Simulate using QuadSupermodular with proper constraint masks for home exclusion.
"""
import numpy as np
import pandas as pd
from bundlechoice.core import BundleChoice


def load_data():
    features = pd.read_csv('datasets/country_features.csv', index_col=0)
    distances = pd.read_csv('datasets/distances.csv', index_col=0)
    return features, distances


def assign_firms(features, num_firms, seed=42):
    np.random.seed(seed)
    gdp = features['gdp_billions'].fillna(0).values
    weights = np.power(gdp, 0.8)
    weights = weights / weights.sum()
    home_countries = np.random.choice(len(features), size=num_firms, p=weights)
    return home_countries


def prepare_data(features, distances, home_countries):
    """Prepare data with proper constraint masks."""
    num_firms = len(home_countries)
    num_countries = len(features)
    
    # Modular features
    gdp = features['gdp_billions'].fillna(0).values
    pop = features['population_millions'].fillna(0).values
    
    gdp_std = (gdp - gdp.mean()) / (gdp.std() + 1e-8)
    pop_std = (pop - pop.mean()) / (pop.std() + 1e-8)
    
    item_modular = np.column_stack([gdp_std, pop_std])
    
    # Quadratic: proximity (inverse log distance)
    log_dist = np.log(distances.values + 1)
    proximity = log_dist.max() - log_dist
    proximity = proximity / proximity.max()
    np.fill_diagonal(proximity, 0)
    
    item_quadratic = proximity[:, :, np.newaxis]  # Shape: (50, 50, 1)
    
    # Agent modular (heterogeneity)
    agent_modular = np.random.normal(0, 0.5, (num_firms, num_countries, 1))
    
    # Errors
    errors = np.random.normal(0, 1.5, (num_firms, num_countries))
    
    # CONSTRAINT MASKS: Exclude home country for each firm
    # Format: List of boolean arrays (True = feasible, False = excluded)
    constraint_mask = []
    for i in range(num_firms):
        home = home_countries[i]
        mask = np.ones(num_countries, dtype=bool)
        mask[home] = False  # Exclude home!
        constraint_mask.append(mask)
    
    print(f"\n✓ Prepared data:")
    print(f"  item_modular: {item_modular.shape}")
    print(f"  item_quadratic: {item_quadratic.shape}")
    print(f"  agent_modular: {agent_modular.shape}")
    print(f"  errors: {errors.shape}")
    print(f"  constraint_mask: {len(constraint_mask)} agents")
    print(f"  Example mask[0] (home={home_countries[0]}): {constraint_mask[0][:5]}... (sum={constraint_mask[0].sum()})")
    
    input_data = {
        "item_data": {
            "modular": item_modular,
            "quadratic": item_quadratic
        },
        "agent_data": {
            "modular": agent_modular
        },
        "errors": errors,
        "constraint_mask": constraint_mask  # HOME EXCLUSION!
    }
    
    return input_data


def main():
    print("="*60)
    print("QUADRATIC SUPERMODULAR SIMULATION")
    print("="*60)
    
    # Config
    num_firms = 10000
    seed = 42
    
    # Load
    features, distances = load_data()
    print(f"\n✓ Loaded: {len(features)} countries")
    
    # Assign homes
    home_countries = assign_firms(features, num_firms, seed)
    unique, counts = np.unique(home_countries, return_counts=True)
    top5 = np.argsort(counts)[::-1][:5]
    print(f"\nFirm distribution (top 5):")
    for idx in top5:
        print(f"  {features.index[unique[idx]]}: {counts[idx]} firms")
    
    # Prepare with constraint masks
    input_data = prepare_data(features, distances, home_countries)
    
    # Parameters (minimal complementarities for sparsity!)
    theta = np.array([
        0.3,    # agent heterogeneity
        2.0,    # GDP (strong)
        0.5,    # Population (moderate)
        0.0,    # Proximity (ZERO complementarity for sparsity!)
    ])
    
    print(f"\nParameters:")
    print(f"  θ_agent: {theta[0]}")
    print(f"  θ_modular (GDP, pop): {theta[1:3]}")
    print(f"  θ_quadratic (proximity): {theta[3]} ← Near zero for sparsity!")
    
    # Configure BundleChoice
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
    
    print(f"\nSolving...")
    bc = BundleChoice()
    bc.load_config(cfg)
    bc.data.load_and_scatter(input_data)
    bc.features.build_from_data()
    
    bundles = bc.subproblems.init_and_solve(theta)
    
    if bc.rank == 0:
        print(f"\n{'='*60}")
        print("RESULTS")
        print("="*60)
        
        partners = bundles.sum(axis=1)
        print(f"\n✓ Generated {len(bundles)} choices")
        print(f"  Partners: {partners.min()}-{partners.max()} (mean: {partners.mean():.1f})")
        print(f"  Sparsity: {(1 - bundles.sum()/(num_firms*len(features)))*100:.1f}%")
        
        # Check home exclusion
        home_exports = sum([bundles[i, home_countries[i]] for i in range(num_firms)])
        print(f"  Self-trade: {home_exports}/{num_firms} ({home_exports/num_firms*100:.1f}%)")
        
        if home_exports == 0:
            print(f"  ✅ HOME EXCLUSION WORKING (constraint mask used correctly!)")
        else:
            print(f"  ❌ Constraint mask NOT working")
        
        # Top destinations
        inflows = bundles.sum(axis=0)
        top_idx = np.argsort(inflows)[::-1][:10]
        print(f"\n  Top 10 destinations:")
        for i, idx in enumerate(top_idx):
            print(f"    {i+1}. {features.index[idx]}: {int(inflows[idx])} firms ({inflows[idx]/num_firms*100:.1f}%)")
        
        # Save
        df = pd.DataFrame(bundles, columns=features.index)
        df.insert(0, 'home_country', [features.index[h] for h in home_countries])
        df.to_csv('datasets/quad_simulation.csv', index=False)
        
        np.savez('datasets/quad_simulation.npz',
                 bundles=bundles,
                 home_countries=home_countries,
                 country_names=features.index.tolist(),
                 theta_true=theta)
        
        print(f"\n✓ Saved: datasets/quad_simulation.csv and .npz")
        print("="*60)


if __name__ == '__main__':
    main()

