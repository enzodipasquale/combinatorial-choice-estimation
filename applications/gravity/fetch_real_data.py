"""
Fetch real data for gravity model calibration:
1. Number of firms/enterprises per country
2. Bilateral trade flows for parameter calibration
"""
import numpy as np
import pandas as pd
from pandas_datareader import wb
import pycountry
import warnings
warnings.filterwarnings('ignore')


def fetch_real_firm_counts_wb(country_codes):
    """
    Fetch firm density indicators from World Bank.
    Simplified to avoid hangs - uses batch approach.
    """
    from pandas_datareader import wb
    
    print(f"Fetching firm indicators for {len([c for c in country_codes if c])} countries...")
    
    firm_estimates = {}
    
    # Use business density indicator (most reliable)
    try:
        print("  Fetching new business registrations...")
        # Fetch for valid codes only
        valid_codes = [c for c in country_codes if c is not None]
        
        for iso2 in valid_codes:
            try:
                df = wb.download(indicator='IC.BUS.NREG', country=iso2, start=2019, end=2022)
                if not df.empty:
                    vals = df['IC.BUS.NREG'].dropna()
                    if len(vals) > 0:
                        # Estimate: 15 years of registrations as proxy for total stock
                        firm_estimates[iso2] = vals.mean() * 15
                        continue
            except:
                pass
            
            # Set to NaN if failed
            firm_estimates[iso2] = np.nan
    
    except Exception as e:
        print(f"  Warning: Batch fetch failed: {e}")
    
    sources_found = sum([not pd.isna(v) for v in firm_estimates.values()])
    print(f"  ✓ Got estimates for {sources_found}/{len(valid_codes)} countries")
    
    return firm_estimates


def fetch_bilateral_trade(country_codes, year=2022):
    """
    Fetch bilateral trade flows between countries.
    This would ideally come from UN Comtrade or WITS, but we'll use a simple proxy.
    """
    print(f"\nFetching bilateral trade data for {year}...")
    
    # For now, create a synthetic but realistic trade matrix based on:
    # - Gravity equation: Trade ~ GDP_i * GDP_j / Distance^gamma
    # - We already have this data
    
    print("  Using gravity-based trade flow estimates")
    return None  # Placeholder - would need WITS/Comtrade API


def calibrate_firm_distribution(features, use_real_firms=True):
    """
    Create firm distribution dynamically from World Bank data.
    
    Priority:
    1. World Bank enterprise indicators
    2. GDP^0.8 scaling (empirically realistic)
    """
    country_codes = features.index.tolist()  # ISO3 codes
    
    # Convert to ISO2 for World Bank API using pycountry
    iso2_codes = []
    print(f"Converting {len(country_codes)} ISO3 codes to ISO2...")
    for iso3 in country_codes:
        try:
            country = pycountry.countries.get(alpha_3=iso3)
            if country:
                iso2_codes.append(country.alpha_2)
                print(f"  {iso3} -> {country.alpha_2}")
            else:
                print(f"  {iso3} -> NOT FOUND")
                iso2_codes.append(None)
        except Exception as e:
            print(f"  {iso3} -> ERROR: {e}")
            iso2_codes.append(None)
    
    if use_real_firms:
        print("\nCalibrating from World Bank + GDP scaling...")
        firm_estimates = fetch_real_firm_counts_wb(iso2_codes)
        
        # Build counts array
        counts = np.array([firm_estimates.get(iso2, np.nan) for iso2 in iso2_codes])
        has_data = ~np.isnan(counts)
        coverage = has_data.sum() / len(country_codes)
        
        if coverage > 0.2:  # If we have data for >20% of countries
            print(f"\n  Strategy: Use real data where available, impute rest with GDP^0.8")
            print(f"  Real data: {has_data.sum()}/{len(country_codes)} countries")
            
            # Impute missing using GDP^0.8, scaled to match real data
            gdp_values = features['gdp_billions'].fillna(0).values
            gdp_scale = np.power(gdp_values, 0.8)
            
            if has_data.any() and gdp_scale[has_data].sum() > 0:
                # Scale GDP to match real firm counts
                scale_factor = counts[has_data].sum() / gdp_scale[has_data].sum()
                counts[~has_data] = gdp_scale[~has_data] * scale_factor
                print(f"  Imputed: {(~has_data).sum()} countries using scale factor {scale_factor:.0f}")
            else:
                # Fallback to just GDP scaling
                counts = gdp_scale
                print(f"  ! Using GDP^0.8 for all countries")
            
            weights = counts
        else:
            print(f"  ! Low coverage ({coverage*100:.0f}%), using GDP^0.8 only")
            weights = features['gdp_billions'].values ** 0.8
    else:
        print("\nUsing GDP^0.8 scaling...")
        weights = features['gdp_billions'].values ** 0.8
    
    # Normalize
    weights = np.maximum(weights, 0)
    total_firms = weights.sum()
    weights = weights / total_firms
    
    # Display
    print(f"\nFirm distribution:")
    print(f"  Total estimated firms: {total_firms:,.0f}")
    for country, prob, count in zip(country_codes, weights, weights * total_firms):
        print(f"    {country}: {prob*100:.1f}% ({count:,.0f} firms)")
    
    return weights, total_firms


def estimate_gravity_parameters(features, distances):
    """
    Estimate realistic parameter values from gravity equation literature.
    
    Standard gravity equation:
    log(Trade_ij) = β0 + β1*log(GDP_i) + β2*log(GDP_j) - γ*log(Distance_ij) + ...
    
    Literature estimates (meta-analysis):
    - GDP elasticity: ~0.7-1.0
    - Distance elasticity: -0.9 to -1.2 (negative!)
    - Common language: +0.3 to +0.5
    - Common border: +0.5 to +0.8
    """
    print("\nCalibrating parameters from gravity equation literature...")
    
    # Literature-based estimates (Head & Mayer 2014 meta-analysis)
    params = {
        'gdp_elasticity': 0.85,        # Trade increases with market size
        'distance_elasticity': -1.1,   # Trade decreases with distance
        'language_effect': 0.35,       # Common language boosts trade
        'border_effect': 0.65,         # Contiguity/regional integration
        'home_bias': 2.5,              # Strong home market effect
    }
    
    print(f"  GDP elasticity: {params['gdp_elasticity']}")
    print(f"  Distance elasticity: {params['distance_elasticity']} (negative = trade cost)")
    print(f"  Common language effect: {params['language_effect']}")
    print(f"  Border/regional effect: {params['border_effect']}")
    print(f"  Home market bias: {params['home_bias']}")
    
    # Convert to our supermodular framework
    # Distance elasticity is NEGATIVE in gravity, but we use PROXIMITY (inverse)
    # So positive coefficient on proximity = negative on distance
    params_supermod = {
        'theta_home': params['home_bias'],
        'theta_gdp': params['gdp_elasticity'],
        'theta_proximity': abs(params['distance_elasticity']),  # Inverse of distance
        'theta_language': params['language_effect'],
        'theta_region': params['border_effect'],
    }
    
    return params_supermod




def main():
    """Test fetching real data."""
    # Load existing country features
    features = pd.read_csv('datasets/country_features.csv', index_col=0)
    distances = pd.read_csv('datasets/distances.csv', index_col=0)
    
    print("=" * 60)
    print("FETCHING REAL DATA FOR CALIBRATION")
    print("=" * 60)
    
    # Get firm distribution from REAL data
    firm_weights, total_real_firms = calibrate_firm_distribution(features, use_real_firms=True)
    
    # Get gravity parameters
    gravity_params = estimate_gravity_parameters(features, distances)
    
    # Save calibration
    np.savez(
        'datasets/calibration_data.npz',
        firm_weights=firm_weights,
        total_real_firms=total_real_firms,
        gravity_params=gravity_params,
        country_names=features.index.tolist()
    )
    
    print(f"\n✓ Saved calibration data to datasets/calibration_data.npz")
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
