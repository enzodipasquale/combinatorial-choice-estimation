"""
Sanity checks for gravity model data.
Validates correctness of generated datasets.
"""
import numpy as np
import pandas as pd
import sys

def check_country_features():
    """Validate country-level features."""
    print("=" * 60)
    print("VALIDATING COUNTRY FEATURES")
    print("=" * 60)
    
    df = pd.read_csv('datasets/country_features.csv', index_col=0)
    
    print(f"\n✓ Loaded {len(df)} countries")
    print(f"✓ Total features: {len(df.columns)}")
    
    errors = []
    warnings = []
    
    # Check for all NaN columns
    all_nan_cols = df.columns[df.isna().all()].tolist()
    if all_nan_cols:
        warnings.append(f"Columns with all NaN: {all_nan_cols}")
    
    # Check critical variables exist and have values
    critical_vars = ['gdp_billions', 'population_millions', 'gdp_per_capita']
    for var in critical_vars:
        if var not in df.columns:
            errors.append(f"Missing critical variable: {var}")
        elif df[var].isna().all():
            errors.append(f"All NaN in critical variable: {var}")
        elif (df[var] <= 0).any():
            errors.append(f"Non-positive values in {var}: {df[var][df[var] <= 0].index.tolist()}")
    
    # Check value ranges
    if 'gdp_billions' in df.columns:
        valid_gdp = df['gdp_billions'].dropna()
        if valid_gdp.min() < 100 or valid_gdp.max() > 50000:
            warnings.append(f"GDP range suspicious: {valid_gdp.min():.1f} to {valid_gdp.max():.1f} billion USD")
        print(f"\n✓ GDP range: ${valid_gdp.min():.1f}B - ${valid_gdp.max():.1f}B")
    
    if 'population_millions' in df.columns:
        valid_pop = df['population_millions'].dropna()
        if valid_pop.min() < 1 or valid_pop.max() > 2000:
            warnings.append(f"Population range suspicious: {valid_pop.min():.1f} to {valid_pop.max():.1f} million")
        print(f"✓ Population range: {valid_pop.min():.1f}M - {valid_pop.max():.1f}M")
    
    if 'gdp_per_capita' in df.columns:
        valid_gdppc = df['gdp_per_capita'].dropna()
        if valid_gdppc.min() < 500 or valid_gdppc.max() > 150000:
            warnings.append(f"GDP per capita range suspicious: ${valid_gdppc.min():.0f} - ${valid_gdppc.max():.0f}")
        print(f"✓ GDP per capita range: ${valid_gdppc.min():.0f} - ${valid_gdppc.max():.0f}")
    
    # Check percentage variables are in valid range
    pct_vars = [col for col in df.columns if 'pct' in col or 'rate' in col or col.endswith('_per_100')]
    for var in pct_vars:
        if var == 'mobile_per_100':  # Can exceed 100
            continue
        valid_vals = df[var].dropna()
        if len(valid_vals) > 0:
            if valid_vals.min() < 0 or valid_vals.max() > 100:
                warnings.append(f"{var} outside [0,100]: {valid_vals.min():.1f} to {valid_vals.max():.1f}")
    
    # Check language/region dummies
    lang_cols = [col for col in df.columns if len(col) <= 3 and col.islower()]
    region_cols = [col for col in df.columns if col in ['Europe', 'Asia', 'Africa', 'Americas', 'America', 'Oceania']]
    
    if lang_cols:
        lang_df = df[lang_cols]
        langs_per_country = lang_df.sum(axis=1)
        if (langs_per_country != 1).any():
            warnings.append(f"Countries with != 1 language: {df.index[langs_per_country != 1].tolist()}")
        print(f"\n✓ Languages found: {len(lang_cols)} ({', '.join(lang_cols[:5])}...)")
    
    if region_cols:
        region_df = df[region_cols]
        regions_per_country = region_df.sum(axis=1)
        if (regions_per_country != 1).any():
            warnings.append(f"Countries with != 1 region: {df.index[regions_per_country != 1].tolist()}")
        print(f"✓ Regions found: {len(region_cols)} ({', '.join(region_cols)})")
    
    # Check for duplicates
    if df.index.duplicated().any():
        errors.append(f"Duplicate countries: {df.index[df.index.duplicated()].tolist()}")
    
    # Missing data summary
    missing_pct = df.isna().sum() / len(df) * 100
    high_missing = missing_pct[missing_pct > 50].sort_values(ascending=False)
    if len(high_missing) > 0:
        warnings.append(f"Variables with >50% missing: {high_missing.index.tolist()}")
        print(f"\n⚠ {len(high_missing)} variables with >50% missing data")
    
    print(f"\n✓ Average missing data: {missing_pct.mean():.1f}%")
    
    return errors, warnings


def check_distances():
    """Validate distance matrix."""
    print("\n" + "=" * 60)
    print("VALIDATING DISTANCE MATRIX")
    print("=" * 60)
    
    df = pd.read_csv('datasets/distances.csv', index_col=0)
    
    errors = []
    warnings = []
    
    print(f"\n✓ Shape: {df.shape}")
    
    # Check square matrix
    if df.shape[0] != df.shape[1]:
        errors.append(f"Not square: {df.shape}")
    
    # Check index == columns
    if not df.index.equals(df.columns):
        errors.append("Index and columns don't match")
    
    # Check diagonal is zero
    diag = np.diag(df.values)
    if not np.allclose(diag, 0):
        errors.append(f"Diagonal not zero: {diag[diag != 0]}")
    else:
        print("✓ Diagonal is zero")
    
    # Check symmetry
    if not np.allclose(df.values, df.values.T, rtol=1e-5):
        errors.append("Matrix not symmetric")
    else:
        print("✓ Matrix is symmetric")
    
    # Check positive values
    off_diag = df.values[~np.eye(df.shape[0], dtype=bool)]
    if (off_diag <= 0).any():
        errors.append(f"Non-positive distances found: {off_diag[off_diag <= 0]}")
    
    # Check reasonable range (Earth's max distance ~20,000 km)
    if off_diag.max() > 21000:
        warnings.append(f"Suspiciously large distance: {off_diag.max():.0f} km")
    if off_diag.min() < 50 and off_diag.min() > 0:
        warnings.append(f"Suspiciously small distance: {off_diag.min():.0f} km")
    
    print(f"✓ Distance range: {off_diag.min():.0f} - {off_diag.max():.0f} km")
    print(f"✓ Mean distance: {off_diag.mean():.0f} km")
    
    return errors, warnings


def check_pairwise_features():
    """Validate pairwise feature matrices."""
    print("\n" + "=" * 60)
    print("VALIDATING PAIRWISE FEATURES")
    print("=" * 60)
    
    pairwise_files = [
        'common_language.csv',
        'common_region.csv',
        'contiguity.csv',
        'timezone_difference.csv',
        'colonial_ties.csv',
        'legal_origin_similarity.csv'
    ]
    
    all_errors = []
    all_warnings = []
    
    for filename in pairwise_files:
        try:
            df = pd.read_csv(f'datasets/{filename}', index_col=0)
            print(f"\n{filename}:")
            
            errors = []
            warnings = []
            
            # Check square
            if df.shape[0] != df.shape[1]:
                errors.append(f"  Not square: {df.shape}")
            
            # Check binary features
            if 'common' in filename or 'contiguity' in filename or 'legal' in filename or 'colonial' in filename:
                unique_vals = set(df.values.flatten())
                if not unique_vals.issubset({0, 1, 0.0, 1.0}):
                    errors.append(f"  Not binary: {unique_vals}")
                else:
                    print(f"  ✓ Binary matrix")
                    
                # Check diagonal for common_* features (should be 1 for common_X)
                if 'common' in filename:
                    diag = np.diag(df.values)
                    if not np.all(diag == 1):
                        errors.append(f"  Diagonal not all 1s for {filename}")
                    else:
                        print(f"  ✓ Diagonal is 1 (self-similarity)")
            
            # Check symmetry for expected symmetric matrices
            if filename not in ['colonial_ties.csv']:  # Colonial ties might be asymmetric
                if not np.allclose(df.values, df.values.T, rtol=1e-5):
                    warnings.append(f"  Matrix not symmetric: {filename}")
                else:
                    print(f"  ✓ Symmetric")
            
            # Check timezone differences
            if 'timezone' in filename:
                valid_vals = df.values[~np.eye(df.shape[0], dtype=bool)]
                if valid_vals.max() > 24:
                    warnings.append(f"  Timezone diff > 24 hours: {valid_vals.max():.1f}")
                print(f"  ✓ Range: 0 - {valid_vals.max():.1f} hours")
            
            # Summary stats
            if 'timezone' not in filename:
                pct_ones = (df.values == 1).sum() / df.size * 100
                print(f"  ✓ {pct_ones:.1f}% are 1s")
            
            all_errors.extend(errors)
            all_warnings.extend(warnings)
            
        except FileNotFoundError:
            all_warnings.append(f"File not found: {filename}")
    
    return all_errors, all_warnings


def check_consistency():
    """Check cross-dataset consistency."""
    print("\n" + "=" * 60)
    print("CHECKING CROSS-DATASET CONSISTENCY")
    print("=" * 60)
    
    errors = []
    warnings = []
    
    # Load datasets
    features = pd.read_csv('datasets/country_features.csv', index_col=0)
    distances = pd.read_csv('datasets/distances.csv', index_col=0)
    
    # Check same countries
    if not features.index.equals(distances.index):
        errors.append("Country lists don't match between features and distances")
        print(f"  Features: {list(features.index)}")
        print(f"  Distances: {list(distances.index)}")
    else:
        print(f"✓ All datasets have same {len(features)} countries")
    
    # Check GDP ranking makes sense
    if 'gdp_billions' in features.columns:
        top5_gdp = features['gdp_billions'].nlargest(5)
        print(f"\n✓ Top 5 by GDP:")
        for country, gdp in top5_gdp.items():
            print(f"  {country}: ${gdp:.0f}B")
    
    # Check population ranking
    if 'population_millions' in features.columns:
        top5_pop = features['population_millions'].nlargest(5)
        print(f"\n✓ Top 5 by population:")
        for country, pop in top5_pop.items():
            print(f"  {country}: {pop:.0f}M")
    
    # Sanity check: GDP per capita = GDP / population (approximately)
    if all(col in features.columns for col in ['gdp_billions', 'population_millions', 'gdp_per_capita']):
        computed_gdppc = (features['gdp_billions'] * 1000) / features['population_millions']
        diff = abs(computed_gdppc - features['gdp_per_capita'])
        # Allow 10% difference due to rounding/different years
        large_diff = diff > 0.1 * features['gdp_per_capita']
        if large_diff.any():
            warnings.append(f"GDP per capita inconsistent for: {features.index[large_diff].tolist()}")
        else:
            print(f"\n✓ GDP per capita consistent with GDP/population")
    
    return errors, warnings


def main():
    """Run all validation checks."""
    print("\n" + "=" * 60)
    print("GRAVITY MODEL DATA VALIDATION")
    print("=" * 60)
    
    all_errors = []
    all_warnings = []
    
    # Run checks
    e, w = check_country_features()
    all_errors.extend(e)
    all_warnings.extend(w)
    
    e, w = check_distances()
    all_errors.extend(e)
    all_warnings.extend(w)
    
    e, w = check_pairwise_features()
    all_errors.extend(e)
    all_warnings.extend(w)
    
    e, w = check_consistency()
    all_errors.extend(e)
    all_warnings.extend(w)
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    if all_errors:
        print(f"\n❌ ERRORS ({len(all_errors)}):")
        for i, error in enumerate(all_errors, 1):
            print(f"  {i}. {error}")
    else:
        print("\n✅ No errors found!")
    
    if all_warnings:
        print(f"\n⚠️  WARNINGS ({len(all_warnings)}):")
        for i, warning in enumerate(all_warnings, 1):
            print(f"  {i}. {warning}")
    else:
        print("\n✅ No warnings!")
    
    print("\n" + "=" * 60)
    
    if all_errors:
        print("❌ VALIDATION FAILED")
        sys.exit(1)
    else:
        print("✅ VALIDATION PASSED")
        sys.exit(0)


if __name__ == '__main__':
    main()
