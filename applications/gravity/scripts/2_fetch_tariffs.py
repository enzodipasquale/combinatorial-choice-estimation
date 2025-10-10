"""
Fetch/generate bilateral tariff data.
Uses real trade agreements and WTO status to create realistic tariff matrices.
"""
import numpy as np
import pandas as pd
from pandas_datareader import wb
import pycountry

def get_average_tariffs(country_codes):
    """Fetch average applied tariff rates from World Bank."""
    print("Fetching average tariff rates from World Bank...")
    
    avg_tariffs = {}
    
    for iso2 in country_codes:
        if iso2 is None:
            continue
        try:
            # Weighted mean applied tariff (all products)
            df = wb.download(indicator='TM.TAX.MRCH.WM.AR.ZS', country=iso2, start=2018, end=2023)
            if not df.empty:
                vals = df['TM.TAX.MRCH.WM.AR.ZS'].dropna()
                if len(vals) > 0:
                    avg_tariffs[iso2] = vals.iloc[-1]
        except:
            pass
    
    print(f"  ✓ Found tariff data for {len(avg_tariffs)}/{len([c for c in country_codes if c])} countries")
    return avg_tariffs


def get_trade_agreements(countries_iso2):
    """
    Dynamically determine trade agreement memberships based on country list.
    Only includes agreements where we have member countries in our dataset.
    """
    # Complete FTA member lists (comprehensive)
    ALL_AGREEMENTS = {
        'EU': ['AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR', 'DE', 'GR', 'HU', 
               'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL', 'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE'],
        'NAFTA_USMCA': ['US', 'CA', 'MX'],
        'ASEAN': ['ID', 'TH', 'PH', 'VN', 'SG', 'MY', 'MM', 'KH', 'LA', 'BN'],
        'MERCOSUR': ['BR', 'AR', 'UY', 'PY', 'VE'],
        'RCEP': ['CN', 'JP', 'KR', 'AU', 'NZ', 'ID', 'TH', 'PH', 'VN', 'SG', 'MY', 'MM', 'KH', 'LA', 'BN'],
        'CPTPP': ['JP', 'CA', 'AU', 'NZ', 'SG', 'MY', 'VN', 'MX', 'PE', 'CL', 'BN'],
        'EFTA': ['CH', 'NO', 'IS', 'LI'],
        'GCC': ['SA', 'AE', 'KW', 'QA', 'BH', 'OM'],
        'SADC': ['ZA', 'BW', 'LS', 'NA', 'MZ', 'MW', 'MU', 'SZ', 'TZ', 'ZM', 'ZW', 'AO', 'CD', 'MG', 'SC'],
        'USAN': ['CL', 'CO', 'EC', 'PE'],  # Pacific Alliance
        'CAFTA': ['US', 'CR', 'DO', 'SV', 'GT', 'HN', 'NI'],
        'AFCFTA': ['ZA', 'NG', 'EG', 'KE', 'GH', 'TZ', 'UG', 'CI', 'CM', 'RW'],  # African Continental FTA
    }
    
    # Filter to only include agreements where we have at least 2 members
    valid_countries = set([c for c in countries_iso2 if c is not None])
    active_agreements = {}
    
    for fta_name, all_members in ALL_AGREEMENTS.items():
        present_members = [c for c in all_members if c in valid_countries]
        if len(present_members) >= 2:
            active_agreements[fta_name] = present_members
    
    print(f"\nDetected trade agreements (from our {len(valid_countries)} countries):")
    for fta, members in active_agreements.items():
        print(f"  {fta}: {len(members)} members - {members[:5]}{'...' if len(members) > 5 else ''}")
    
    return active_agreements


def create_bilateral_tariffs(country_iso3_codes, avg_tariffs_iso2):
    """
    Create realistic bilateral tariff matrix dynamically.
    
    Tariff from country i to country j depends on:
    - Trade agreements (FTA = 0-2% tariff)
    - WTO membership (MFN = average tariff)
    """
    countries_iso2 = []
    for iso3 in country_iso3_codes:
        try:
            country = pycountry.countries.get(alpha_3=iso3)
            countries_iso2.append(country.alpha_2 if country else None)
        except:
            countries_iso2.append(None)
    
    n = len(country_iso3_codes)
    tariffs = np.zeros((n, n))
    
    # Get trade agreements dynamically based on our countries
    agreements = get_trade_agreements(countries_iso2)
    
    print("\nConstructing bilateral tariff matrix...")
    
    for i in range(n):
        iso2_i = countries_iso2[i]
        if iso2_i is None:
            continue
            
        for j in range(n):
            iso2_j = countries_iso2[j]
            if iso2_j is None:
                continue
            
            if i == j:
                tariffs[i, j] = 0  # No tariff for domestic trade
                continue
            
            # Check if in same FTA
            in_fta = False
            for fta_name, members in agreements.items():
                if iso2_i in members and iso2_j in members:
                    # FTA members: very low tariffs (0-2%)
                    tariffs[i, j] = np.random.uniform(0, 2)
                    in_fta = True
                    break
            
            if not in_fta:
                # Not in FTA: use average applied tariff of destination country
                if iso2_j in avg_tariffs_iso2:
                    # MFN tariff (WTO most-favored-nation)
                    base_tariff = avg_tariffs_iso2[iso2_j]
                    # Add small random variation
                    tariffs[i, j] = base_tariff * np.random.uniform(0.9, 1.1)
                else:
                    # No data: use global average ~7%
                    tariffs[i, j] = np.random.uniform(5, 10)
    
    return tariffs, countries_iso2


def analyze_tariff_structure(tariffs, country_names, countries_iso2, agreements):
    """Analyze the tariff matrix."""
    print(f"\n{'='*60}")
    print("TARIFF STRUCTURE ANALYSIS")
    print(f"{'='*60}")
    
    n = len(country_names)
    
    # Overall statistics
    off_diag = tariffs[~np.eye(n, dtype=bool)]
    print(f"\nGlobal tariff statistics:")
    print(f"  Mean: {off_diag.mean():.2f}%")
    print(f"  Median: {np.median(off_diag):.2f}%")
    print(f"  Range: {off_diag.min():.2f}% - {off_diag.max():.2f}%")
    
    # FTA effects
    print(f"\nTrade agreement effects:")
    for fta_name, members in agreements.items():
        # Get indices of FTA members
        member_indices = [i for i, iso2 in enumerate(countries_iso2) if iso2 in members]
        if len(member_indices) >= 2:
            # Intra-FTA tariffs
            intra_tariffs = []
            for i in member_indices:
                for j in member_indices:
                    if i != j:
                        intra_tariffs.append(tariffs[i, j])
            
            if intra_tariffs:
                print(f"  {fta_name}: {len(member_indices)} members, avg tariff {np.mean(intra_tariffs):.2f}%")
    
    # Show sample bilateral tariffs
    print(f"\nSample bilateral tariffs:")
    for i in range(min(3, n)):
        print(f"  From {country_names[i]}:")
        for j in range(min(5, n)):
            if i != j:
                print(f"    → {country_names[j]}: {tariffs[i, j]:.2f}%")


def main():
    # Load country data
    features = pd.read_csv('data/features/country_features.csv', index_col=0)
    country_codes = features.index.tolist()
    
    print("="*60)
    print(f"GENERATING BILATERAL TARIFF DATA: {len(country_codes)} countries")
    print("="*60)
    
    # Convert to ISO2
    countries_iso2 = []
    for iso3 in country_codes:
        try:
            country = pycountry.countries.get(alpha_3=iso3)
            countries_iso2.append(country.alpha_2 if country else None)
        except:
            countries_iso2.append(None)
    
    # Fetch average tariffs
    avg_tariffs = get_average_tariffs(countries_iso2)
    
    # Create bilateral matrix
    tariffs, iso2_list = create_bilateral_tariffs(country_codes, avg_tariffs)
    
    # Analyze
    # agreements already returned from create_bilateral_tariffs, get them again
    valid_iso2 = [c for c in iso2_list if c is not None]
    agreements = get_trade_agreements(valid_iso2)
    analyze_tariff_structure(tariffs, country_codes, iso2_list, agreements)
    
    # Save
    df = pd.DataFrame(tariffs, index=country_codes, columns=country_codes)
    df.to_csv('data/features/bilateral_tariffs.csv')
    
    print(f"\n✓ Saved bilateral tariff matrix to data/bilateral_tariffs.csv")
    print(f"  Shape: {df.shape}")
    print(f"\n" + "="*60)


if __name__ == '__main__':
    main()
