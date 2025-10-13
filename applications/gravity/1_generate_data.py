"""
Step 1: Generate country features, distances, and tariffs.

Usage:
    python 1_generate_data.py --num_countries 50
    python 1_generate_data.py --countries USA CHN JPN DEU GBR
"""
import argparse
import numpy as np
import pandas as pd
import pandas_datareader.wb as wb
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import pycountry
import time


def select_top_countries(num_countries, sort_by='gdp'):
    """Select top countries by GDP or population."""
    countries_by_gdp = [
        'USA', 'CHN', 'JPN', 'DEU', 'IND', 'GBR', 'FRA', 'ITA', 'BRA', 'CAN',
        'RUS', 'KOR', 'AUS', 'ESP', 'MEX', 'IDN', 'NLD', 'SAU', 'TUR', 'CHE',
        'POL', 'BEL', 'SWE', 'ARG', 'NOR', 'AUT', 'IRE', 'ISR', 'SGP', 'DNK',
        'ZAF', 'THA', 'MYS', 'CHL', 'NGA', 'PHL', 'PAK', 'BGD', 'EGY', 'VNM',
        'COL', 'FIN', 'ROM', 'PRT', 'CZE', 'NZL', 'GRC', 'HUN', 'KWT', 'PER'
    ]
    countries_by_pop = [
        'CHN', 'IND', 'USA', 'IDN', 'PAK', 'BRA', 'NGA', 'BGD', 'RUS', 'MEX',
        'JPN', 'ETH', 'PHL', 'EGY', 'VNM', 'COD', 'TUR', 'IRN', 'DEU', 'THA',
        'GBR', 'FRA', 'ITA', 'ZAF', 'TZA', 'MMR', 'KEN', 'KOR', 'COL', 'ESP',
        'ARG', 'DZA', 'SDN', 'UGA', 'UKR', 'CAN', 'MAR', 'POL', 'IRQ', 'AFG',
        'SAU', 'PER', 'MYS', 'UZB', 'VEN', 'NPL', 'GHA', 'YEM', 'MOZ', 'AUS'
    ]
    
    if sort_by == 'population':
        return countries_by_pop[:num_countries]
    else:
        return countries_by_gdp[:num_countries]


def fetch_country_features(countries_iso3):
    """Fetch GDP and population from World Bank."""
    print(f"Fetching data for {len(countries_iso3)} countries...")
    
    # GDP (current US$)
    gdp = wb.download(indicator='NY.GDP.MKTP.CD', country=countries_iso3, start=2022, end=2022)
    gdp = gdp.reset_index().pivot(index='country', columns='year', values='NY.GDP.MKTP.CD')[2022]
    
    # Population
    pop = wb.download(indicator='SP.POP.TOTL', country=countries_iso3, start=2022, end=2022)
    pop = pop.reset_index().pivot(index='country', columns='year', values='SP.POP.TOTL')[2022]
    
    features = pd.DataFrame({
        'gdp_billions': gdp / 1e9,
        'population_millions': pop / 1e6
    })
    
    # Fill missing values
    features['gdp_billions'] = features['gdp_billions'].fillna(features['gdp_billions'].median())
    features['population_millions'] = features['population_millions'].fillna(features['population_millions'].median())
    
    return features


def compute_distances(countries_iso3):
    """Compute pairwise distances between country capitals."""
    print("Computing distances...")
    geolocator = Nominatim(user_agent="gravity_model")
    coords = {}
    
    for iso3 in countries_iso3:
        try:
            country = pycountry.countries.get(alpha_3=iso3)
            location = geolocator.geocode(country.name, timeout=10)
            if location:
                coords[iso3] = (location.latitude, location.longitude)
            time.sleep(1)  # Rate limiting
        except:
            coords[iso3] = (0, 0)  # Fallback
    
    num_countries = len(countries_iso3)
    distances = np.zeros((num_countries, num_countries))
    
    for i, c1 in enumerate(countries_iso3):
        for j, c2 in enumerate(countries_iso3):
            if i != j:
                distances[i, j] = geodesic(coords[c1], coords[c2]).kilometers
            else:
                distances[i, j] = 0
    
    return pd.DataFrame(distances, index=countries_iso3, columns=countries_iso3)


def fetch_tariffs(countries_iso3):
    """Fetch average tariff rates and create bilateral matrix."""
    print("Fetching tariff data...")
    
    # Convert to ISO2
    iso2_codes = []
    for iso3 in countries_iso3:
        try:
            country = pycountry.countries.get(alpha_3=iso3)
            iso2_codes.append(country.alpha_2 if country else None)
        except:
            iso2_codes.append(None)
    
    # Fetch tariff data from World Bank
    tariffs = wb.download(indicator='TM.TAX.MRCH.SM.AR.ZS', country=iso2_codes, start=2020, end=2022)
    tariffs = tariffs.reset_index().groupby('country')['TM.TAX.MRCH.SM.AR.ZS'].mean()
    
    # Create bilateral matrix
    num_countries = len(countries_iso3)
    tariff_matrix = np.zeros((num_countries, num_countries))
    
    for i, iso3 in enumerate(countries_iso3):
        avg_tariff = tariffs.get(iso3, 5.0)  # Default 5%
        tariff_matrix[i, :] = avg_tariff
        tariff_matrix[i, i] = 0  # No self-tariff
    
    # Apply FTA discounts
    ftas = {
        'EU': ['AUT', 'BEL', 'BGR', 'HRV', 'CYP', 'CZE', 'DNK', 'EST', 'FIN', 'FRA', 
               'DEU', 'GRC', 'HUN', 'IRL', 'ITA', 'LVA', 'LTU', 'LUX', 'MLT', 'NLD', 
               'POL', 'PRT', 'ROU', 'SVK', 'SVN', 'ESP', 'SWE'],
        'NAFTA': ['USA', 'CAN', 'MEX'],
        'ASEAN': ['BRN', 'KHM', 'IDN', 'LAO', 'MYS', 'MMR', 'PHL', 'SGP', 'THA', 'VNM'],
    }
    
    for fta_members in ftas.values():
        members = [i for i, c in enumerate(countries_iso3) if c in fta_members]
        for i in members:
            for j in members:
                if i != j:
                    tariff_matrix[i, j] *= 0.1  # 90% reduction
    
    return pd.DataFrame(tariff_matrix, index=countries_iso3, columns=countries_iso3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_countries', type=int, default=50)
    parser.add_argument('--sort_by', choices=['gdp', 'population'], default='gdp')
    parser.add_argument('--countries', nargs='+', default=None)
    args = parser.parse_args()
    
    # Select countries
    if args.countries:
        countries = args.countries[:args.num_countries]
    else:
        countries = select_top_countries(args.num_countries, args.sort_by)
    
    print(f"\n{'='*60}")
    print(f"GENERATING DATA FOR {len(countries)} COUNTRIES")
    print(f"{'='*60}\n")
    
    # Fetch features
    features = fetch_country_features(countries)
    print(f"✓ Features: {features.shape}")
    
    # Compute distances
    distances = compute_distances(countries)
    print(f"✓ Distances: {distances.shape}")
    
    # Fetch tariffs
    tariffs = fetch_tariffs(countries)
    print(f"✓ Tariffs: {tariffs.shape}")
    
    # Save
    features.to_csv('datasets/country_features.csv')
    distances.to_csv('datasets/distances.csv')
    tariffs.to_csv('datasets/tariffs.csv')
    
    print(f"\n✓ Saved to datasets/")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

