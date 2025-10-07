"""
Generate gravity model data for export destination choice.
Fetches real country data including distances, GDP, population, and trade characteristics.
"""
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import pycountry
import time
import argparse


def fetch_country_metadata():
    """Fetch country names, coordinates, languages, and regions from real data sources."""
    import country_converter as coco
    
    geolocator = Nominatim(user_agent="gravity_model_generator")
    cc = coco.CountryConverter()
    
    # Fetch language data from REST Countries API
    import requests
    
    countries = {}
    for iso2 in COUNTRY_CODES:
        # Get country info from pycountry
        country = pycountry.countries.get(alpha_2=iso2)
        if not country:
            continue
            
        print(f"  Fetching metadata for {country.name}...")
        
        # Get coordinates via geocoding
        try:
            location = geolocator.geocode(country.name, timeout=10)
            lat = location.latitude if location else np.nan
            lon = location.longitude if location else np.nan
        except:
            lat, lon = np.nan, np.nan
        
        # Get region from country_converter (uses UN classifications)
        try:
            region = cc.convert(names=iso2, to='continent')
            if not region or region == 'not found':
                region = 'Unknown'
        except:
            region = 'Unknown'
        
        # Get language from REST Countries API
        try:
            response = requests.get(f'https://restcountries.com/v3.1/alpha/{iso2}')
            if response.status_code == 200:
                data = response.json()[0]
                # Get first official language ISO code
                languages = data.get('languages', {})
                lang_code = list(languages.keys())[0] if languages else 'unknown'
            else:
                lang_code = 'unknown'
        except:
            lang_code = 'unknown'
        
        countries[country.alpha_3] = {
            'iso2': iso2,
            'name': country.name,
            'lat': lat,
            'lon': lon,
            'language': lang_code,
            'region': region
        }
        
        time.sleep(1.5)  # Rate limiting for API calls
    
    return countries


def compute_distance_matrix(countries):
    """Compute pairwise distances between countries in km."""
    country_codes = list(countries.keys())
    n = len(country_codes)
    distances = np.zeros((n, n))
    
    for i, c1 in enumerate(country_codes):
        for j, c2 in enumerate(country_codes):
            if i != j:
                coord1 = (countries[c1]['lat'], countries[c1]['lon'])
                coord2 = (countries[c2]['lat'], countries[c2]['lon'])
                distances[i, j] = geodesic(coord1, coord2).kilometers
    
    return pd.DataFrame(distances, index=country_codes, columns=country_codes)


def select_top_countries(num_countries, sort_by='gdp'):
    """Select top N countries by specified indicator."""
    from pandas_datareader import wb
    
    print(f"Selecting top {num_countries} countries by {sort_by}...")
    
    # Mapping of sort criteria to WB indicators
    indicator_map = {
        'gdp': 'NY.GDP.MKTP.CD',
        'population': 'SP.POP.TOTL',
        'gdp_per_capita': 'NY.GDP.PCAP.CD',
        'trade': 'NE.TRD.GNFS.ZS',
        'exports': 'NE.EXP.GNFS.CD',
    }
    
    indicator = indicator_map.get(sort_by, 'NY.GDP.MKTP.CD')
    
    # Fetch data for all countries
    try:
        df = wb.download(indicator=indicator, country='all', start=2022, end=2023)
        if not df.empty:
            # Get most recent values and sort
            latest = df.groupby(level=0).last()
            top_countries = latest.nlargest(num_countries, indicator)
            iso_codes = [pycountry.countries.get(alpha_3=code).alpha_2 
                        for code in top_countries.index 
                        if pycountry.countries.get(alpha_3=code)]
            return iso_codes[:num_countries]
    except Exception as e:
        print(f"  Warning: Could not fetch ranking data: {e}")
    
    # Fallback to default list
    return ['US', 'CN', 'JP', 'DE', 'IN', 'GB', 'FR', 'BR', 'IT', 'CA', 'KR', 'RU', 'ES', 'AU', 'MX'][:num_countries]


def fetch_world_bank_data(countries):
    """Fetch comprehensive economic data from World Bank."""
    from pandas_datareader import wb
    
    # Comprehensive World Bank indicators
    indicators = {
        # Economic size & growth
        'gdp': 'NY.GDP.MKTP.CD',              # GDP current USD
        'gdp_growth': 'NY.GDP.MKTP.KD.ZG',    # GDP growth annual %
        'gdp_pc': 'NY.GDP.PCAP.CD',           # GDP per capita
        
        # Population & demographics
        'pop': 'SP.POP.TOTL',                 # Population
        'pop_density': 'EN.POP.DNST',         # Population density
        'urban_pop': 'SP.URB.TOTL.IN.ZS',     # Urban population %
        
        # Trade & openness
        'trade': 'NE.TRD.GNFS.ZS',            # Trade % of GDP
        'exports': 'NE.EXP.GNFS.CD',          # Exports goods/services
        'imports': 'NE.IMP.GNFS.CD',          # Imports goods/services
        'tariff': 'TM.TAX.MRCH.WM.AR.ZS',     # Tariff rate
        
        # Business environment
        'fdi_inflow': 'BX.KLT.DINV.CD.WD',    # FDI net inflows
        'inflation': 'FP.CPI.TOTL.ZG',        # Inflation consumer prices
        
        # Infrastructure & technology
        'internet': 'IT.NET.USER.ZS',         # Internet users %
        'mobile': 'IT.CEL.SETS.P2',           # Mobile subscriptions per 100
        'electric': 'EG.ELC.ACCS.ZS',         # Access to electricity %
        
        # Human capital
        'literacy': 'SE.ADT.LITR.ZS',         # Literacy rate adult
        'unemployment': 'SL.UEM.TOTL.ZS',     # Unemployment %
        
        # Market characteristics
        'household_consumption': 'NE.CON.PRVT.ZS',  # Household consumption % GDP
    }
    
    result = {}
    for country_code, info in countries.items():
        iso2 = info['iso2']
        print(f"  Fetching {country_code}...")
        
        country_data = {}
        for key, indicator in indicators.items():
            try:
                df = wb.download(indicator=indicator, country=iso2, start=2020, end=2023)
                if not df.empty:
                    country_data[key] = df[indicator].iloc[-1]
                else:
                    country_data[key] = np.nan
            except:
                country_data[key] = np.nan
        
        # Process and store
        result[country_code] = {
            'gdp_billions': country_data.get('gdp', np.nan) / 1e9 if not pd.isna(country_data.get('gdp')) else np.nan,
            'population_millions': country_data.get('pop', np.nan) / 1e6 if not pd.isna(country_data.get('pop')) else np.nan,
            'gdp_per_capita': country_data.get('gdp_pc', np.nan),
            'gdp_growth_pct': country_data.get('gdp_growth', np.nan),
            'population_density': country_data.get('pop_density', np.nan),
            'urban_population_pct': country_data.get('urban_pop', np.nan),
            'trade_openness_pct': country_data.get('trade', np.nan),
            'exports_billions': country_data.get('exports', np.nan) / 1e9 if not pd.isna(country_data.get('exports')) else np.nan,
            'imports_billions': country_data.get('imports', np.nan) / 1e9 if not pd.isna(country_data.get('imports')) else np.nan,
            'tariff_rate': country_data.get('tariff', np.nan),
            'fdi_inflow_billions': country_data.get('fdi_inflow', np.nan) / 1e9 if not pd.isna(country_data.get('fdi_inflow')) else np.nan,
            'inflation_pct': country_data.get('inflation', np.nan),
            'internet_users_pct': country_data.get('internet', np.nan),
            'mobile_per_100': country_data.get('mobile', np.nan),
            'electricity_access_pct': country_data.get('electric', np.nan),
            'literacy_rate': country_data.get('literacy', np.nan),
            'unemployment_pct': country_data.get('unemployment', np.nan),
            'household_consumption_pct': country_data.get('household_consumption', np.nan),
        }
    
    return result


def create_country_features(countries):
    """Create country-level features (modular) from real World Bank data."""
    country_codes = list(countries.keys())
    
    # Fetch real World Bank data
    wb_data = fetch_world_bank_data(countries)
    
    # Language dummies
    languages = list(set(c['language'] for c in countries.values()))
    lang_df = pd.DataFrame(0, index=country_codes, columns=languages)
    for code in country_codes:
        lang_df.loc[code, countries[code]['language']] = 1
    
    # Region dummies
    regions = list(set(c['region'] for c in countries.values()))
    region_df = pd.DataFrame(0, index=country_codes, columns=regions)
    for code in country_codes:
        region_df.loc[code, countries[code]['region']] = 1
    
    # Economic indicators from World Bank
    econ_data = {
        'gdp_billions': [wb_data[c]['gdp_billions'] for c in country_codes],
        'population_millions': [wb_data[c]['population_millions'] for c in country_codes],
        'gdp_per_capita': [wb_data[c]['gdp_per_capita'] for c in country_codes],
        'trade_openness_pct': [wb_data[c]['trade_openness_pct'] for c in country_codes],
    }
    econ_df = pd.DataFrame(econ_data, index=country_codes)
    
    # Combine all features
    features = pd.concat([econ_df, lang_df, region_df], axis=1)
    return features


def create_pairwise_features(countries, distances):
    """Create comprehensive pairwise features (quadratic)."""
    country_codes = list(countries.keys())
    n = len(country_codes)
    
    # Common language indicator
    common_lang = np.zeros((n, n))
    for i, c1 in enumerate(country_codes):
        for j, c2 in enumerate(country_codes):
            if countries[c1]['language'] == countries[c2]['language']:
                common_lang[i, j] = 1
    
    # Common region indicator  
    common_region = np.zeros((n, n))
    for i, c1 in enumerate(country_codes):
        for j, c2 in enumerate(country_codes):
            if countries[c1]['region'] == countries[c2]['region']:
                common_region[i, j] = 1
    
    # Contiguity (shared border) - approximated by very short distance
    contiguity = (distances.values < 100).astype(int)
    np.fill_diagonal(contiguity, 0)
    
    # Time zone difference (approximate from longitude)
    timezone_diff = np.zeros((n, n))
    for i, c1 in enumerate(country_codes):
        for j, c2 in enumerate(country_codes):
            lon1 = countries[c1]['lon']
            lon2 = countries[c2]['lon']
            if not (pd.isna(lon1) or pd.isna(lon2)):
                timezone_diff[i, j] = abs(lon1 - lon2) / 15  # 15 degrees per hour
    
    # Colonial ties (simplified - same language + different region suggests historical ties)
    colonial_ties = np.zeros((n, n))
    for i, c1 in enumerate(country_codes):
        for j, c2 in enumerate(country_codes):
            if i != j and countries[c1]['language'] == countries[c2]['language'] and countries[c1]['region'] != countries[c2]['region']:
                colonial_ties[i, j] = 1
    
    # Legal origin similarity (approximated by common region/language)
    legal_origin = np.zeros((n, n))
    for i, c1 in enumerate(country_codes):
        for j, c2 in enumerate(country_codes):
            # Common law countries (approximate)
            common_law = {'USA', 'GBR', 'IND', 'AUS', 'CAN'}
            if country_codes[i] in common_law and country_codes[j] in common_law:
                legal_origin[i, j] = 1
    
    return {
        'common_language': pd.DataFrame(common_lang, index=country_codes, columns=country_codes),
        'common_region': pd.DataFrame(common_region, index=country_codes, columns=country_codes),
        'contiguity': pd.DataFrame(contiguity, index=country_codes, columns=country_codes),
        'timezone_difference': pd.DataFrame(timezone_diff, index=country_codes, columns=country_codes),
        'colonial_ties': pd.DataFrame(colonial_ties, index=country_codes, columns=country_codes),
        'legal_origin_similarity': pd.DataFrame(legal_origin, index=country_codes, columns=country_codes),
    }


def main():
    print("Generating gravity model data...")
    
    # Fetch country metadata
    print("Fetching country metadata (coords, languages, regions)...")
    countries = fetch_country_metadata()
    
    # Distance matrix
    print("\nComputing distances...")
    distances = compute_distance_matrix(countries)
    distances.to_csv('datasets/distances.csv')
    print(f"  Saved distances: {distances.shape}")
    
    # Country features
    print("\nCreating country features...")
    features = create_country_features(countries)
    features.to_csv('datasets/country_features.csv')
    print(f"  Saved country features: {features.shape}")
    print(f"  Features: {list(features.columns)}")
    
    # Pairwise features
    print("\nCreating pairwise features...")
    pairwise = create_pairwise_features(countries)
    for name, df in pairwise.items():
        df.to_csv(f'datasets/{name}.csv')
        print(f"  Saved {name}: {df.shape}")
    
    # Summary
    print(f"\nCountries ({len(countries)}): {', '.join(countries.keys())}")
    print("\nData generated successfully!")


if __name__ == '__main__':
    main()
