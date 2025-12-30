#!/usr/bin/env python3
"""
Create realistic data for major airlines (American, United, Delta) with their real hubs.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def create_major_airlines_data():
    """Create CSV files with major US cities and airline hubs."""
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Major US cities with real coordinates and populations
    cities_data = {
        'city_name': [
            'New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
            'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose',
            'Austin', 'Jacksonville', 'San Francisco', 'Columbus', 'Fort Worth',
            'Charlotte', 'Seattle', 'Denver', 'Boston', 'El Paso',
            'Detroit', 'Nashville', 'Memphis', 'Portland', 'Oklahoma City',
            'Las Vegas', 'Louisville', 'Baltimore', 'Milwaukee', 'Albuquerque',
            'Tucson', 'Fresno', 'Sacramento', 'Kansas City', 'Mesa',
            'Atlanta', 'Miami', 'Oakland', 'Minneapolis', 'Tulsa',
            'Cleveland', 'Wichita', 'Arlington', 'New Orleans', 'Honolulu'
        ],
        'latitude': [
            40.7128, 34.0522, 41.8781, 29.7604, 33.4484,
            39.9526, 29.4241, 32.7157, 32.7767, 37.3382,
            30.2672, 30.3322, 37.7749, 39.9612, 32.7555,
            35.2271, 47.6062, 39.7392, 42.3601, 31.7619,
            42.3314, 36.1627, 35.1495, 45.5152, 35.4676,
            36.1699, 38.2527, 39.2904, 43.0389, 35.0844,
            32.2226, 36.7378, 38.5816, 39.0997, 33.4152,
            33.7490, 25.7617, 37.8044, 44.9778, 36.1540,
            41.4993, 37.6872, 32.7357, 29.9511, 21.3099
        ],
        'longitude': [
            -74.0060, -118.2437, -87.6298, -95.3698, -112.0740,
            -75.1652, -98.4936, -117.1611, -96.7970, -121.8863,
            -97.7431, -81.6557, -122.4194, -82.9988, -97.3308,
            -80.8431, -122.3321, -104.9903, -71.0589, -106.4850,
            -83.0458, -86.7816, -90.0490, -122.6784, -97.5164,
            -115.1398, -85.7585, -76.6122, -87.9065, -106.6504,
            -110.9747, -119.7871, -121.4944, -94.5786, -111.8315,
            -84.3880, -80.1918, -122.2712, -93.2650, -95.9928,
            -81.6944, -97.3301, -97.1081, -90.0715, -157.8581
        ],
        'population': [
            8336817, 3979576, 2693976, 2320268, 1680992,
            1584064, 1547253, 1423851, 1343573, 1021795,
            978908, 949611, 873965, 898553, 918915,
            885708, 753675, 727211, 692600, 681124,
            672662, 670820, 651073, 647214, 655057,
            641676, 620118, 602495, 594833, 560513,
            542629, 542107, 524943, 508090, 508958,
            506811, 442241, 433031, 429606, 413066,
            383793, 397532, 394266, 391006, 350964
        ]
    }
    
    cities_df = pd.DataFrame(cities_data)
    cities_df.to_csv(data_dir / 'cities_major.csv', index=False)
    print(f"Created cities file: {data_dir / 'cities_major.csv'}")
    print(f"  {len(cities_df)} cities")
    
    # Major airline hubs (real hub locations)
    # American Airlines: Dallas, Miami, Charlotte, Chicago, Philadelphia, Phoenix
    # United Airlines: Chicago, Denver, San Francisco, Houston, Newark, Washington Dulles
    # Delta Air Lines: Atlanta, Minneapolis, Detroit, Salt Lake City, Seattle, New York JFK
    
    hubs_data = {
        'airline': [
            # American Airlines hubs
            'American Airlines', 'American Airlines', 'American Airlines', 
            'American Airlines', 'American Airlines', 'American Airlines',
            # United Airlines hubs  
            'United Airlines', 'United Airlines', 'United Airlines',
            'United Airlines', 'United Airlines', 'United Airlines',
            # Delta Air Lines hubs
            'Delta Air Lines', 'Delta Air Lines', 'Delta Air Lines',
            'Delta Air Lines', 'Delta Air Lines', 'Delta Air Lines',
        ],
        'hub_city': [
            # American Airlines
            'Dallas', 'Miami', 'Charlotte', 'Chicago', 'Philadelphia', 'Phoenix',
            # United Airlines
            'Chicago', 'Denver', 'San Francisco', 'Houston', 'Newark', 'Washington',
            # Delta Air Lines
            'Atlanta', 'Minneapolis', 'Detroit', 'Salt Lake City', 'Seattle', 'New York',
        ]
    }
    
    # Map some hub cities to our city list (approximate matches)
    hub_mapping = {
        'Dallas': 'Dallas',
        'Miami': 'Miami',
        'Charlotte': 'Charlotte',
        'Chicago': 'Chicago',
        'Philadelphia': 'Philadelphia',
        'Phoenix': 'Phoenix',
        'Denver': 'Denver',
        'San Francisco': 'San Francisco',
        'Houston': 'Houston',
        'Newark': 'New York',  # Newark is near NYC
        'Washington': 'Baltimore',  # Close to DC
        'Atlanta': 'Atlanta',
        'Minneapolis': 'Minneapolis',
        'Detroit': 'Detroit',
        'Salt Lake City': 'Denver',  # Approximate - use Denver
        'Seattle': 'Seattle',
        'New York': 'New York',
    }
    
    # Update hub cities to match our city list
    hubs_df = pd.DataFrame(hubs_data)
    hubs_df['hub_city'] = hubs_df['hub_city'].map(hub_mapping).fillna(hubs_df['hub_city'])
    
    # Remove hubs that don't map to our cities
    valid_cities = set(cities_df['city_name'])
    hubs_df = hubs_df[hubs_df['hub_city'].isin(valid_cities)]
    
    hubs_df.to_csv(data_dir / 'airline_hubs_major.csv', index=False)
    print(f"Created airline hubs file: {data_dir / 'airline_hubs_major.csv'}")
    print(f"  {len(hubs_df)} hub assignments")
    print("\nHub assignments:")
    for airline in hubs_df['airline'].unique():
        airline_hubs = hubs_df[hubs_df['airline'] == airline]['hub_city'].tolist()
        print(f"  {airline}: {airline_hubs}")


if __name__ == "__main__":
    create_major_airlines_data()


