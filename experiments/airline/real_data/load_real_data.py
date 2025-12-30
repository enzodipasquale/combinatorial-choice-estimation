#!/usr/bin/env python3
"""
Load and process real airline data for bundlechoice analysis.

This script loads real data on:
- Airline routes between cities
- City statistics (population, coordinates, etc.)
- Airline hub locations

Data sources can include:
- OpenFlights data
- Bureau of Transportation Statistics (BTS)
- Other public airline datasets
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json


class RealAirlineDataLoader:
    """Load and process real airline network data."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize data loader.
        
        Args:
            data_dir: Directory containing data files. If None, uses ./data/
        """
        if data_dir is None:
            data_dir = Path(__file__).parent / "data"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.cities = None
        self.routes = None
        self.airlines = None
        self.city_stats = None
    
    def load_cities_from_csv(self, filename: str) -> pd.DataFrame:
        """
        Load city data from CSV.
        
        Expected columns:
        - city_name: Name of city
        - iata_code: Airport IATA code (optional)
        - latitude: Latitude coordinate
        - longitude: Longitude coordinate
        - population: City population (optional)
        - gdp: City GDP (optional)
        - country: Country name (optional)
        """
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"City data file not found: {filepath}")
        
        df = pd.read_csv(filepath)
        required_cols = ['city_name', 'latitude', 'longitude']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        self.cities = df
        return df
    
    def load_routes_from_csv(self, filename: str) -> pd.DataFrame:
        """
        Load route data from CSV.
        
        Expected columns:
        - origin: Origin city name or IATA code
        - destination: Destination city name or IATA code
        - airline: Airline name (optional)
        - distance: Route distance in km (optional)
        - passengers: Annual passengers (optional)
        """
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Route data file not found: {filepath}")
        
        df = pd.read_csv(filepath)
        required_cols = ['origin', 'destination']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        self.routes = df
        return df
    
    def load_airline_hubs_from_csv(self, filename: str) -> pd.DataFrame:
        """
        Load airline hub data from CSV.
        
        Expected columns:
        - airline: Airline name
        - hub_city: City name or IATA code where airline has a hub
        """
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Airline hub data file not found: {filepath}")
        
        df = pd.read_csv(filepath)
        required_cols = ['airline', 'hub_city']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        self.airlines = df
        return df
    
    def process_for_bundlechoice(
        self,
        min_city_population: Optional[int] = None,
        top_n_cities: Optional[int] = None,
        include_all_routes: bool = True
    ) -> Dict:
        """
        Process real data into bundlechoice format.
        
        Args:
            min_city_population: Minimum city population to include
            top_n_cities: If specified, only include top N cities by population
            include_all_routes: If True, include all possible routes (not just observed)
        
        Returns:
            Dictionary with processed data ready for bundlechoice:
            - cities: (num_cities, 2) array of [longitude, latitude] or [x, y]
            - city_names: List of city names
            - markets: (num_markets, 2) array of [origin_idx, dest_idx]
            - origin_city: (num_markets,) array of origin city indices
            - dest_city: (num_markets,) array of dest city indices
            - airline_hubs: Dict mapping airline_name -> list of hub city indices
            - city_stats: DataFrame with city statistics
        """
        if self.cities is None:
            raise ValueError("Cities data not loaded. Call load_cities_from_csv first.")
        
        # Filter cities
        cities_df = self.cities.copy()
        
        if min_city_population is not None and 'population' in cities_df.columns:
            cities_df = cities_df[cities_df['population'] >= min_city_population]
        
        if top_n_cities is not None and 'population' in cities_df.columns:
            cities_df = cities_df.nlargest(top_n_cities, 'population')
        
        cities_df = cities_df.reset_index(drop=True)
        self.city_stats = cities_df
        
        # Create city index mapping
        city_to_idx = {row['city_name']: idx for idx, row in cities_df.iterrows()}
        num_cities = len(cities_df)
        
        # Extract coordinates (use longitude, latitude for geographic projection)
        # Or convert to x, y if needed
        if 'longitude' in cities_df.columns and 'latitude' in cities_df.columns:
            # Use lat/lon directly (can be projected later if needed)
            cities_coords = cities_df[['longitude', 'latitude']].values
        else:
            raise ValueError("Cities must have longitude and latitude columns")
        
        # Process routes
        if include_all_routes:
            # Generate all possible routes (like synthetic case)
            markets = []
            origin_city = []
            dest_city = []
            for i in range(num_cities):
                for j in range(num_cities):
                    if i != j:
                        markets.append([i, j])
                        origin_city.append(i)
                        dest_city.append(j)
            markets = np.array(markets)
            origin_city = np.array(origin_city)
            dest_city = np.array(dest_city)
        else:
            # Use only observed routes
            if self.routes is None:
                raise ValueError("Routes data not loaded for observed routes mode.")
            
            markets = []
            origin_city = []
            dest_city = []
            for _, row in self.routes.iterrows():
                orig = row['origin']
                dest = row['destination']
                if orig in city_to_idx and dest in city_to_idx:
                    orig_idx = city_to_idx[orig]
                    dest_idx = city_to_idx[dest]
                    if orig_idx != dest_idx:  # No self-loops
                        markets.append([orig_idx, dest_idx])
                        origin_city.append(orig_idx)
                        dest_city.append(dest_idx)
            
            # Remove duplicates
            markets_array = np.array(markets)
            unique_markets = np.unique(markets_array, axis=0)
            markets = unique_markets
            origin_city = markets[:, 0]
            dest_city = markets[:, 1]
        
        # Process airline hubs
        airline_hubs = {}
        if self.airlines is not None:
            for _, row in self.airlines.iterrows():
                airline = row['airline']
                hub_city = row['hub_city']
                if hub_city in city_to_idx:
                    hub_idx = city_to_idx[hub_city]
                    if airline not in airline_hubs:
                        airline_hubs[airline] = []
                    airline_hubs[airline].append(hub_idx)
        
        return {
            'cities': cities_coords,
            'city_names': cities_df['city_name'].tolist(),
            'markets': markets,
            'origin_city': origin_city,
            'dest_city': dest_city,
            'airline_hubs': airline_hubs,
            'city_stats': cities_df,
        }
    
    def create_sample_data(self, output_dir: Optional[Path] = None):
        """
        Create sample data files for testing.
        This creates example CSV files that can be replaced with real data.
        """
        if output_dir is None:
            output_dir = self.data_dir
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Sample cities (major US cities)
        cities_data = {
            'city_name': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
                         'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose'],
            'latitude': [40.7128, 34.0522, 41.8781, 29.7604, 33.4484,
                        39.9526, 29.4241, 32.7157, 32.7767, 37.3382],
            'longitude': [-74.0060, -118.2437, -87.6298, -95.3698, -112.0740,
                          -75.1652, -98.4936, -117.1611, -96.7970, -121.8863],
            'population': [8336817, 3979576, 2693976, 2320268, 1680992,
                          1584064, 1547253, 1423851, 1343573, 1021795],
        }
        cities_df = pd.DataFrame(cities_data)
        cities_df.to_csv(output_dir / 'cities_sample.csv', index=False)
        print(f"Created sample cities file: {output_dir / 'cities_sample.csv'}")
        
        # Sample routes (some major routes)
        routes_data = {
            'origin': ['New York', 'Los Angeles', 'Chicago', 'New York', 'Los Angeles'],
            'destination': ['Los Angeles', 'Chicago', 'Houston', 'Chicago', 'San Francisco'],
        }
        routes_df = pd.DataFrame(routes_data)
        routes_df.to_csv(output_dir / 'routes_sample.csv', index=False)
        print(f"Created sample routes file: {output_dir / 'routes_sample.csv'}")
        
        # Sample airline hubs
        hubs_data = {
            'airline': ['American Airlines', 'United Airlines', 'Delta Air Lines',
                       'American Airlines', 'United Airlines'],
            'hub_city': ['Dallas', 'Chicago', 'Atlanta', 'Miami', 'San Francisco'],
        }
        hubs_df = pd.DataFrame(hubs_data)
        hubs_df.to_csv(output_dir / 'airline_hubs_sample.csv', index=False)
        print(f"Created sample airline hubs file: {output_dir / 'airline_hubs_sample.csv'}")


if __name__ == "__main__":
    # Example usage
    loader = RealAirlineDataLoader()
    
    # Create sample data files
    print("Creating sample data files...")
    loader.create_sample_data()
    
    print("\nSample data files created in data/ directory.")
    print("Replace these with your real data files using the same format.")


