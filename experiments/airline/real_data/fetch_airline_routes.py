#!/usr/bin/env python3
"""
Fetch real airline route data from OpenFlights.

This script downloads and processes airline route data from OpenFlights,
identifying hubs and creating clean CSV files ready for analysis.

Usage:
    python fetch_airline_routes.py

Output:
    - data/airline_routes_real.csv: Routes for all airlines
    - data/airline_hubs_all.csv: Hub cities for all airlines (auto-generated)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import urllib.request
from typing import Optional


def download_openflights_data(data_dir: Path) -> Optional[dict]:
    """Download and parse OpenFlights route data."""
    print("Downloading OpenFlights data...")
    
    routes_url = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/routes.dat"
    airlines_url = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airlines.dat"
    airports_url = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"
    
    try:
        # Download routes
        print("  Downloading routes...")
        with urllib.request.urlopen(routes_url) as response:
            routes_data = response.read().decode('utf-8')
        
        routes_lines = routes_data.strip().split('\n')
        routes = []
        for line in routes_lines:
            if not line.strip() or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) >= 4:
                try:
                    routes.append({
                        'airline_code': parts[0].strip().strip('"'),
                        'airline_id': parts[1].strip().strip('"'),
                        'origin_code': parts[2].strip().strip('"'),
                        'origin_id': parts[3].strip().strip('"'),
                        'dest_code': parts[4].strip().strip('"') if len(parts) > 4 else '',
                        'dest_id': parts[5].strip().strip('"') if len(parts) > 5 else '',
                    })
                except:
                    continue
        
        routes_df = pd.DataFrame(routes)
        
        # Download airlines mapping
        print("  Downloading airline names...")
        with urllib.request.urlopen(airlines_url) as response:
            airlines_data = response.read().decode('utf-8')
        
        airlines_lines = airlines_data.strip().split('\n')
        airline_map = {}
        for line in airlines_lines:
            if not line.strip() or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) >= 2:
                try:
                    airline_id = parts[0].strip().strip('"')
                    airline_name = parts[1].strip().strip('"')
                    airline_map[airline_id] = airline_name
                except:
                    continue
        
        # Download airports mapping
        print("  Downloading airport names...")
        with urllib.request.urlopen(airports_url) as response:
            airports_data = response.read().decode('utf-8')
        
        airports_lines = airports_data.strip().split('\n')
        airport_map = {}
        for line in airports_lines:
            if not line.strip() or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) >= 5:
                try:
                    airport_id = parts[0].strip().strip('"')
                    airport_city = parts[2].strip().strip('"')
                    airport_map[airport_id] = airport_city
                except:
                    continue
        
        # Map airline IDs to names
        routes_df['airline'] = routes_df['airline_id'].map(airline_map)
        
        # Map airport IDs to cities
        routes_df['origin_city'] = routes_df['origin_id'].map(airport_map)
        routes_df['dest_city'] = routes_df['dest_id'].map(airport_map)
        
        # Filter to valid routes
        routes_df = routes_df[
            routes_df['airline'].notna() & 
            routes_df['origin_city'].notna() & 
            routes_df['dest_city'].notna() &
            (routes_df['origin_city'] != '') &
            (routes_df['dest_city'] != '')
        ].copy()
        
        print(f"  Loaded {len(routes_df)} routes")
        return {'routes': routes_df, 'airline_map': airline_map, 'airport_map': airport_map}
        
    except Exception as e:
        print(f"Error downloading OpenFlights data: {e}")
        return None


def standardize_airline_names(routes_df: pd.DataFrame) -> pd.DataFrame:
    """Standardize airline names to consistent format."""
    def standardize(name):
        if pd.isna(name):
            return None
        name = name.strip()
        name_lower = name.lower()
        
        # Standardize major airlines
        if 'american' in name_lower and 'eagle' not in name_lower and 'connection' not in name_lower:
            return 'American Airlines'
        elif 'united' in name_lower and 'express' not in name_lower and 'connection' not in name_lower:
            return 'United Airlines'
        elif 'delta' in name_lower and 'connection' not in name_lower:
            return 'Delta Air Lines'
        elif 'southwest' in name_lower:
            return 'Southwest Airlines'
        elif 'jetblue' in name_lower:
            return 'JetBlue Airways'
        elif 'alaska' in name_lower:
            return 'Alaska Airlines'
        elif 'spirit' in name_lower:
            return 'Spirit Airlines'
        elif 'frontier' in name_lower:
            return 'Frontier Airlines'
        elif 'allegiant' in name_lower:
            return 'Allegiant Air'
        elif 'hawaiian' in name_lower:
            return 'Hawaiian Airlines'
        elif 'virgin america' in name_lower:
            return 'Virgin America'
        elif 'sun country' in name_lower:
            return 'Sun Country Airlines'
        else:
            return name
    
    routes_df['airline_standardized'] = routes_df['airline'].apply(standardize)
    return routes_df[routes_df['airline_standardized'].notna()].copy()




def main():
    """Main function to fetch and process airline route data."""
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("FETCHING REAL AIRLINE ROUTE DATA")
    print("=" * 70)
    
    # Download data
    openflights_data = download_openflights_data(data_dir)
    if openflights_data is None:
        print("\nFailed to download OpenFlights data.")
        return
    
    routes_df = openflights_data['routes']
    print(f"\nTotal routes downloaded: {len(routes_df)}")
    print(f"Unique airlines: {routes_df['airline'].nunique()}")
    
    # Standardize airline names
    print("\nStandardizing airline names...")
    routes_df = standardize_airline_names(routes_df)
    
    # Filter to airlines with enough routes
    route_counts = routes_df['airline_standardized'].value_counts()
    airlines_with_enough_routes = route_counts[route_counts >= 3].index
    routes_df = routes_df[routes_df['airline_standardized'].isin(airlines_with_enough_routes)].copy()
    
    # Create routes CSV
    output_routes = pd.DataFrame({
        'origin': routes_df['origin_city'],
        'destination': routes_df['dest_city'],
        'airline': routes_df['airline_standardized'],
    }).drop_duplicates(subset=['origin', 'destination', 'airline']).sort_values(['airline', 'origin', 'destination'])
    
    routes_path = data_dir / 'airline_routes_real.csv'
    output_routes.to_csv(routes_path, index=False)
    print(f"\n✓ Saved routes to: {routes_path}")
    print(f"  {len(output_routes)} unique routes")
    print(f"  {output_routes['airline'].nunique()} airlines")
    
    # Note: Official hubs should be fetched separately using fetch_official_hubs.py
    # This ensures we use official hub designations, not inferred ones.
    # Run: python fetch_official_hubs.py
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Routes: {len(output_routes)} routes across {output_routes['airline'].nunique()} airlines")
    print(f"\nNote: To get official hub data, run: python fetch_official_hubs.py")
    print(f"\nTop 10 airlines by routes:")
    print(output_routes['airline'].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
