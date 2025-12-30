#!/usr/bin/env python3
"""
Extract airline hub data from route data.

For each airline, identify hub cities as cities with:
- High number of routes originating from them
- Or cities that appear frequently as origins relative to the airline's total routes
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Optional
from collections import Counter


def identify_hubs_from_routes(
    routes_df: pd.DataFrame,
    min_routes_from_hub: int = 5,
    hub_percentile: float = 75.0
) -> pd.DataFrame:
    """
    Identify hub cities for each airline based on route patterns.
    
    A hub is defined as a city that has:
    - At least min_routes_from_hub routes originating from it
    - Or is in the top hub_percentile percentile of cities by route count for that airline
    
    Args:
        routes_df: DataFrame with columns: origin, destination, airline
        min_routes_from_hub: Minimum routes from a city to be considered a hub
        hub_percentile: Percentile threshold for hub identification
    
    Returns:
        DataFrame with columns: airline, hub_city
    """
    hub_data = []
    
    for airline in routes_df['airline'].unique():
        airline_routes = routes_df[routes_df['airline'] == airline]
        
        # Count routes from each origin city
        origin_counts = airline_routes['origin'].value_counts()
        
        # Method 1: Cities with at least min_routes_from_hub routes
        hubs_by_count = origin_counts[origin_counts >= min_routes_from_hub].index.tolist()
        
        # Method 2: Top percentile cities
        if len(origin_counts) > 0:
            threshold = np.percentile(origin_counts.values, hub_percentile)
            hubs_by_percentile = origin_counts[origin_counts >= threshold].index.tolist()
        else:
            hubs_by_percentile = []
        
        # Combine both methods (union)
        all_hubs = list(set(hubs_by_count) | set(hubs_by_percentile))
        
        # Ensure at least one hub (the top city)
        if len(all_hubs) == 0 and len(origin_counts) > 0:
            all_hubs = [origin_counts.index[0]]
        
        # Add hub data
        for hub_city in all_hubs:
            route_count = origin_counts[hub_city]
            hub_data.append({
                'airline': airline,
                'hub_city': hub_city,
                'routes_from_hub': route_count
            })
    
    return pd.DataFrame(hub_data)


def create_comprehensive_airline_data(
    routes_file: str = 'airline_routes_real.csv',
    output_hubs_file: str = 'airline_hubs_all.csv',
    data_dir: Optional[Path] = None
):
    """
    Create comprehensive airline data: routes and hubs for all airlines.
    """
    if data_dir is None:
        data_dir = Path(__file__).parent / "data"
    else:
        data_dir = Path(data_dir)
    
    print("=" * 70)
    print("EXTRACTING AIRLINE HUBS FROM ROUTE DATA")
    print("=" * 70)
    
    # Load routes
    routes_path = data_dir / routes_file
    if not routes_path.exists():
        raise FileNotFoundError(f"Routes file not found: {routes_path}")
    
    print(f"\nLoading routes from: {routes_path}")
    routes_df = pd.read_csv(routes_path)
    print(f"  Loaded {len(routes_df)} routes")
    print(f"  Airlines: {routes_df['airline'].nunique()}")
    
    # Identify hubs
    print(f"\nIdentifying hubs for all airlines...")
    print(f"  Criteria: Cities with >= 5 routes OR top 75th percentile")
    hubs_df = identify_hubs_from_routes(
        routes_df,
        min_routes_from_hub=5,
        hub_percentile=75.0
    )
    
    # Save hubs
    output_path = data_dir / output_hubs_file
    hubs_df[['airline', 'hub_city']].to_csv(output_path, index=False)
    print(f"\n✓ Saved hub data to: {output_path}")
    print(f"  Total hub assignments: {len(hubs_df)}")
    print(f"  Airlines with hubs: {hubs_df['airline'].nunique()}")
    
    # Statistics
    print("\n" + "=" * 70)
    print("HUB STATISTICS")
    print("=" * 70)
    
    hubs_per_airline = hubs_df.groupby('airline').size()
    print(f"\nHubs per airline:")
    print(f"  Mean: {hubs_per_airline.mean():.1f}")
    print(f"  Median: {hubs_per_airline.median():.1f}")
    print(f"  Min: {hubs_per_airline.min()}")
    print(f"  Max: {hubs_per_airline.max()}")
    
    print(f"\nTop 20 airlines by number of hubs:")
    top_airlines = hubs_per_airline.sort_values(ascending=False).head(20)
    for airline, num_hubs in top_airlines.items():
        print(f"  {airline}: {num_hubs} hubs")
    
    # Show sample hubs for a few airlines
    print(f"\nSample hub assignments:")
    for airline in routes_df['airline'].unique()[:5]:
        airline_hubs = hubs_df[hubs_df['airline'] == airline]['hub_city'].tolist()
        route_counts = hubs_df[hubs_df['airline'] == airline].set_index('hub_city')['routes_from_hub'].to_dict()
        hubs_str = ", ".join([f"{h}({route_counts[h]})" for h in airline_hubs[:5]])
        if len(airline_hubs) > 5:
            hubs_str += f" ... (+{len(airline_hubs)-5} more)"
        print(f"  {airline}: {hubs_str}")
    
    # Verify routes data is complete
    print(f"\n" + "=" * 70)
    print("ROUTES DATA SUMMARY")
    print("=" * 70)
    print(f"Total routes: {len(routes_df)}")
    print(f"Total airlines: {routes_df['airline'].nunique()}")
    print(f"Unique origin cities: {routes_df['origin'].nunique()}")
    print(f"Unique destination cities: {routes_df['destination'].nunique()}")
    
    routes_per_airline = routes_df.groupby('airline').size()
    print(f"\nRoutes per airline:")
    print(f"  Mean: {routes_per_airline.mean():.1f}")
    print(f"  Median: {routes_per_airline.median():.1f}")
    print(f"  Min: {routes_per_airline.min()}")
    print(f"  Max: {routes_per_airline.max()}")
    
    print(f"\n✓ Complete! You now have:")
    print(f"  - Routes data: {routes_file} ({len(routes_df)} routes, {routes_df['airline'].nunique()} airlines)")
    print(f"  - Hubs data: {output_hubs_file} ({len(hubs_df)} hub assignments)")


if __name__ == "__main__":
    create_comprehensive_airline_data()

