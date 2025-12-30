#!/usr/bin/env python3
"""
Load real airline route data.

This script helps load route data that specifies which routes each airline serves.
Data sources can include:
- Bureau of Transportation Statistics (BTS) T-100 data
- OpenFlights data
- Other public airline datasets

Expected format: CSV with columns:
- origin: Origin city/airport
- destination: Destination city/airport  
- airline: Airline name
- (optional) passengers: Number of passengers
- (optional) flights: Number of flights
- (optional) distance: Route distance
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


def load_airline_routes_from_csv(filename: str, data_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load airline route data from CSV.
    
    Expected columns:
    - origin: Origin city name or IATA code
    - destination: Destination city name or IATA code
    - airline: Airline name (required for airline-specific routes)
    - passengers: Annual passengers (optional)
    - flights: Number of flights (optional)
    - distance: Route distance in km (optional)
    
    Returns:
        DataFrame with route data
    """
    if data_dir is None:
        data_dir = Path(__file__).parent / "data"
    else:
        data_dir = Path(data_dir)
    
    filepath = data_dir / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Route data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    required_cols = ['origin', 'destination', 'airline']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    return df


def get_airline_routes(route_data: pd.DataFrame, airline_name: str) -> pd.DataFrame:
    """
    Get routes for a specific airline.
    
    Args:
        route_data: DataFrame with route data
        airline_name: Name of airline to filter
    
    Returns:
        DataFrame with routes for that airline
    """
    return route_data[route_data['airline'] == airline_name].copy()


def create_airline_route_summary(route_data: pd.DataFrame) -> pd.DataFrame:
    """
    Create summary statistics by airline.
    
    Returns:
        DataFrame with columns: airline, num_routes, total_passengers (if available)
    """
    summary = route_data.groupby('airline').agg({
        'origin': 'count',  # Count routes
    }).rename(columns={'origin': 'num_routes'})
    
    if 'passengers' in route_data.columns:
        summary['total_passengers'] = route_data.groupby('airline')['passengers'].sum()
    
    if 'flights' in route_data.columns:
        summary['total_flights'] = route_data.groupby('airline')['flights'].sum()
    
    return summary.reset_index()


def create_sample_airline_routes(data_dir: Optional[Path] = None):
    """
    Create sample airline route data for testing.
    This creates realistic-looking route data for major airlines.
    """
    if data_dir is None:
        data_dir = Path(__file__).parent / "data"
    else:
        data_dir = Path(data_dir)
    
    data_dir.mkdir(exist_ok=True)
    
    # Major routes for American Airlines (hub-focused)
    american_routes = [
        # From Dallas (hub)
        ('Dallas', 'New York', 'American Airlines'),
        ('Dallas', 'Los Angeles', 'American Airlines'),
        ('Dallas', 'Chicago', 'American Airlines'),
        ('Dallas', 'Miami', 'American Airlines'),
        ('Dallas', 'Phoenix', 'American Airlines'),
        # From Miami (hub)
        ('Miami', 'New York', 'American Airlines'),
        ('Miami', 'Chicago', 'American Airlines'),
        ('Miami', 'Dallas', 'American Airlines'),
        # From Charlotte (hub)
        ('Charlotte', 'New York', 'American Airlines'),
        ('Charlotte', 'Chicago', 'American Airlines'),
        ('Charlotte', 'Dallas', 'American Airlines'),
        # From Chicago (hub)
        ('Chicago', 'New York', 'American Airlines'),
        ('Chicago', 'Los Angeles', 'American Airlines'),
        ('Chicago', 'Dallas', 'American Airlines'),
    ]
    
    # Major routes for United Airlines
    united_routes = [
        # From Chicago (hub)
        ('Chicago', 'New York', 'United Airlines'),
        ('Chicago', 'Los Angeles', 'United Airlines'),
        ('Chicago', 'San Francisco', 'United Airlines'),
        ('Chicago', 'Denver', 'United Airlines'),
        ('Chicago', 'Houston', 'United Airlines'),
        # From Denver (hub)
        ('Denver', 'New York', 'United Airlines'),
        ('Denver', 'Los Angeles', 'United Airlines'),
        ('Denver', 'Chicago', 'United Airlines'),
        ('Denver', 'San Francisco', 'United Airlines'),
        # From San Francisco (hub)
        ('San Francisco', 'New York', 'United Airlines'),
        ('San Francisco', 'Los Angeles', 'United Airlines'),
        ('San Francisco', 'Chicago', 'United Airlines'),
        ('San Francisco', 'Seattle', 'United Airlines'),
        # From Houston (hub)
        ('Houston', 'New York', 'United Airlines'),
        ('Houston', 'Chicago', 'United Airlines'),
        ('Houston', 'Los Angeles', 'United Airlines'),
    ]
    
    # Combine all routes
    all_routes = american_routes + united_routes
    
    # Create DataFrame
    routes_data = {
        'origin': [r[0] for r in all_routes],
        'destination': [r[1] for r in all_routes],
        'airline': [r[2] for r in all_routes],
    }
    
    routes_df = pd.DataFrame(routes_data)
    
    # Add some optional columns with sample data
    routes_df['passengers'] = np.random.randint(100000, 2000000, len(routes_df))
    routes_df['flights'] = np.random.randint(100, 2000, len(routes_df))
    
    # Save
    output_path = data_dir / 'airline_routes_sample.csv'
    routes_df.to_csv(output_path, index=False)
    print(f"Created sample airline routes file: {output_path}")
    print(f"  {len(routes_df)} routes")
    print(f"  Airlines: {routes_df['airline'].unique().tolist()}")
    
    # Print summary
    summary = create_airline_route_summary(routes_df)
    print("\nRoute summary:")
    print(summary.to_string(index=False))


def compare_model_vs_actual(
    model_routes: np.ndarray,  # Boolean array of selected routes
    actual_routes: pd.DataFrame,  # DataFrame with actual airline routes
    markets: np.ndarray,  # (num_markets, 2) array of [origin_idx, dest_idx]
    city_names: List[str],
    airline_name: str
) -> Dict:
    """
    Compare model predictions vs actual airline routes.
    
    Returns:
        Dictionary with comparison metrics:
        - num_model_routes: Number of routes selected by model
        - num_actual_routes: Number of routes actually served
        - overlap: Number of routes in both
        - precision: Overlap / num_model_routes
        - recall: Overlap / num_actual_routes
        - f1: 2 * precision * recall / (precision + recall)
    """
    # Get actual routes for this airline
    airline_actual = actual_routes[actual_routes['airline'] == airline_name]
    
    # Create mapping from city names to indices
    city_to_idx = {name: idx for idx, name in enumerate(city_names)}
    
    # Get actual route indices
    actual_route_indices = set()
    for _, row in airline_actual.iterrows():
        orig = row['origin']
        dest = row['destination']
        if orig in city_to_idx and dest in city_to_idx:
            orig_idx = city_to_idx[orig]
            dest_idx = city_to_idx[dest]
            # Find market index
            for market_idx, (m_orig, m_dest) in enumerate(markets):
                if m_orig == orig_idx and m_dest == dest_idx:
                    actual_route_indices.add(market_idx)
                    break
    
    # Get model route indices
    model_route_indices = set(np.where(model_routes)[0])
    
    # Compute metrics
    overlap = len(model_route_indices & actual_route_indices)
    num_model = len(model_route_indices)
    num_actual = len(actual_route_indices)
    
    precision = overlap / num_model if num_model > 0 else 0.0
    recall = overlap / num_actual if num_actual > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'num_model_routes': num_model,
        'num_actual_routes': num_actual,
        'overlap': overlap,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'model_only': num_model - overlap,
        'actual_only': num_actual - overlap,
    }


if __name__ == "__main__":
    # Create sample data
    print("Creating sample airline route data...")
    create_sample_airline_routes()
    
    print("\n" + "=" * 70)
    print("To use real airline route data:")
    print("1. Download route data from BTS, OpenFlights, or other sources")
    print("2. Format as CSV with columns: origin, destination, airline")
    print("3. Save to data/airline_routes.csv")
    print("4. Use load_airline_routes_from_csv() to load it")
    print("=" * 70)


