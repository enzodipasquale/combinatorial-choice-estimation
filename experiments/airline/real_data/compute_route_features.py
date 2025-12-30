#!/usr/bin/env python3
"""
Compute real route features for airline networks.

Features:
1. Population-weighted centroid: exp(pop_origin * pop_dest / distance) / normalization
2. (Cost feature is handled separately as congestion term)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from scipy.spatial.distance import cdist


def haversine_distance(lat1: np.ndarray, lon1: np.ndarray, 
                       lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """
    Calculate great circle distance between points on Earth.
    
    Returns distance in kilometers.
    """
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = (np.sin(dlat / 2) ** 2 + 
         np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2)
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Earth radius in km
    R = 6371.0
    
    return R * c


def compute_population_weighted_feature(
    cities: np.ndarray,
    city_stats: pd.DataFrame,
    markets: np.ndarray,
    origin_city: np.ndarray,
    dest_city: np.ndarray,
    temperature: float = 1.0
) -> np.ndarray:
    """
    Compute population-weighted centroid feature for each route.
    
    Feature = exp(pop_origin * pop_dest / distance) / sum_{dest'} exp(pop_origin * pop_dest' / distance)
    
    This is a softmax over destinations from each origin, weighted by:
    - Population product (pop_origin * pop_dest)
    - Inverse distance (1/distance)
    
    Args:
        cities: (num_cities, 2) array of [longitude, latitude]
        city_stats: DataFrame with city statistics including 'population'
        markets: (num_markets, 2) array of [origin_idx, dest_idx]
        origin_city: (num_markets,) array of origin city indices
        dest_city: (num_markets,) array of dest city indices
        temperature: Temperature parameter for softmax (default 1.0)
    
    Returns:
        (num_markets,) array of population-weighted features
    """
    num_cities = len(cities)
    num_markets = len(markets)
    
    # Get populations
    if 'population' not in city_stats.columns:
        raise ValueError("City statistics must include 'population' column")
    
    populations = city_stats['population'].values  # (num_cities,)
    
    # Compute distances between all city pairs
    # cities is [lon, lat], we need [lat, lon] for haversine
    coords = cities[:, [1, 0]]  # Swap to [lat, lon]
    distances = np.zeros((num_cities, num_cities))
    
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                dist = haversine_distance(
                    coords[i, 0], coords[i, 1],
                    coords[j, 0], coords[j, 1]
                )
                distances[i, j] = dist
            else:
                distances[i, j] = 1.0  # Avoid division by zero
    
    # Compute feature for each route
    features = np.zeros(num_markets)
    
    # Group routes by origin for efficient computation
    for origin_idx in range(num_cities):
        # Find all routes from this origin
        origin_mask = origin_city == origin_idx
        if not np.any(origin_mask):
            continue
        
        origin_pop = populations[origin_idx]
        
        # Get all destinations from this origin
        dest_indices = dest_city[origin_mask]
        route_indices = np.where(origin_mask)[0]
        
        # Compute unnormalized scores for all destinations from this origin
        raw_scores = []
        valid_dest_indices = []
        for dest_idx in range(num_cities):
            if dest_idx != origin_idx:
                dest_pop = populations[dest_idx]
                dist = distances[origin_idx, dest_idx]
                
                # Score = pop_origin * pop_dest / distance
                # Scale populations to millions to avoid overflow
                pop_origin_millions = origin_pop / 1e6
                pop_dest_millions = dest_pop / 1e6
                score = (pop_origin_millions * pop_dest_millions) / (dist + 1e-6)
                raw_scores.append(score)
                valid_dest_indices.append(dest_idx)
        
        if len(raw_scores) == 0:
            continue
        
        raw_scores = np.array(raw_scores)
        
        # Normalize scores before exponentiating for numerical stability
        # Use log-sum-exp trick: exp(x - max(x)) / sum(exp(x - max(x)))
        max_score = np.max(raw_scores)
        normalized_scores = (raw_scores - max_score) / temperature
        exp_scores = np.exp(normalized_scores)
        
        # Normalize to get probabilities
        normalization = np.sum(exp_scores)
        
        if normalization > 0:
            probabilities = exp_scores / normalization
            # Create full array for all destinations
            scores_full = np.zeros(num_cities)
            for i, dest_idx in enumerate(valid_dest_indices):
                scores_full[dest_idx] = probabilities[i]
            
            # Assign probabilities to routes
            for route_idx, dest_idx in zip(route_indices, dest_indices):
                features[route_idx] = scores_full[dest_idx]
        else:
            # If normalization is zero, assign uniform probability
            num_dests = len(dest_indices)
            if num_dests > 0:
                features[route_indices] = 1.0 / num_dests
    
    return features


def compute_route_features(
    cities: np.ndarray,
    city_stats: pd.DataFrame,
    markets: np.ndarray,
    origin_city: np.ndarray,
    dest_city: np.ndarray,
    temperature: float = 1.0
) -> np.ndarray:
    """
    Compute all route features.
    
    Currently returns:
    - Feature 1: Population-weighted centroid feature
    
    Args:
        cities: (num_cities, 2) array of [longitude, latitude]
        city_stats: DataFrame with city statistics
        markets: (num_markets, 2) array of [origin_idx, dest_idx]
        origin_city: (num_markets,) array of origin city indices
        dest_city: (num_markets,) array of dest city indices
        temperature: Temperature for softmax
    
    Returns:
        (num_markets, num_features) array of route features
    """
    # Compute population-weighted feature
    pop_feature = compute_population_weighted_feature(
        cities, city_stats, markets, origin_city, dest_city, temperature
    )
    
    # Stack features (currently just one, but can add more)
    features = pop_feature[:, None]  # (num_markets, 1)
    
    return features


if __name__ == "__main__":
    # Test with sample data
    from load_real_data import RealAirlineDataLoader
    from pathlib import Path
    
    data_dir = Path(__file__).parent / "data"
    loader = RealAirlineDataLoader(data_dir=data_dir)
    
    # Load sample data
    loader.load_cities_from_csv('cities_sample.csv')
    processed = loader.process_for_bundlechoice(include_all_routes=True)
    
    cities = processed['cities']
    city_stats = processed['city_stats']
    markets = processed['markets']
    origin_city = processed['origin_city']
    dest_city = processed['dest_city']
    
    print(f"Computing features for {len(markets)} routes...")
    features = compute_route_features(
        cities, city_stats, markets, origin_city, dest_city, temperature=1.0
    )
    
    print(f"Features shape: {features.shape}")
    print(f"Feature range: [{features.min():.6f}, {features.max():.6f}]")
    print(f"Feature mean: {features.mean():.6f}")
    print(f"\nSample features (first 10 routes):")
    for i in range(min(10, len(features))):
        orig = origin_city[i]
        dest = dest_city[i]
        orig_name = processed['city_names'][orig]
        dest_name = processed['city_names'][dest]
        print(f"  {orig_name} -> {dest_name}: {features[i, 0]:.6f}")

