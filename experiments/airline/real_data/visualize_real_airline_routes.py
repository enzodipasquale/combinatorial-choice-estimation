#!/usr/bin/env python3
"""
Visualize actual routes served by real airlines vs model predictions.

This script:
1. Loads real airline route data
2. Generates model predictions
3. Compares and visualizes both
"""

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from pathlib import Path
import sys
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from bundlechoice.core import BundleChoice
from load_real_data import RealAirlineDataLoader
from integrate_real_data import create_real_airline_scenario
from load_airline_routes import load_airline_routes_from_csv, compare_model_vs_actual


def plot_comparison(
    airline_name, model_bundle, actual_routes_df, cities, markets, 
    agent_hubs, city_names, city_stats, markets_dict, ax
):
    """Plot both model predictions and actual routes for comparison."""
    # Extract model routes
    model_origins = markets[model_bundle, 0]
    model_dests = markets[model_bundle, 1]
    
    # Get actual routes for this airline
    airline_actual = actual_routes_df[actual_routes_df['airline'] == airline_name]
    city_to_idx = {name: idx for idx, name in enumerate(city_names)}
    
    # Map actual routes to market indices
    actual_route_indices = set()
    for _, row in airline_actual.iterrows():
        orig = row['origin']
        dest = row['destination']
        if orig in city_to_idx and dest in city_to_idx:
            orig_idx = city_to_idx[orig]
            dest_idx = city_to_idx[dest]
            market_key = (orig_idx, dest_idx)
            if market_key in markets_dict:
                actual_route_indices.add(markets_dict[market_key])
    
    # Plot actual routes (in blue)
    for route_idx in actual_route_indices:
        orig_idx = markets[route_idx, 0]
        dest_idx = markets[route_idx, 1]
        origin = cities[orig_idx]
        dest = cities[dest_idx]
        ax.plot(
            [origin[0], dest[0]], 
            [origin[1], dest[1]],
            color='#1976D2', 
            linewidth=1.0, 
            alpha=0.3, 
            zorder=1,
            label='Actual routes' if route_idx == list(actual_route_indices)[0] else ''
        )
    
    # Plot model routes (in green, thicker)
    for orig_idx, dest_idx in zip(model_origins, model_dests):
        origin = cities[orig_idx]
        dest = cities[dest_idx]
        ax.plot(
            [origin[0], dest[0]], 
            [origin[1], dest[1]],
            color='#2E7D32', 
            linewidth=2.0, 
            alpha=0.6, 
            zorder=2,
            label='Model routes' if orig_idx == model_origins[0] else ''
        )
    
    # Plot cities
    for i, (lon, lat) in enumerate(cities):
        city_name = city_names[i] if i < len(city_names) else f'C{i}'
        
        if agent_hubs[i]:
            color = '#D32F2F'  # Red for hubs
            size = 200
            edgewidth = 3.0
        else:
            color = '#FFA726'  # Orange for non-hubs
            size = 100
            edgewidth = 2.0
        
        ax.scatter(lon, lat, c=color, s=size, zorder=3, 
                  edgecolors='white', linewidths=edgewidth)
        
        # Add city label for hubs
        if agent_hubs[i]:
            label = city_name[:4].upper() if len(city_name) > 4 else city_name.upper()
            ax.text(lon, lat, label, fontsize=9, ha='center', va='bottom',
                   weight='bold', color='black', zorder=4,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            alpha=0.8, edgecolor='none'))
    
    # Compute comparison metrics
    model_route_indices = set(np.where(model_bundle)[0])
    overlap = len(model_route_indices & actual_route_indices)
    num_model = len(model_route_indices)
    num_actual = len(actual_route_indices)
    precision = overlap / num_model if num_model > 0 else 0.0
    recall = overlap / num_actual if num_actual > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Add title with metrics
    hub_list = [i for i, is_hub in enumerate(agent_hubs) if is_hub]
    hub_names = [city_names[i] for i in hub_list] if hub_list else []
    hub_info = f"Hubs: {', '.join(hub_names[:3])}"
    if len(hub_names) > 3:
        hub_info += f" (+{len(hub_names)-3})"
    
    title = f"{airline_name}\n"
    title += f"Model: {num_model} routes | Actual: {num_actual} routes | Overlap: {overlap}\n"
    title += f"Precision: {precision:.2%} | Recall: {recall:.2%} | F1: {f1:.2%}\n"
    title += hub_info
    
    ax.set_title(title, fontsize=14, pad=15, weight='bold')
    
    # Set map bounds
    lons = cities[:, 0]
    lats = cities[:, 1]
    lon_margin = (lons.max() - lons.min()) * 0.1
    lat_margin = (lats.max() - lats.min()) * 0.1
    
    ax.set_xlim(lons.min() - lon_margin, lons.max() + lon_margin)
    ax.set_ylim(lats.min() - lat_margin, lats.max() + lat_margin)
    
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax.set_facecolor('#FAFAFA')
    ax.set_aspect('equal', adjustable='box')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#D32F2F', edgecolor='white', label='Hub'),
        Patch(facecolor='#FFA726', edgecolor='white', label='City'),
        plt.Line2D([0], [0], color='#1976D2', linewidth=2, alpha=0.6, label='Actual routes'),
        plt.Line2D([0], [0], color='#2E7D32', linewidth=2, alpha=0.6, label='Model routes'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, 
             framealpha=0.95, edgecolor='gray')


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("=" * 70)
        print("REAL AIRLINE ROUTES vs MODEL PREDICTIONS")
        print("=" * 70)
    
    # Load real route data
    data_dir = Path(__file__).parent / "data"
    try:
        actual_routes_df = load_airline_routes_from_csv('airline_routes_real.csv', data_dir)
        if rank == 0:
            print(f"\nLoaded {len(actual_routes_df)} actual routes")
            print(f"Airlines: {actual_routes_df['airline'].value_counts().to_dict()}")
    except FileNotFoundError:
        if rank == 0:
            print("\nReal route data not found. Run fetch_airline_routes.py first.")
        return
    
    # Get cities that appear in actual routes
    all_cities_in_routes = set(actual_routes_df['origin'].unique()) | set(actual_routes_df['destination'].unique())
    
    # Load city data
    loader = RealAirlineDataLoader(data_dir=data_dir)
    loader.load_cities_from_csv('cities_major.csv')
    loader.load_airline_hubs_from_csv('airline_hubs_major.csv')
    
    # Filter to cities that appear in routes
    cities_df = loader.cities.copy()
    cities_in_data = set(cities_df['city_name'].unique())
    cities_to_use = list(all_cities_in_routes & cities_in_data)
    
    if rank == 0:
        print(f"\nCities in both datasets: {len(cities_to_use)}")
        print(f"  {sorted(cities_to_use)[:10]}...")
    
    # Filter cities
    cities_df = cities_df[cities_df['city_name'].isin(cities_to_use)]
    loader.cities = cities_df
    
    # Process data
    processed = loader.process_for_bundlechoice(
        include_all_routes=True
    )
    
    # Select 2 airlines to compare
    airline_names = ['American Airlines', 'United Airlines']
    airline_hubs = processed['airline_hubs']
    
    # Filter to only airlines we have data for
    available_airlines = set(actual_routes_df['airline'].unique())
    airline_names = [a for a in airline_names if a in available_airlines][:2]
    
    if rank == 0:
        print(f"\nComparing: {airline_names}")
    
    # Create scenario
    scenario = create_real_airline_scenario(
        data_loader=loader,
        num_agents=len(airline_names),
        theta_gs=0.15,
        temperature=1.0,
        seed=42
    )
    
    # Generate predictions
    prepared = scenario.prepare(comm=comm, timeout_seconds=120, seed=42)
    
    if rank == 0:
        generation_data = prepared.generation_data
        cities = generation_data["_metadata"]["cities"]
        markets = generation_data["_metadata"]["markets"]
        agent_hubs = generation_data["agent_data"]["hubs"]
        city_names = generation_data["_metadata"]["city_names"]
        city_stats = generation_data["_metadata"]["city_stats"]
        
        # Create market index mapping
        markets_dict = {(markets[i, 0], markets[i, 1]): i for i in range(len(markets))}
        
        # Generate model predictions
        bc = BundleChoice()
        prepared.apply(bc, comm=comm, stage="generation")
        theta = prepared.theta_star
        bc.subproblems.load()
        obs_bundles = bc.subproblems.init_and_solve(theta)
        
        # Create comparison plots
        fig, axes = plt.subplots(1, len(airline_names), figsize=(12 * len(airline_names), 10))
        if len(airline_names) == 1:
            axes = [axes]
        
        for agent_idx, airline_name in enumerate(airline_names):
            plot_comparison(
                airline_name, obs_bundles[agent_idx], actual_routes_df,
                cities, markets, agent_hubs[agent_idx], city_names, city_stats,
                markets_dict, axes[agent_idx]
            )
        
        plt.suptitle('Model Predictions vs Actual Routes', fontsize=18, weight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=2.0, w_pad=2.0)
        
        # Save
        output_dir = Path(__file__).parent / "plots"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / 'real_vs_model_routes.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\n✓ Saved comparison to '{output_path}'")
        plt.close()
        
        # Print detailed comparison
        print("\n" + "=" * 70)
        print("DETAILED COMPARISON")
        print("=" * 70)
        for agent_idx, airline_name in enumerate(airline_names):
            metrics = compare_model_vs_actual(
                obs_bundles[agent_idx], actual_routes_df, markets,
                city_names, airline_name
            )
            print(f"\n{airline_name}:")
            print(f"  Model routes: {metrics['num_model_routes']}")
            print(f"  Actual routes: {metrics['num_actual_routes']}")
            print(f"  Overlap: {metrics['overlap']}")
            print(f"  Precision: {metrics['precision']:.2%}")
            print(f"  Recall: {metrics['recall']:.2%}")
            print(f"  F1 Score: {metrics['f1']:.2%}")


if __name__ == "__main__":
    main()


