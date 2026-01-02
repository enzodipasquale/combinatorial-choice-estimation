#!/usr/bin/env python3
"""
Visualize real airline networks using bundlechoice.

This script:
1. Loads real airline data
2. Processes it for bundlechoice
3. Generates demand using the airline factory
4. Visualizes the networks
"""

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from pathlib import Path
import sys

# Add parent directory to path to import bundlechoice
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from bundlechoice.core import BundleChoice
from bundlechoice.scenarios import ScenarioLibrary
from load_real_data import RealAirlineDataLoader


def plot_agent_network(agent_idx, bundle, cities, markets, agent_hubs, city_names, ax, title_suffix=""):
    """Plot the network for a single agent with real city names."""
    # Extract selected markets
    selected_markets = markets[bundle]
    selected_origins = markets[bundle, 0]
    selected_dests = markets[bundle, 1]
    
    # Plot selected routes (directed edges) first (so they appear behind cities)
    for origin_idx, dest_idx in zip(selected_origins, selected_dests):
        origin = cities[origin_idx]
        dest = cities[dest_idx]
        
        # Draw arrow
        dx = dest[0] - origin[0]
        dy = dest[1] - origin[1]
        
        # Normalize for arrow head
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            dx_norm = dx / length
            dy_norm = dy / length
            
            # Shorten line to avoid arrow overlapping with nodes
            shorten = 0.12
            start_x = origin[0] + shorten * dx_norm
            start_y = origin[1] + shorten * dy_norm
            end_x = dest[0] - shorten * dx_norm
            end_y = dest[1] - shorten * dy_norm
            
            ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                       arrowprops=dict(arrowstyle='->', lw=2.0, color='#2E7D32', alpha=0.5, zorder=1))
    
    # Plot cities on top
    for i, (x, y) in enumerate(cities):
        city_name = city_names[i] if i < len(city_names) else f'C{i}'
        if agent_hubs[i]:
            color = '#D32F2F'  # Red for hubs
            size = 600
            edgewidth = 4.0
        else:
            color = '#1976D2'  # Blue for regular cities
            size = 300
            edgewidth = 2.5
        
        ax.scatter(x, y, c=color, s=size, zorder=3, edgecolors='white', linewidths=edgewidth)
        # Use city name or abbreviation
        label = city_name[:3].upper() if len(city_name) > 3 else city_name.upper()
        ax.text(x, y, label, fontsize=10, ha='center', va='center', 
                weight='bold' if agent_hubs[i] else 'normal', color='white', zorder=4)
    
    # Count routes from each hub
    hub_counts = {}
    for origin_idx in selected_origins:
        if agent_hubs[origin_idx]:
            hub_counts[origin_idx] = hub_counts.get(origin_idx, 0) + 1
    
    # Add title with statistics
    num_routes = len(selected_markets)
    hub_list = [i for i, is_hub in enumerate(agent_hubs) if is_hub]
    hub_names = [city_names[i] for i in hub_list] if hub_list else []
    hub_info = f"Hubs: {hub_names}"
    if hub_counts:
        counts_str = ", ".join([f"{city_names[h]}:{c}" for h, c in sorted(hub_counts.items())])
        hub_info += f" | Routes from hubs: [{counts_str}]"
    
    ax.set_title(f"Agent {agent_idx}{title_suffix}\n{num_routes} routes | {hub_info}", 
                 fontsize=18, pad=30, weight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.8)
    ax.set_xlabel('Longitude', fontsize=15, weight='normal')
    ax.set_ylabel('Latitude', fontsize=15, weight='normal')
    ax.tick_params(labelsize=13)
    
    # Set nice background
    ax.set_facecolor('#FAFAFA')
    
    # Add legend only on first subplot
    if agent_idx == 0:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#D32F2F', edgecolor='white', label='Hub'),
            Patch(facecolor='#1976D2', edgecolor='white', label='City'),
            Patch(facecolor='#2E7D32', edgecolor='#2E7D32', alpha=0.6, label='Route')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=14, framealpha=0.95, 
                  edgecolor='gray', fancybox=True, shadow=False)


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("Loading real airline data...")
        print("=" * 70)
    
    # Load real data
    data_dir = Path(__file__).parent / "data"
    loader = RealAirlineDataLoader(data_dir=data_dir)
    
    # For now, use sample data if real data doesn't exist
    # In production, load real data files
    try:
        loader.load_cities_from_csv('cities_sample.csv')
        # loader.load_routes_from_csv('routes_sample.csv')  # Optional
        # loader.load_airline_hubs_from_csv('airline_hubs_sample.csv')  # Optional
    except FileNotFoundError:
        if rank == 0:
            print("Real data files not found. Creating sample data...")
            loader.create_sample_data()
            loader.load_cities_from_csv('cities_sample.csv')
    
    # Process data for bundlechoice
    processed = loader.process_for_bundlechoice(
        top_n_cities=8,  # Use top 8 cities
        include_all_routes=True  # All possible routes
    )
    
    cities = processed['cities']
    city_names = processed['city_names']
    markets = processed['markets']
    origin_city = processed['origin_city']
    dest_city = processed['dest_city']
    num_cities = len(cities)
    num_items = len(markets)
    
    if rank == 0:
        print(f"Loaded {num_cities} cities")
        print(f"Generated {num_items} markets (routes)")
        print(f"Cities: {city_names}")
    
    # For now, we'll use the synthetic factory but with real city coordinates
    # TODO: Create a real-data version that uses actual city features and airline hubs
    
    # Use synthetic factory with real coordinates
    # We'll need to adapt the factory to accept real city coordinates
    # For now, let's create a simple adaptation
    
    if rank == 0:
        print("\nNote: Currently using synthetic factory with real city coordinates.")
        print("Full real-data integration coming soon.")
        print("\nTo use real data fully, we need to:")
        print("1. Map real airline hubs to agents")
        print("2. Use real city features (population, GDP, etc.)")
        print("3. Use real route characteristics")
    
    # For demonstration, we'll just show the city layout
    if rank == 0:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Plot all cities
        for i, (x, y) in enumerate(cities):
            ax.scatter(x, y, c='#1976D2', s=400, zorder=3, edgecolors='white', linewidths=2.5)
            label = city_names[i][:3].upper() if len(city_names[i]) > 3 else city_names[i].upper()
            ax.text(x, y, label, fontsize=12, ha='center', va='center', 
                    weight='normal', color='white', zorder=4)
        
        ax.set_title('Real City Locations', fontsize=18, weight='bold', pad=20)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.8)
        ax.set_xlabel('Longitude', fontsize=15)
        ax.set_ylabel('Latitude', fontsize=15)
        ax.set_facecolor('#FAFAFA')
        
        output_dir = Path(__file__).parent / "plots"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / 'real_cities_map.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\n✓ Saved city map to '{output_path}'")
        plt.close()


if __name__ == "__main__":
    main()


