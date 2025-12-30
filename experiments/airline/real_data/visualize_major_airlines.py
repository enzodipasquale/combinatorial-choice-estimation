#!/usr/bin/env python3
"""
Visualize routes for 2 major airlines on a map.

This script:
1. Loads real airline data (major airlines with their hubs)
2. Generates demand using bundlechoice
3. Creates a beautiful map visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from bundlechoice.core import BundleChoice
from load_real_data import RealAirlineDataLoader
from integrate_real_data import create_real_airline_scenario


def plot_airline_network_map(
    agent_idx, airline_name, bundle, cities, markets, agent_hubs, 
    city_names, city_stats, ax, title_suffix=""
):
    """Plot airline network on a geographic map."""
    # Extract selected markets
    selected_markets = markets[bundle]
    selected_origins = markets[bundle, 0]
    selected_dests = markets[bundle, 1]
    
    # Plot selected routes first (behind cities)
    for origin_idx, dest_idx in zip(selected_origins, selected_dests):
        origin = cities[origin_idx]  # [lon, lat]
        dest = cities[dest_idx]
        
        # Draw great circle route
        ax.plot(
            [origin[0], dest[0]], 
            [origin[1], dest[1]],
            color='#2E7D32', 
            linewidth=1.5, 
            alpha=0.4, 
            zorder=1,
            transform=ax.transData
        )
        # Add arrow in the middle
        mid_lon = (origin[0] + dest[0]) / 2
        mid_lat = (origin[1] + dest[1]) / 2
        dx = dest[0] - origin[0]
        dy = dest[1] - origin[1]
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            ax.annotate('', 
                       xy=(dest[0], dest[1]), 
                       xytext=(mid_lon, mid_lat),
                       arrowprops=dict(arrowstyle='->', lw=1.5, color='#2E7D32', alpha=0.6),
                       zorder=2)
    
    # Plot cities on top
    for i, (lon, lat) in enumerate(cities):
        city_name = city_names[i] if i < len(city_names) else f'C{i}'
        
        if agent_hubs[i]:
            color = '#D32F2F'  # Red for hubs
            size = 150
            edgewidth = 2.5
            marker = 'o'
        else:
            color = '#1976D2'  # Blue for regular cities
            size = 80
            edgewidth = 1.5
            marker = 'o'
        
        ax.scatter(lon, lat, c=color, s=size, zorder=3, 
                  edgecolors='white', linewidths=edgewidth, marker=marker)
        
        # Add city label (abbreviated)
        if agent_hubs[i] or len(city_names) <= 20:  # Show all if few cities, or hubs
            label = city_name[:4].upper() if len(city_name) > 4 else city_name.upper()
            ax.text(lon, lat, label, fontsize=8, ha='center', va='bottom',
                   weight='bold' if agent_hubs[i] else 'normal', 
                   color='black', zorder=4, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
    
    # Count routes from each hub
    hub_counts = {}
    for origin_idx in selected_origins:
        if agent_hubs[origin_idx]:
            hub_counts[origin_idx] = hub_counts.get(origin_idx, 0) + 1
    
    # Add title
    num_routes = len(selected_markets)
    hub_list = [i for i, is_hub in enumerate(agent_hubs) if is_hub]
    hub_names = [city_names[i] for i in hub_list] if hub_list else []
    hub_info = f"Hubs: {', '.join(hub_names[:4])}"  # Limit to first 4
    if len(hub_names) > 4:
        hub_info += f" (+{len(hub_names)-4} more)"
    
    if hub_counts:
        counts_str = ", ".join([f"{city_names[h]}:{c}" for h, c in sorted(list(hub_counts.items())[:3])])
        if len(hub_counts) > 3:
            counts_str += "..."
        hub_info += f"\nRoutes from hubs: [{counts_str}]"
    
    ax.set_title(f"{airline_name}{title_suffix}\n{num_routes} routes | {hub_info}", 
                 fontsize=16, pad=20, weight='bold')
    
    # Set map bounds with some padding
    lons = cities[:, 0]
    lats = cities[:, 1]
    lon_margin = (lons.max() - lons.min()) * 0.1
    lat_margin = (lats.max() - lats.min()) * 0.1
    
    ax.set_xlim(lons.min() - lon_margin, lons.max() + lon_margin)
    ax.set_ylim(lats.min() - lat_margin, lats.max() + lat_margin)
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_facecolor('#F5F5F5')
    ax.set_aspect('equal', adjustable='box')
    
    # Add legend only on first subplot
    if agent_idx == 0:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#D32F2F', edgecolor='white', label='Hub City'),
            Patch(facecolor='#1976D2', edgecolor='white', label='City'),
            Patch(facecolor='#2E7D32', edgecolor='#2E7D32', alpha=0.6, label='Route')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=11, 
                 framealpha=0.95, edgecolor='gray', fancybox=True)


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("=" * 70)
        print("MAJOR AIRLINES NETWORK VISUALIZATION")
        print("=" * 70)
        print("\nLoading major airline data...")
    
    # Create major airlines data if needed
    data_dir = Path(__file__).parent / "data"
    from create_major_airlines_data import create_major_airlines_data
    
    cities_file = data_dir / 'cities_major.csv'
    hubs_file = data_dir / 'airline_hubs_major.csv'
    
    if not cities_file.exists() or not hubs_file.exists():
        if rank == 0:
            print("Creating major airlines data files...")
        create_major_airlines_data()
    
    # Load data
    loader = RealAirlineDataLoader(data_dir=data_dir)
    loader.load_cities_from_csv('cities_major.csv')
    loader.load_airline_hubs_from_csv('airline_hubs_major.csv')
    
    # Process data - use top cities for manageable size
    processed = loader.process_for_bundlechoice(
        top_n_cities=20,  # Use top 20 cities
        include_all_routes=True
    )
    
    if rank == 0:
        print(f"\nLoaded {len(processed['city_names'])} cities")
        print(f"Generated {len(processed['markets'])} markets")
        print(f"\nCities: {', '.join(processed['city_names'][:10])}...")
    
    # Select 2 airlines to visualize
    airline_hubs = processed['airline_hubs']
    airline_names = list(airline_hubs.keys())[:2]  # First 2 airlines
    
    if rank == 0:
        print(f"\nVisualizing networks for: {airline_names}")
    
    # Create scenario for 2 agents (one per airline)
    scenario = create_real_airline_scenario(
        data_loader=loader,
        num_agents=2,
        theta_gs=0.15,  # Moderate congestion cost
        temperature=1.0,
        seed=42
    )
    
    # Prepare and generate bundles
    prepared = scenario.prepare(comm=comm, timeout_seconds=120, seed=42)
    
    if rank == 0:
        print("\nGenerating route selections...")
        
        # Get data
        generation_data = prepared.generation_data
        cities = generation_data["_metadata"]["cities"]
        markets = generation_data["_metadata"]["markets"]
        agent_hubs = generation_data["agent_data"]["hubs"]
        city_names = generation_data["_metadata"]["city_names"]
        city_stats = generation_data["_metadata"]["city_stats"]
        
        # Generate bundles
        bc = BundleChoice()
        prepared.apply(bc, comm=comm, stage="generation")
        
        theta = prepared.theta_star
        bc.subproblems.load()
        obs_bundles = bc.subproblems.init_and_solve(theta)
        
        print(f"Generated bundles for {len(obs_bundles)} airlines")
        print(f"Bundle sizes: min={np.min([np.sum(b) for b in obs_bundles])}, "
              f"max={np.max([np.sum(b) for b in obs_bundles])}, "
              f"mean={np.mean([np.sum(b) for b in obs_bundles]):.1f}")
        
        # Create figure with map
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        fig.patch.set_facecolor('white')
        
        # Plot each airline
        for agent_idx in range(2):
            airline_name = airline_names[agent_idx] if agent_idx < len(airline_names) else f"Airline {agent_idx}"
            bundle = obs_bundles[agent_idx]
            
            plot_airline_network_map(
                agent_idx, airline_name, bundle, cities, markets, 
                agent_hubs[agent_idx], city_names, city_stats,
                axes[agent_idx], title_suffix=""
            )
        
        plt.suptitle('Major Airlines Network Selection', fontsize=20, weight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=3.0, w_pad=3.0)
        
        # Save
        output_dir = Path(__file__).parent / "plots"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / 'major_airlines_networks.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\n✓ Saved map visualization to '{output_path}'")
        plt.close()
        
        # Print summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        for agent_idx in range(2):
            airline_name = airline_names[agent_idx] if agent_idx < len(airline_names) else f"Airline {agent_idx}"
            bundle = obs_bundles[agent_idx]
            selected_origins = markets[bundle, 0]
            
            hub_list = [i for i, is_hub in enumerate(agent_hubs[agent_idx]) if is_hub]
            hub_names = [city_names[i] for i in hub_list]
            hub_counts = {}
            for origin_idx in selected_origins:
                if agent_hubs[agent_idx][origin_idx]:
                    hub_counts[origin_idx] = hub_counts.get(origin_idx, 0) + 1
            
            print(f"\n{airline_name}:")
            print(f"  Hubs: {hub_names}")
            print(f"  Routes selected: {np.sum(bundle)}")
            if hub_counts:
                counts_dict = {city_names[h]: c for h, c in hub_counts.items()}
                print(f"  Routes from hubs: {counts_dict}")


if __name__ == "__main__":
    main()


