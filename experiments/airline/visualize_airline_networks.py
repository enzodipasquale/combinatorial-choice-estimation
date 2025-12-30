#!/usr/bin/env python3
"""Visualize airline networks for individual agents."""
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

from bundlechoice.core import BundleChoice
from bundlechoice.factory import ScenarioLibrary


def plot_agent_network(agent_idx, bundle, cities, markets, agent_hubs, ax, title_suffix=""):
    """Plot the network for a single agent."""
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
        if agent_hubs[i]:
            color = '#D32F2F'  # Red for hubs
            size = 600
            edgewidth = 4.0
        else:
            color = '#1976D2'  # Blue for regular cities
            size = 300
            edgewidth = 2.5
        
        ax.scatter(x, y, c=color, s=size, zorder=3, edgecolors='white', linewidths=edgewidth)
        ax.text(x, y, f'{i}', fontsize=14, ha='center', va='center', 
                weight='bold' if agent_hubs[i] else 'normal', color='white', zorder=4)
    
    # Count routes from each hub
    hub_counts = {}
    for origin_idx in selected_origins:
        if agent_hubs[origin_idx]:
            hub_counts[origin_idx] = hub_counts.get(origin_idx, 0) + 1
    
    # Add title with statistics
    num_routes = len(selected_markets)
    num_hubs = np.sum(agent_hubs)
    hub_list = [i for i, is_hub in enumerate(agent_hubs) if is_hub]
    hub_info = f"Hubs: {hub_list}"
    if hub_counts:
        counts_str = ", ".join([f"C{h}:{c}" for h, c in sorted(hub_counts.items())])
        hub_info += f" | Routes from hubs: [{counts_str}]"
    
    ax.set_title(f"Agent {agent_idx}{title_suffix}\n{num_routes} routes | {hub_info}", 
                 fontsize=18, pad=30, weight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.8)
    ax.set_xlabel('X coordinate', fontsize=15, weight='normal')
    ax.set_ylabel('Y coordinate', fontsize=15, weight='normal')
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
        print("Generating airline network scenario...")
        print("=" * 70)
    
    # Create scenario with small number of cities and agents for visualization
    num_agents = 2
    num_cities = 8
    
    scenario = (
        ScenarioLibrary.airline()
        .with_dimensions(num_agents=num_agents, num_cities=num_cities)
        .with_num_modular_features(2)
        .with_theta_gs(0.3)  # Higher congestion cost for more selective networks
        .with_hub_range(min_hubs=1, max_hubs=3)
        .with_sigma(1.0)
        .with_theta(np.array([0.5, 0.5, 0.3]))  # Lower modular values for selectivity
        .build()
    )
    
    if rank == 0:
        print(f"Scenario: {num_agents} agents, {num_cities} cities")
        print(f"Number of markets (items): {num_cities * (num_cities - 1)}")
    
    # Prepare scenario
    prepared = scenario.prepare(comm=comm, timeout_seconds=60, seed=42)
    
    if rank == 0:
        print("\nGenerating bundles for agents...")
        
        # Get data
        generation_data = prepared.generation_data
        cities = generation_data["_metadata"]["cities"]
        markets = generation_data["_metadata"]["markets"]
        agent_hubs = generation_data["agent_data"]["hubs"]
        
        # Generate bundles using BundleChoice
        bc = BundleChoice()
        prepared.apply(bc, comm=comm, stage="generation")
        
        # Solve for all agents to get their bundles
        theta = prepared.theta_star
        bc.subproblems.load()
        obs_bundles = bc.subproblems.init_and_solve(theta)
        
        print(f"Generated bundles for {len(obs_bundles)} agents")
        print(f"Bundle sizes: min={np.min([np.sum(b) for b in obs_bundles])}, "
              f"max={np.max([np.sum(b) for b in obs_bundles])}, "
              f"mean={np.mean([np.sum(b) for b in obs_bundles]):.1f}")
        
        # Create figure with subplots for each agent - give more space
        n_cols = num_agents
        n_rows = 1
        
        # Much larger figure size for more space per graph - make them square and spacious
        # Use square aspect ratio: 16x16 per agent for truly square plots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16 * n_cols, 16 * n_rows))
        fig.patch.set_facecolor('white')
        
        # Handle axes array
        if num_agents == 1:
            axes = [axes]
        else:
            axes = axes if isinstance(axes, np.ndarray) else [axes]
        
        # Plot each agent's network
        for agent_idx in range(num_agents):
            bundle = obs_bundles[agent_idx]
            plot_agent_network(
                agent_idx, bundle, cities, markets, agent_hubs[agent_idx], 
                axes[agent_idx], title_suffix=""
            )
        
        # Hide unused subplots
        for idx in range(num_agents, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Airline Network Selection by Agent', fontsize=20, weight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.98], h_pad=4.0, w_pad=4.0)
        
        # Save to output folder
        import os
        output_dir = 'airline_network_plots'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'airline_networks.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"\n✓ Saved publication-quality plot to '{output_path}'")
        plt.close()  # Close to free memory
        
        # Print summary statistics
        print("\n" + "=" * 70)
        print("SUMMARY STATISTICS")
        print("=" * 70)
        for agent_idx in range(num_agents):
            bundle = obs_bundles[agent_idx]
            selected_markets = markets[bundle]
            selected_origins = markets[bundle, 0]
            
            hub_list = [i for i, is_hub in enumerate(agent_hubs[agent_idx]) if is_hub]
            hub_counts = {}
            for origin_idx in selected_origins:
                if agent_hubs[agent_idx][origin_idx]:
                    hub_counts[origin_idx] = hub_counts.get(origin_idx, 0) + 1
            
            print(f"\nAgent {agent_idx}:")
            print(f"  Hubs: {hub_list}")
            print(f"  Routes selected: {np.sum(bundle)}")
            if hub_counts:
                print(f"  Routes from hubs: {dict(hub_counts)}")


if __name__ == "__main__":
    main()

