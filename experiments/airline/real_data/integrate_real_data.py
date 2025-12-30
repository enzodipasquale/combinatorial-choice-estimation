#!/usr/bin/env python3
"""
Integrate real airline data with bundlechoice framework.

This script creates a custom scenario using real:
- City locations and statistics
- Airline hub locations
- Route characteristics

It adapts the airline factory to work with real data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from mpi4py import MPI
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from bundlechoice.core import BundleChoice
from bundlechoice.factory.base import FeatureSpec, SyntheticScenario
from bundlechoice.factory.data_generator import DataGenerator
from load_real_data import RealAirlineDataLoader
from compute_route_features import compute_route_features


def create_real_airline_scenario(
    data_loader: RealAirlineDataLoader,
    num_agents: int,
    theta_gs: float = 0.1,
    sigma: float = 1.0,
    temperature: float = 1.0,
    seed: Optional[int] = None
) -> SyntheticScenario:
    """
    Create a bundlechoice scenario from real airline data.
    
    Args:
        data_loader: RealAirlineDataLoader with loaded data
        num_agents: Number of agents (airlines) to simulate
        num_modular_features: Number of modular features per route
        theta_gs: Gross substitutability parameter
        sigma: Error standard deviation
        seed: Random seed
    
    Returns:
        SyntheticScenario ready for bundlechoice
    """
    processed = data_loader.process_for_bundlechoice(
        include_all_routes=True
    )
    
    cities = processed['cities']
    city_names = processed['city_names']
    markets = processed['markets']
    origin_city = processed['origin_city']
    dest_city = processed['dest_city']
    airline_hubs = processed['airline_hubs']
    city_stats = processed['city_stats']
    
    num_cities = len(cities)
    num_items = len(markets)
    
    # Compute real route features (population-weighted centroid)
    route_features = compute_route_features(
        cities, city_stats, markets, origin_city, dest_city, temperature
    )
    num_modular_features = route_features.shape[1]  # Should be 1 (population-weighted feature)
    
    def config_factory() -> Dict:
        return {
            "dimensions": {
                "num_agents": num_agents,
                "num_items": num_items,
                "num_features": num_modular_features + 1,  # modular + congestion
                "num_simuls": 1,
            },
            "subproblem": {"name": "Greedy"},
            "row_generation": {
                "max_iters": 100,
                "tolerance_optimality": 0.001,
                "min_iters": 1,
                "gurobi_settings": {"OutputFlag": 0},
                "theta_ubs": 100,
            },
        }
    
    def features_oracle(agent_idx: int, bundles: np.ndarray, data: Dict) -> np.ndarray:
        """Compute features using real city data."""
        modular_item = data["item_data"]["modular"]
        origin_city_arr = data["item_data"]["origin_city"]
        agent_hubs = data["agent_data"]["hubs"][agent_idx]
        
        if bundles.ndim == 1:
            modular_features = modular_item.T @ bundles
            bundle_markets = bundles.astype(bool)
            congestion = 0.0
            for h in range(len(agent_hubs)):
                if agent_hubs[h]:
                    count = np.sum((origin_city_arr[bundle_markets] == h))
                    congestion += count ** 2
            return np.concatenate((modular_features, [-congestion]))
        else:
            modular_features = modular_item.T @ bundles
            num_bundles = bundles.shape[1]
            congestion = np.zeros(num_bundles)
            for h in range(len(agent_hubs)):
                if agent_hubs[h]:
                    hub_mask = (origin_city_arr == h)[:, None]
                    counts = np.sum(bundles * hub_mask, axis=0)
                    congestion += counts ** 2
            return np.concatenate((modular_features, -congestion[None, :]), axis=0)
    
    def feature_initializer(bc: BundleChoice) -> None:
        bc.features.set_oracle(features_oracle)
        # Install find_best_item if needed
        if bc.subproblems.demand_oracle is not None:
            from bundlechoice.subproblems.registry.greedy import GreedySubproblem
            if isinstance(bc.subproblems.demand_oracle, GreedySubproblem):
                _install_find_best_item(bc.subproblems.demand_oracle)
    
    feature_spec = FeatureSpec(mode="oracle", initializer=feature_initializer)
    
    def payload_factory(
        bc: BundleChoice,
        comm: MPI.Comm,
        spec: FeatureSpec,
        timeout: Optional[int],
        seed: Optional[int],
        theta: np.ndarray,
    ) -> Dict:
        rank = comm.Get_rank()
        generator = DataGenerator(seed=seed)
        
        generation_data = None
        if rank == 0:
            # Generate hubs for each agent
            # Use real airline hubs if available, otherwise assign randomly
            agent_hubs = []
            airline_names = list(airline_hubs.keys()) if airline_hubs else []
            
            for agent_idx in range(num_agents):
                if agent_idx < len(airline_names):
                    # Use real airline hubs
                    airline = airline_names[agent_idx]
                    real_hub_indices = airline_hubs[airline]
                    hub_array = np.zeros(num_cities, dtype=bool)
                    for hub_idx in real_hub_indices:
                        if hub_idx < num_cities:
                            hub_array[hub_idx] = True
                    agent_hubs.append(hub_array)
                else:
                    # Randomly assign hubs for additional agents
                    num_hubs = generator.rng.integers(1, min(4, num_cities))
                    hubs = generator.rng.choice(num_cities, size=num_hubs, replace=False)
                    hub_array = np.zeros(num_cities, dtype=bool)
                    hub_array[hubs] = True
                    agent_hubs.append(hub_array)
            
            agent_hubs = np.array(agent_hubs)
            
            # Use real route features (population-weighted centroid)
            # route_features is already computed above: (num_items, num_modular_features)
            modular_item = route_features  # Use real features instead of synthetic
            
            # Generate errors
            errors = generator.generate_errors(
                (num_agents, num_items), sigma
            )
            
            generation_data = {
                "agent_data": {"hubs": agent_hubs},
                "item_data": {
                    "modular": modular_item,
                    "origin_city": origin_city,
                    "dest_city": dest_city,
                },
                "errors": errors,
                "_metadata": {
                    "markets": markets,
                    "cities": cities,
                    "city_names": city_names,
                    "city_stats": city_stats,
                } if rank == 0 else None,
            }
        
        bc.data.load_and_scatter(generation_data if rank == 0 else None)
        spec.initializer(bc)
        
        bc.subproblems.load()
        from bundlechoice.subproblems.registry.greedy import GreedySubproblem
        if isinstance(bc.subproblems.demand_oracle, GreedySubproblem):
            _install_find_best_item(bc.subproblems.demand_oracle)
        
        obs_bundles = bc.subproblems.init_and_solve(theta)
        
        estimation_data = None
        if rank == 0:
            estimation_errors = generator.generate_errors(
                (1, num_agents, num_items), sigma
            )
            estimation_data = {
                "agent_data": generation_data["agent_data"],
                "item_data": generation_data["item_data"],
                "errors": estimation_errors,
                "obs_bundle": obs_bundles,
            }
        
        return {
            "generation": generation_data if rank == 0 else None,
            "estimation": estimation_data if rank == 0 else None,
        }
    
    def find_best_item(
        self, local_id: int, base_bundle: np.ndarray, items_left: np.ndarray,
        theta: np.ndarray, error_j: np.ndarray
    ) -> tuple:
        """find_best_item for real airline data."""
        modular_item = self.local_data["item_data"]["modular"]
        origin_city_arr = self.local_data["item_data"]["origin_city"]
        agent_hubs = self.local_data["agent_data"]["hubs"][local_id]
        
        theta_gs = theta[-1]
        theta_modular = theta[:-1]
        
        base_bundle_markets = base_bundle.astype(bool)
        base_hub_counts = np.zeros(len(agent_hubs))
        for h in range(len(agent_hubs)):
            if agent_hubs[h]:
                base_hub_counts[h] = np.sum(origin_city_arr[base_bundle_markets] == h)
        
        base_modular = modular_item.T @ base_bundle
        
        modular_values = np.zeros(len(items_left))
        congestion_costs = np.zeros(len(items_left))
        
        for idx, j in enumerate(items_left):
            modular_values[idx] = (base_modular + modular_item[j, :]) @ theta_modular
            
            a_j = origin_city_arr[j]
            if agent_hubs[a_j]:
                marginal_congestion = 2 * base_hub_counts[a_j] + 1
                congestion_costs[idx] = theta_gs * marginal_congestion
        
        full_values = modular_values + error_j[items_left] - congestion_costs
        
        best_idx = np.argmax(full_values)
        best_item = items_left[best_idx]
        best_val = full_values[best_idx]
        
        return best_item, best_val
    
    def _install_find_best_item(greedy_solver) -> None:
        import types
        def bound_method(local_id, base_bundle, items_left, theta, error_j):
            return find_best_item(greedy_solver, local_id, base_bundle, items_left, theta, error_j)
        greedy_solver.find_best_item = bound_method
    
    # Initialize theta: [theta_pop_weighted, theta_gs]
    theta_star = np.array([1.0, theta_gs])  # Positive weight for pop feature, theta_gs for congestion
    
    return SyntheticScenario(
        name="real_airline",
        config_factory=config_factory,
        feature_spec=feature_spec,
        payload_factory=payload_factory,
        theta_factory=lambda: theta_star.copy(),
        metadata={
            "num_agents": num_agents,
            "num_items": num_items,
            "num_cities": num_cities,
            "num_features": num_modular_features + 1,
            "sigma": sigma,
            "theta_gs": theta_gs,
            "temperature": temperature,
            "city_names": city_names,
        },
    )


if __name__ == "__main__":
    # Example usage
    data_dir = Path(__file__).parent / "data"
    loader = RealAirlineDataLoader(data_dir=data_dir)
    
    # Load data
    loader.load_cities_from_csv('cities_sample.csv')
    try:
        loader.load_airline_hubs_from_csv('airline_hubs_sample.csv')
    except FileNotFoundError:
        pass  # Optional
    
    # Create scenario
    scenario = create_real_airline_scenario(
        data_loader=loader,
        num_agents=3,
        theta_gs=0.2,
        temperature=1.0,
        seed=42
    )
    
    print("Real airline scenario created successfully!")
    print(f"Metadata: {scenario.metadata}")

