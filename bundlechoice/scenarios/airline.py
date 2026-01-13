"""Scenario factory builder for the Airline Network subproblem."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional

import numpy as np
from mpi4py import MPI

from bundlechoice.core import BundleChoice

from .base import FeatureSpec, SyntheticScenario
from .data_generator import DataGenerator, ModularAgentConfig, ModularItemConfig
from . import utils


@dataclass(frozen=True)
class AirlineParams:
    num_agents: int = 200
    num_cities: int = 10  # C cities
    num_simulations: int = 1
    sigma: float = 1.0
    theta_star: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0, 0.1]))
    num_modular_features: int = 3  # Number of modular features (x_j^T θ^mod)
    theta_gs: float = 0.1  # Gross substitutability parameter (θ^gs)
    min_hubs: int = 1  # Minimum number of hubs per agent
    max_hubs: int = 3  # Maximum number of hubs per agent
    agent_config: ModularAgentConfig = field(default_factory=lambda: ModularAgentConfig(apply_abs=True))
    item_config: Optional[ModularItemConfig] = None


class AirlineScenarioBuilder:
    """Builder-style interface for the Airline Network scenario factory."""

    def __init__(self, params: Optional[AirlineParams] = None) -> None:
        self._params = params or AirlineParams()
        # num_items = num_cities * (num_cities - 1) (all directed arcs except self-loops)
        num_items = self._params.num_cities * (self._params.num_cities - 1)
        if self._params.theta_star.shape[0] != self._params.num_modular_features + 1:
            raise ValueError("theta_star length must match num_modular_features + 1 (for θ^gs)")

    def with_dimensions(
        self, *, num_agents: Optional[int] = None, num_cities: Optional[int] = None
    ) -> "AirlineScenarioBuilder":
        params = self._params
        if num_agents is not None:
            params = replace(params, num_agents=num_agents)
        if num_cities is not None:
            params = replace(params, num_cities=num_cities)
        return AirlineScenarioBuilder(params)

    def with_num_modular_features(self, num_modular_features: int) -> "AirlineScenarioBuilder":
        theta = np.ones(num_modular_features + 1)
        theta[-1] = self._params.theta_gs
        return AirlineScenarioBuilder(replace(self._params, num_modular_features=num_modular_features, theta_star=theta))

    def with_num_simulations(self, num_simulations: int) -> "AirlineScenarioBuilder":
        return AirlineScenarioBuilder(replace(self._params, num_simulations=num_simulations))

    def with_sigma(self, sigma: float) -> "AirlineScenarioBuilder":
        return AirlineScenarioBuilder(replace(self._params, sigma=sigma))

    def with_theta(self, theta_star: np.ndarray) -> "AirlineScenarioBuilder":
        if theta_star.ndim != 1:
            raise ValueError("theta_star must be 1-dimensional")
        return AirlineScenarioBuilder(replace(self._params, theta_star=theta_star))

    def with_theta_gs(self, theta_gs: float) -> "AirlineScenarioBuilder":
        theta = self._params.theta_star.copy()
        theta[-1] = theta_gs
        return AirlineScenarioBuilder(replace(self._params, theta_gs=theta_gs, theta_star=theta))

    def with_hub_range(self, min_hubs: int, max_hubs: int) -> "AirlineScenarioBuilder":
        return AirlineScenarioBuilder(replace(self._params, min_hubs=min_hubs, max_hubs=max_hubs))

    def with_agent_config(
        self,
        apply_abs: bool = True,
        multiplier: float = 1.0,
        mean: float = 0.0,
        std: float = 1.0,
    ) -> "AirlineScenarioBuilder":
        """Configure modular agent feature generation."""
        agent_config = ModularAgentConfig(
            apply_abs=apply_abs, multiplier=multiplier, mean=mean, std=std
        )
        return AirlineScenarioBuilder(replace(self._params, agent_config=agent_config))

    def with_item_config(
        self,
        apply_abs: bool = False,
        multiplier: float = 1.0,
        mean: float = 0.0,
        std: float = 1.0,
    ) -> "AirlineScenarioBuilder":
        """Configure modular item feature generation."""
        item_config = ModularItemConfig(
            apply_abs=apply_abs, multiplier=multiplier, mean=mean, std=std
        )
        return AirlineScenarioBuilder(replace(self._params, item_config=item_config))

    def build(self) -> SyntheticScenario:
        params = self._params
        num_items = params.num_cities * (params.num_cities - 1)

        def config_factory() -> Dict[str, Any]:
            return {
                "dimensions": {
                    "num_agents": params.num_agents,
                    "num_items": num_items,
                    "num_features": params.num_modular_features + 1,  # modular + congestion
                    "num_simulations": 1,
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

        def _airline_feature_initializer(bc: BundleChoice) -> None:
            """Initialize features for airline factory."""
            bc.oracles.set_features_oracle(_airline_features_oracle)
            if bc.subproblems.subproblem_instance is not None:
                from bundlechoice.subproblems.registry.greedy import GreedySubproblem
                if isinstance(bc.subproblems.subproblem_instance, GreedySubproblem):
                    _install_find_best_item(bc.subproblems.subproblem_instance)

        feature_spec = FeatureSpec(mode="oracle", initializer=_airline_feature_initializer)

        def payload_factory(
            bc: BundleChoice,
            comm: MPI.Comm,
            spec: FeatureSpec,
            timeout: Optional[int],
            seed: Optional[int],
            theta: Any,
        ) -> Dict[str, Dict[str, Any]]:
            rank = comm.Get_rank()

            generator = DataGenerator(seed=seed)
            generation_data = None

            if rank == 0:
                # Generate cities on a grid
                cities = _generate_cities_on_grid(params.num_cities, generator)
                
                # Generate markets (directed arcs): all pairs (i, j) where i != j
                markets, origin_city, dest_city = _generate_markets(params.num_cities)
                
                # Generate hubs for each agent (heterogeneous)
                agent_hubs = _generate_agent_hubs(
                    params.num_agents, params.num_cities, params.min_hubs, params.max_hubs, generator
                )
                
                # Generate modular features for markets (item-specific profits/costs)
                item_config = params.item_config or ModularItemConfig(apply_abs=True)
                modular_item = generator.generate_modular_item(
                    (num_items, params.num_modular_features), item_config
                )
                
                # Generate errors
                errors = generator.generate_errors(
                    (params.num_agents, num_items), params.sigma
                )
                
                generation_data = {
                    "agent_data": {"hubs": agent_hubs},
                    "item_data": {
                        "modular": modular_item,
                        "origin_city": origin_city,  # (num_items,) array: origin city index for each market
                        "dest_city": dest_city,  # (num_items,) array: dest city index for each market
                    },
                    "errors": errors,
                    # Metadata (not scattered, only on rank 0)
                    "_metadata": {
                        "markets": markets,  # (num_items, 2) array: [origin, dest] for each market
                        "cities": cities,  # (num_cities, 2) array: [x, y] coordinates
                    } if rank == 0 else None,
                }

            bc.data.load_and_scatter(generation_data if rank == 0 else None)
            spec.initializer(bc)

            # Ensure subproblems are loaded and find_best_item is installed
            bc.subproblems.load()
            from bundlechoice.subproblems.registry.greedy import GreedySubproblem
            if isinstance(bc.subproblems.subproblem_instance, GreedySubproblem):
                _install_find_best_item(bc.subproblems.subproblem_instance)

            obs_bundles = bc.subproblems.init_and_solve(theta)

            estimation_data = None
            if rank == 0:
                estimation_errors = generator.generate_errors(
                    (params.num_simulations, params.num_agents, num_items), params.sigma
                )
                estimation_data = {
                    "agent_data": generation_data["agent_data"],
                    "item_data": generation_data["item_data"],
                    "errors": estimation_errors,
                    "obs_bundle": obs_bundles,
                }

            return {
                "generation": utils.root_dict(comm, generation_data),
                "estimation": utils.root_dict(comm, estimation_data),
            }

        return SyntheticScenario(
            name="airline",
            config_factory=config_factory,
            feature_spec=feature_spec,
            payload_factory=payload_factory,
            theta_factory=lambda: params.theta_star.copy(),
            metadata={
                "num_agents": params.num_agents,
                "num_items": num_items,
                "num_cities": params.num_cities,
                "num_features": params.num_modular_features + 1,
                "num_simulations": params.num_simulations,
                "sigma": params.sigma,
                "theta_gs": params.theta_gs,
            },
        )


def _generate_cities_on_grid(num_cities: int, generator: Optional[DataGenerator] = None) -> np.ndarray:
    """Generate cities at realistic geographic locations (not a grid)."""
    # Use provided generator or create a temporary one
    if generator is None:
        rng = np.random.default_rng(42)
    else:
        rng = generator.rng
    
    # Generate cities with a more realistic distribution
    # Use a mix of uniform random and some clustering to simulate real geography
    cities = []
    
    # Create a roughly square region (e.g., 0 to 10 in both dimensions)
    region_size = 10.0
    
    # Use a mix: some cities uniformly distributed, some in clusters
    num_clusters = max(2, num_cities // 4)  # 2-4 clusters depending on num_cities
    cities_per_cluster = num_cities // num_clusters
    remaining_cities = num_cities - cities_per_cluster * num_clusters
    
    # Generate cluster centers
    cluster_centers = []
    for _ in range(num_clusters):
        cluster_centers.append([
            rng.uniform(1.0, region_size - 1.0),
            rng.uniform(1.0, region_size - 1.0)
        ])
    
    # Generate cities in clusters
    for cluster_idx, center in enumerate(cluster_centers):
        cluster_size = cities_per_cluster
        if cluster_idx < remaining_cities:
            cluster_size += 1
        
        for _ in range(cluster_size):
            # Cities around cluster center with some spread
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.exponential(1.0)  # Exponential for more realistic clustering
            radius = min(radius, 2.5)  # Cap the radius
            
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            
            # Keep within bounds
            x = np.clip(x, 0.5, region_size - 0.5)
            y = np.clip(y, 0.5, region_size - 0.5)
            
            cities.append([x, y])
    
    # Shuffle to avoid obvious clustering in the order
    cities_array = np.array(cities, dtype=np.float64)
    indices = rng.permutation(len(cities_array))
    cities_array = cities_array[indices]
    
    return cities_array


def _generate_markets(num_cities: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate all directed arcs (markets) between cities, excluding self-loops."""
    markets = []
    origin_city = []
    dest_city = []
    market_idx = 0
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                markets.append([i, j])
                origin_city.append(i)
                dest_city.append(j)
                market_idx += 1
    return np.array(markets), np.array(origin_city), np.array(dest_city)


def _generate_agent_hubs(
    num_agents: int,
    num_cities: int,
    min_hubs: int,
    max_hubs: int,
    generator: DataGenerator,
) -> np.ndarray:
    """Generate hubs for each agent heterogeneously."""
    agent_hubs = []
    for _ in range(num_agents):
        num_hubs = generator.rng.integers(min_hubs, max_hubs + 1)
        hubs = generator.rng.choice(num_cities, size=num_hubs, replace=False)
        # Store as boolean array for each agent: (num_cities,) where True indicates hub
        hub_array = np.zeros(num_cities, dtype=bool)
        hub_array[hubs] = True
        agent_hubs.append(hub_array)
    return np.array(agent_hubs)  # (num_agents, num_cities)


def _airline_features_oracle(agent_idx: int, bundles: np.ndarray, data: Dict[str, Any]) -> np.ndarray:
    """
    Compute features for airline factory.
    Features: [modular_item.T @ bundle, -θ^gs * Σ(h∈H) |{ab ∈ B: a = h}|^2]
    """
    modular_item = data["item_data"]["modular"]  # (num_items, num_modular_features)
    origin_city = data["item_data"]["origin_city"]  # (num_items,)
    agent_hubs = data["agent_data"]["hubs"][agent_idx]  # (num_cities,) boolean
    
    if bundles.ndim == 1:
        # Single bundle
        modular_features = modular_item.T @ bundles  # (num_modular_features,)
        
        # Compute congestion: Σ(h∈H) |{ab ∈ B: a = h}|^2
        # For each hub h, count markets in bundle that originate from h
        bundle_markets = bundles.astype(bool)
        congestion = 0.0
        for h in range(len(agent_hubs)):
            if agent_hubs[h]:
                # Count markets in bundle that originate from hub h
                count = np.sum((origin_city[bundle_markets] == h))
                congestion += count ** 2
        
        return np.concatenate((modular_features, [-congestion]))
    else:
        # Multiple bundles (vectorized)
        modular_features = modular_item.T @ bundles  # (num_modular_features, num_bundles)
        
        # Compute congestion for each bundle: vectorized
        num_bundles = bundles.shape[1]
        congestion = np.zeros(num_bundles)
        
        # For each hub, compute counts across all bundles
        for h in range(len(agent_hubs)):
            if agent_hubs[h]:
                # bundles is (num_items, num_bundles), origin_city is (num_items,)
                # For each bundle, count markets with origin == h
                # Vectorized: (bundles > 0) & (origin_city[:, None] == h)
                hub_mask = (origin_city == h)[:, None]  # (num_items, 1)
                counts = np.sum(bundles * hub_mask, axis=0)  # (num_bundles,)
                congestion += counts ** 2
        
        return np.concatenate((modular_features, -congestion[None, :]), axis=0)


def _find_best_item(
    self, local_id: int, base_bundle: np.ndarray, items_left: np.ndarray,
    theta: np.ndarray, error_j: np.ndarray
) -> tuple[int, float]:
    """
    find_best_item for airline network with congestion.
    
    For each market j in items_left, computes the full value of bundle = base + {j}:
    value = modular_features(base + j) @ theta_modular + error_j[j] - theta_gs * congestion(base + j)
    
    The congestion term is: Σ(h∈H) |{ab ∈ (base + j): a = h}|^2
    Marginal congestion when adding market j with origin a_j:
    - If a_j ∈ H: marginal = 2 * n_{a_j}(base) + 1
    - If a_j ∉ H: marginal = 0
    """
    modular_item = self.local_data["item_data"]["modular"]  # (num_items, num_modular_features)
    origin_city = self.local_data["item_data"]["origin_city"]  # (num_items,)
    agent_hubs = self.local_data["agent_data"]["hubs"][local_id]  # (num_cities,) boolean
    
    theta_gs = theta[-1]
    theta_modular = theta[:-1]
    
    # Compute current hub counts for base bundle
    base_bundle_markets = base_bundle.astype(bool)
    base_hub_counts = np.zeros(len(agent_hubs))
    for h in range(len(agent_hubs)):
        if agent_hubs[h]:
            base_hub_counts[h] = np.sum(origin_city[base_bundle_markets] == h)
    
    # Base modular features
    base_modular = modular_item.T @ base_bundle  # (num_modular_features,)
    
    # For each candidate market j in items_left
    modular_values = np.zeros(len(items_left))
    congestion_costs = np.zeros(len(items_left))
    
    for idx, j in enumerate(items_left):
        # Modular value: base_modular + modular_item[j, :]
        modular_values[idx] = (base_modular + modular_item[j, :]) @ theta_modular
        
        # Congestion cost: marginal increase when adding market j
        a_j = origin_city[j]  # origin city of market j
        if agent_hubs[a_j]:
            # Marginal congestion: 2 * n_{a_j}(base) + 1
            marginal_congestion = 2 * base_hub_counts[a_j] + 1
            congestion_costs[idx] = theta_gs * marginal_congestion
        # else: congestion_costs[idx] = 0 (already initialized)
    
    # Full values: modular + errors - congestion
    full_values = modular_values + error_j[items_left] - congestion_costs
    
    best_idx = np.argmax(full_values)
    best_item = items_left[best_idx]
    best_val = full_values[best_idx]
    
    return best_item, best_val


def _install_find_best_item(greedy_solver: Any) -> None:
    """Install find_best_item method on GreedySubproblem instance."""
    import types
    def bound_method(local_id, base_bundle, items_left, theta, error_j):
        return _find_best_item(greedy_solver, local_id, base_bundle, items_left, theta, error_j)
    greedy_solver.find_best_item = bound_method


def build() -> AirlineScenarioBuilder:
    return AirlineScenarioBuilder()


__all__ = ["AirlineScenarioBuilder", "build"]

