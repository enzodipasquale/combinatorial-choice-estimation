"""Scenario factory builder for the LinearKnapsack subproblem."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional

import numpy as np
from mpi4py import MPI

from bundlechoice.core import BundleChoice

from .base import FeatureSpec, SyntheticScenario
from .data_generator import (
    CapacityConfig,
    DataGenerator,
    ModularAgentConfig,
    ModularItemConfig,
    WeightConfig,
)
from . import utils


@dataclass(frozen=True)
class LinearKnapsackParams:
    num_agents: int = 20
    num_items: int = 20
    num_agent_modular_features: int = 2
    num_item_modular_features: int = 2
    num_simuls: int = 1
    sigma: float = 1.0
    agent_config: ModularAgentConfig = field(
        default_factory=lambda: ModularAgentConfig(apply_abs=True)
    )
    item_config: ModularItemConfig = field(
        default_factory=lambda: ModularItemConfig(apply_abs=True)
    )
    weight_config: WeightConfig = field(default_factory=WeightConfig)
    capacity_config: CapacityConfig = field(default_factory=CapacityConfig)
    _random_capacity_low: Optional[int] = None
    _random_capacity_high: Optional[int] = None

    @property
    def num_features(self) -> int:
        return self.num_agent_modular_features + self.num_item_modular_features

    @property
    def theta_star(self) -> np.ndarray:
        return np.ones(self.num_features)


class LinearKnapsackScenarioBuilder:
    def __init__(self, params: Optional[LinearKnapsackParams] = None) -> None:
        self._params = params or LinearKnapsackParams()

    def with_dimensions(
        self, *, num_agents: Optional[int] = None, num_items: Optional[int] = None
    ) -> "LinearKnapsackScenarioBuilder":
        params = self._params
        if num_agents is not None:
            params = replace(params, num_agents=num_agents)
        if num_items is not None:
            params = replace(params, num_items=num_items)
        return LinearKnapsackScenarioBuilder(params)

    def with_feature_counts(
        self,
        *,
        num_agent_features: Optional[int] = None,
        num_item_features: Optional[int] = None,
    ) -> "LinearKnapsackScenarioBuilder":
        params = self._params
        if num_agent_features is not None:
            params = replace(params, num_agent_modular_features=num_agent_features)
        if num_item_features is not None:
            params = replace(params, num_item_modular_features=num_item_features)
        return LinearKnapsackScenarioBuilder(params)

    def with_sigma(self, sigma: float) -> "LinearKnapsackScenarioBuilder":
        return LinearKnapsackScenarioBuilder(replace(self._params, sigma=sigma))

    def with_num_simuls(self, num_simuls: int) -> "LinearKnapsackScenarioBuilder":
        return LinearKnapsackScenarioBuilder(replace(self._params, num_simuls=num_simuls))

    def with_weight_config(
        self,
        distribution: str = "uniform",
        low: int = 1,
        high: int = 10,
        log_mean: float = 0.0,
        log_std: float = 1.0,
        exp_scale: float = 2.0,
    ) -> "LinearKnapsackScenarioBuilder":
        """Configure weight generation parameters.
        
        Args:
            distribution: 'uniform', 'lognormal', or 'exponential' (default: 'uniform')
            low: Minimum weight value
            high: Maximum weight value
            log_mean: Mean for log-normal distribution
            log_std: Std for log-normal distribution
            exp_scale: Scale for exponential distribution
        """
        weight_config = WeightConfig(
            distribution=distribution,
            low=low,
            high=high,
            log_mean=log_mean,
            log_std=log_std,
            exp_scale=exp_scale,
        )
        return LinearKnapsackScenarioBuilder(
            replace(self._params, weight_config=weight_config)
        )

    def with_capacity_config(
        self,
        mean_multiplier: float = 0.45,
        lower_multiplier: float = 0.85,
        upper_multiplier: float = 1.15,
    ) -> "LinearKnapsackScenarioBuilder":
        """Configure capacity generation using variance-based method (legacy)."""
        capacity_config = CapacityConfig(
            mean_multiplier=mean_multiplier,
            lower_multiplier=lower_multiplier,
            upper_multiplier=upper_multiplier,
        )
        return LinearKnapsackScenarioBuilder(
            replace(self._params, capacity_config=capacity_config)
        )

    def with_capacity_fraction(
        self,
        fraction: float,
    ) -> "LinearKnapsackScenarioBuilder":
        """Configure capacity as fixed fraction of total weight (most standard method).
        
        Args:
            fraction: Fraction of total weight sum (e.g., 0.5 for 50%).
        """
        capacity_config = CapacityConfig(fraction=fraction)
        return LinearKnapsackScenarioBuilder(
            replace(self._params, capacity_config=capacity_config)
        )

    def with_random_capacity(
        self,
        low: int = 1,
        high: int = 100,
    ) -> "LinearKnapsackScenarioBuilder":
        """Use random capacity generation (not based on weights)."""
        # Store random capacity params in metadata for payload_factory to use
        return LinearKnapsackScenarioBuilder(
            replace(self._params, _random_capacity_low=low, _random_capacity_high=high)
        )

    def build(self) -> SyntheticScenario:
        params = self._params

        def config_factory() -> Dict[str, Any]:
            return {
                "dimensions": {
                    "num_agents": params.num_agents,
                    "num_items": params.num_items,
                    "num_features": params.num_features,
                    "num_simuls": params.num_simuls,
                },
                "subproblem": {
                    "name": "LinearKnapsack",
                    "settings": {"TimeLimit": 10, "MIPGap_tol": 0.01},
                },
                "row_generation": {
                    "max_iters": 100,
                    "tolerance_optimality": 0.0001,
                    "min_iters": 1,
                    "gurobi_settings": {"OutputFlag": 0},
                },
            }

        feature_spec = FeatureSpec.build()

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
                # Generate in same order as manual: agent, item, weights, capacity, errors
                agent_modular = generator.generate_modular_agent(
                    (
                        params.num_agents,
                        params.num_items,
                        params.num_agent_modular_features,
                    ),
                    params.agent_config,
                )
                item_modular = generator.generate_modular_item(
                    (params.num_items, params.num_item_modular_features),
                    params.item_config,
                )
                item_weights = generator.generate_weights(params.num_items, params.weight_config)
                if params._random_capacity_low is not None:
                    agent_capacity = generator.generate_random_capacities(
                        params.num_agents, params._random_capacity_low, params._random_capacity_high
                    )
                else:
                    agent_capacity = generator.generate_capacities(
                        params.num_agents, item_weights, params.capacity_config
                    )
                errors = generator.generate_errors(
                    (params.num_simuls, params.num_agents, params.num_items), params.sigma
                )
                generation_data = {
                    "item_data": {
                        "modular": item_modular,
                        "weights": item_weights,
                    },
                    "agent_data": {
                        "modular": agent_modular,
                        "capacity": agent_capacity,
                    },
                    "errors": errors,
                }

            bc.data.load_and_scatter(generation_data if rank == 0 else None)
            spec.initializer(bc)

            obs_bundles = bc.subproblems.init_and_solve(theta)

            estimation_data = None
            if rank == 0:
                estimation_errors = generator.generate_errors(
                    (params.num_simuls, params.num_agents, params.num_items), params.sigma
                )
                estimation_data = {
                    "item_data": generation_data["item_data"],
                    "agent_data": generation_data["agent_data"],
                    "errors": estimation_errors,
                    "obs_bundle": obs_bundles,
                }

            return {
                "generation": utils.root_dict(comm, generation_data),
                "estimation": utils.root_dict(comm, estimation_data),
            }

        return SyntheticScenario(
            name="linear_knapsack",
            config_factory=config_factory,
            feature_spec=feature_spec,
            payload_factory=payload_factory,
            theta_factory=lambda: params.theta_star.copy(),
            metadata={
                "num_agents": params.num_agents,
                "num_items": params.num_items,
                "num_features": params.num_features,
                "num_simuls": params.num_simuls,
                "sigma": params.sigma,
            },
        )


def build() -> LinearKnapsackScenarioBuilder:
    return LinearKnapsackScenarioBuilder()


__all__ = ["LinearKnapsackScenarioBuilder", "build"]

