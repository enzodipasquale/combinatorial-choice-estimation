"""Scenario factory builder for the QuadraticKnapsack subproblem."""

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
    QuadraticGenerationMethod,
    QuadraticItemConfig,
)
from . import utils


@dataclass(frozen=True)
class QuadraticKnapsackParams:
    num_agents: int = 20
    num_items: int = 20
    num_agent_modular_features: int = 1
    num_agent_quadratic_features: int = 1
    num_item_modular_features: int = 1
    num_item_quadratic_features: int = 1
    num_simuls: int = 1
    sigma: float = 1.0
    agent_config: ModularAgentConfig = field(
        default_factory=lambda: ModularAgentConfig(apply_abs=False)
    )
    item_config: ModularItemConfig = field(
        default_factory=lambda: ModularItemConfig(apply_abs=False)
    )
    agent_quadratic_config: QuadraticItemConfig = field(
        default_factory=lambda: QuadraticItemConfig(
            method=QuadraticGenerationMethod.BINARY_CHOICE,
            binary_prob=0.2,
            binary_value=0.3,
        )
    )
    item_quadratic_config: QuadraticItemConfig = field(
        default_factory=lambda: QuadraticItemConfig(
            method=QuadraticGenerationMethod.BINARY_CHOICE,
            binary_prob=0.2,
            binary_value=0.25,
        )
    )
    capacity_config: CapacityConfig = field(default_factory=CapacityConfig)

    @property
    def num_features(self) -> int:
        return (
            self.num_agent_modular_features
            + self.num_agent_quadratic_features
            + self.num_item_modular_features
            + self.num_item_quadratic_features
        )

    @property
    def theta_star(self) -> np.ndarray:
        return np.ones(self.num_features)


class QuadraticKnapsackScenarioBuilder:
    def __init__(self, params: Optional[QuadraticKnapsackParams] = None) -> None:
        self._params = params or QuadraticKnapsackParams()

    def with_dimensions(
        self, *, num_agents: Optional[int] = None, num_items: Optional[int] = None
    ) -> "QuadraticKnapsackScenarioBuilder":
        params = self._params
        if num_agents is not None:
            params = replace(params, num_agents=num_agents)
        if num_items is not None:
            params = replace(params, num_items=num_items)
        return QuadraticKnapsackScenarioBuilder(params)

    def with_feature_counts(
        self,
        *,
        num_agent_modular: Optional[int] = None,
        num_agent_quadratic: Optional[int] = None,
        num_item_modular: Optional[int] = None,
        num_item_quadratic: Optional[int] = None,
    ) -> "QuadraticKnapsackScenarioBuilder":
        params = self._params
        if num_agent_modular is not None:
            params = replace(params, num_agent_modular_features=num_agent_modular)
        if num_agent_quadratic is not None:
            params = replace(params, num_agent_quadratic_features=num_agent_quadratic)
        if num_item_modular is not None:
            params = replace(params, num_item_modular_features=num_item_modular)
        if num_item_quadratic is not None:
            params = replace(params, num_item_quadratic_features=num_item_quadratic)
        return QuadraticKnapsackScenarioBuilder(params)

    def with_sigma(self, sigma: float) -> "QuadraticKnapsackScenarioBuilder":
        return QuadraticKnapsackScenarioBuilder(replace(self._params, sigma=sigma))

    def with_num_simuls(self, num_simuls: int) -> "QuadraticKnapsackScenarioBuilder":
        return QuadraticKnapsackScenarioBuilder(
            replace(self._params, num_simuls=num_simuls)
        )

    def with_capacity_config(
        self,
        mean_multiplier: float = 0.45,
        lower_multiplier: float = 0.85,
        upper_multiplier: float = 1.15,
    ) -> "QuadraticKnapsackScenarioBuilder":
        """Configure capacity generation parameters."""
        capacity_config = CapacityConfig(
            mean_multiplier=mean_multiplier,
            lower_multiplier=lower_multiplier,
            upper_multiplier=upper_multiplier,
        )
        return QuadraticKnapsackScenarioBuilder(
            replace(self._params, capacity_config=capacity_config)
        )

    def with_agent_modular_config(
        self,
        multiplier: float = 1.0,
        mean: float = 0.0,
        std: float = 1.0,
        apply_abs: bool = False,
    ) -> "QuadraticKnapsackScenarioBuilder":
        """Configure modular agent feature generation."""
        agent_config = ModularAgentConfig(
            multiplier=multiplier, mean=mean, std=std, apply_abs=apply_abs
        )
        return QuadraticKnapsackScenarioBuilder(
            replace(self._params, agent_config=agent_config)
        )

    def with_agent_quadratic_config(
        self,
        method: QuadraticGenerationMethod,
        binary_prob: float = 0.2,
        binary_value: float = 0.3,
        mask_threshold: float = 0.3,
    ) -> "QuadraticKnapsackScenarioBuilder":
        """Configure agent quadratic feature generation."""
        if method == QuadraticGenerationMethod.BINARY_CHOICE:
            agent_quadratic_config = QuadraticItemConfig(
                method=method,
                binary_prob=binary_prob,
                binary_value=binary_value,
                mask_threshold=mask_threshold,
            )
        else:
            agent_quadratic_config = QuadraticItemConfig(
                method=method, mask_threshold=mask_threshold
            )
        return QuadraticKnapsackScenarioBuilder(
            replace(self._params, agent_quadratic_config=agent_quadratic_config)
        )

    def with_item_quadratic_config(
        self,
        method: QuadraticGenerationMethod,
        binary_prob: float = 0.2,
        binary_value: float = 0.25,
        mask_threshold: float = 0.3,
    ) -> "QuadraticKnapsackScenarioBuilder":
        """Configure item quadratic feature generation."""
        if method == QuadraticGenerationMethod.BINARY_CHOICE:
            item_quadratic_config = QuadraticItemConfig(
                method=method,
                binary_prob=binary_prob,
                binary_value=binary_value,
                mask_threshold=mask_threshold,
            )
        else:
            item_quadratic_config = QuadraticItemConfig(
                method=method, mask_threshold=mask_threshold
            )
        return QuadraticKnapsackScenarioBuilder(
            replace(self._params, item_quadratic_config=item_quadratic_config)
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
                "subproblem": {"name": "QuadKnapsack"},
                "row_generation": {
                    "max_iters": 100,
                    "tolerance_optimality": 0.001,
                    "min_iters": 1,
                    "gurobi_settings": {"OutputFlag": 0},
                    "theta_ubs": 100,
                },
            }

        feature_spec = FeatureSpec.build()  # Auto-generate from data structure

        def payload_factory(
            bc: BundleChoice,
            comm: MPI.Comm,
            spec: FeatureSpec,
            timeout: Optional[int],
            seed: Optional[int],
        ) -> Dict[str, Dict[str, Any]]:
            rank = comm.Get_rank()
            generator = DataGenerator(seed=seed)
            generation_data = None
            if rank == 0:
                # Generate data in the same order as manual scripts
                agent_modular = generator.generate_modular_agent(
                    (
                        params.num_agents,
                        params.num_items,
                        params.num_agent_modular_features,
                    ),
                    params.agent_config,
                )
                agent_quadratic = generator.generate_quadratic_agent(
                    (
                        params.num_agents,
                        params.num_items,
                        params.num_items,
                        params.num_agent_quadratic_features,
                    ),
                    params.agent_quadratic_config,
                )
                item_modular = generator.generate_modular_item(
                    (params.num_items, params.num_item_modular_features),
                    params.item_config,
                )
                item_quadratic = generator.generate_quadratic_item(
                    (
                        params.num_items,
                        params.num_items,
                        params.num_item_quadratic_features,
                    ),
                    params.item_quadratic_config,
                )
                item_weights = generator.generate_weights(params.num_items)
                agent_capacity = generator.generate_capacities(
                    params.num_agents, item_weights, params.capacity_config
                )
                errors = generator.generate_errors(
                    (params.num_simuls, params.num_agents, params.num_items),
                    params.sigma,
                )

                generation_data = {
                    "agent_data": {
                        "modular": agent_modular,
                        "quadratic": agent_quadratic,
                        "capacity": agent_capacity,
                    },
                    "item_data": {
                        "modular": item_modular,
                        "quadratic": item_quadratic,
                        "weights": item_weights,
                    },
                    "errors": errors,
                }

            bc.data.load_and_scatter(generation_data if rank == 0 else None)
            spec.initializer(bc)

            theta_star = params.theta_star

            def solve() -> np.ndarray:
                return bc.subproblems.init_and_solve(theta_star)

            obs_bundles = utils.mpi_call_with_timeout(
                comm,
                solve,
                timeout,
                label=f"{params.num_agents}x{params.num_items}-quadratic-knapsack",
            )

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
            name="quadratic_knapsack",
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


def build() -> QuadraticKnapsackScenarioBuilder:
    return QuadraticKnapsackScenarioBuilder()


__all__ = ["QuadraticKnapsackScenarioBuilder", "build"]

