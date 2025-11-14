"""Scenario factory builder for the PlainSingleItem subproblem."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional

import numpy as np
from mpi4py import MPI

from bundlechoice.core import BundleChoice

from .base import FeatureSpec, SyntheticScenario
from .data_generator import (
    CorrelationConfig,
    DataGenerator,
    ModularAgentConfig,
    ModularItemConfig,
)
from . import utils


@dataclass(frozen=True)
class PlainSingleItemParams:
    num_agents: int = 500
    num_items: int = 2
    num_agent_features: int = 4
    num_item_features: int = 1
    num_simuls: int = 1
    sigma: float = 1.0
    agent_config: ModularAgentConfig = field(default_factory=ModularAgentConfig)
    item_config: ModularItemConfig = field(default_factory=ModularItemConfig)

    @property
    def num_features(self) -> int:
        return self.num_agent_features + self.num_item_features

    @property
    def theta_star(self) -> np.ndarray:
        return np.ones(self.num_features)


class PlainSingleItemScenarioBuilder:
    def __init__(self, params: Optional[PlainSingleItemParams] = None) -> None:
        self._params = params or PlainSingleItemParams()

    def with_dimensions(
        self, *, num_agents: Optional[int] = None, num_items: Optional[int] = None
    ) -> "PlainSingleItemScenarioBuilder":
        params = self._params
        if num_agents is not None:
            params = replace(params, num_agents=num_agents)
        if num_items is not None:
            params = replace(params, num_items=num_items)
        return PlainSingleItemScenarioBuilder(params)

    def with_feature_counts(
        self, *, num_agent_features: Optional[int] = None, num_item_features: Optional[int] = None
    ) -> "PlainSingleItemScenarioBuilder":
        params = self._params
        if num_agent_features is not None:
            params = replace(params, num_agent_features=num_agent_features)
        if num_item_features is not None:
            params = replace(params, num_item_features=num_item_features)
        return PlainSingleItemScenarioBuilder(params)

    def with_sigma(self, sigma: float) -> "PlainSingleItemScenarioBuilder":
        return PlainSingleItemScenarioBuilder(replace(self._params, sigma=sigma))

    def with_num_simuls(self, num_simuls: int) -> "PlainSingleItemScenarioBuilder":
        return PlainSingleItemScenarioBuilder(replace(self._params, num_simuls=num_simuls))

    def with_correlation(
        self,
        enabled: bool = True,
        matrix_range: tuple[int, int] = (0, 4),
        normalize: bool = True,
    ) -> "PlainSingleItemScenarioBuilder":
        """Enable/configure correlation matrix transformation for agent features."""
        agent_config = replace(
            self._params.agent_config,
            correlation=CorrelationConfig(
                enabled=enabled, matrix_range=matrix_range, normalize=normalize
            ),
        )
        return PlainSingleItemScenarioBuilder(
            replace(self._params, agent_config=agent_config)
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
                "subproblem": {"name": "PlainSingleItem"},
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
        ) -> Dict[str, Dict[str, Any]]:
            rank = comm.Get_rank()
            generator = DataGenerator(seed=seed)

            generation_data = None
            if rank == 0:
                # Generate in same order as manual: agent first, then errors
                agent_modular = generator.generate_modular_agent(
                    (params.num_agents, params.num_items, params.num_agent_features),
                    params.agent_config,
                )
                item_modular = generator.generate_modular_item(
                    (params.num_items, params.num_item_features), params.item_config
                )
                errors = generator.generate_errors(
                    (params.num_agents, params.num_items), params.sigma
                )
                generation_data = {
                    "item_data": {"modular": item_modular},
                    "agent_data": {"modular": agent_modular},
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
                label=f"{params.num_agents}x{params.num_items}-plain-single-item",
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
            name="plain_single_item",
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


def build() -> PlainSingleItemScenarioBuilder:
    return PlainSingleItemScenarioBuilder()


__all__ = ["PlainSingleItemScenarioBuilder", "build"]

