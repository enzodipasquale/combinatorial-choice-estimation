"""Scenario factory builder for the quadratic supermodular subproblem."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional

import numpy as np
from mpi4py import MPI

from bundlechoice.core import BundleChoice

from .base import FeatureSpec, SyntheticScenario
from .data_generator import (
    DataGenerator,
    ModularAgentConfig,
    ModularItemConfig,
    QuadraticGenerationMethod,
    QuadraticItemConfig,
)
from . import utils


@dataclass(frozen=True)
class SupermodParams:
    num_agents: int = 20
    num_items: int = 50
    num_modular_agent_features: int = 2
    num_modular_item_features: int = 2
    num_quadratic_agent_features: int = 0
    num_quadratic_item_features: int = 2
    num_simuls: int = 1
    sigma: float = 5.0
    agent_config: ModularAgentConfig = field(
        default_factory=lambda: ModularAgentConfig(
            multiplier=-2.0, mean=2.0, std=1.0, apply_abs=True
        )
    )
    item_config: ModularItemConfig = field(
        default_factory=lambda: ModularItemConfig(
            multiplier=-2.0, mean=2.0, std=1.0, apply_abs=True
        )
    )
    quadratic_config: QuadraticItemConfig = field(
        default_factory=lambda: QuadraticItemConfig(
            method=QuadraticGenerationMethod.EXPONENTIAL
        )
    )

    @property
    def num_features(self) -> int:
        return (
            self.num_modular_agent_features
            + self.num_modular_item_features
            + self.num_quadratic_agent_features
            + self.num_quadratic_item_features
        )

    @property
    def theta_star(self) -> np.ndarray:
        return np.ones(self.num_features)


class SupermodScenarioBuilder:
    def __init__(self, params: Optional[SupermodParams] = None) -> None:
        self._params = params or SupermodParams()

    def with_dimensions(
        self, *, num_agents: Optional[int] = None, num_items: Optional[int] = None
    ) -> "SupermodScenarioBuilder":
        params = self._params
        if num_agents is not None:
            params = replace(params, num_agents=num_agents)
        if num_items is not None:
            params = replace(params, num_items=num_items)
        return SupermodScenarioBuilder(params)

    def with_feature_counts(
        self,
        *,
        num_mod_agent: Optional[int] = None,
        num_mod_item: Optional[int] = None,
        num_quad_agent: Optional[int] = None,
        num_quad_item: Optional[int] = None,
    ) -> "SupermodScenarioBuilder":
        params = self._params
        if num_mod_agent is not None:
            params = replace(params, num_modular_agent_features=num_mod_agent)
        if num_mod_item is not None:
            params = replace(params, num_modular_item_features=num_mod_item)
        if num_quad_agent is not None:
            params = replace(params, num_quadratic_agent_features=num_quad_agent)
        if num_quad_item is not None:
            params = replace(params, num_quadratic_item_features=num_quad_item)
        return SupermodScenarioBuilder(params)

    def with_sigma(self, sigma: float) -> "SupermodScenarioBuilder":
        return SupermodScenarioBuilder(replace(self._params, sigma=sigma))

    def with_num_simuls(self, num_simuls: int) -> "SupermodScenarioBuilder":
        return SupermodScenarioBuilder(replace(self._params, num_simuls=num_simuls))

    def with_agent_modular_config(
        self,
        multiplier: float = -2.0,
        mean: float = 2.0,
        std: float = 1.0,
    ) -> "SupermodScenarioBuilder":
        """Configure modular agent feature generation."""
        agent_config = ModularAgentConfig(
            multiplier=multiplier, mean=mean, std=std, apply_abs=True
        )
        return SupermodScenarioBuilder(replace(self._params, agent_config=agent_config))

    def with_quadratic_method(
        self,
        method: QuadraticGenerationMethod,
        binary_prob: float = 0.2,
        binary_value: float = 1.0,
        mask_threshold: float = 0.3,
    ) -> "SupermodScenarioBuilder":
        """Configure quadratic item generation method."""
        if method == QuadraticGenerationMethod.BINARY_CHOICE:
            quadratic_config = QuadraticItemConfig(
                method=method,
                binary_prob=binary_prob,
                binary_value=binary_value,
                mask_threshold=mask_threshold,
            )
        else:
            quadratic_config = QuadraticItemConfig(
                method=method, mask_threshold=mask_threshold
            )
        return SupermodScenarioBuilder(
            replace(self._params, quadratic_config=quadratic_config)
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
                "subproblem": {"name": "QuadSupermodularNetwork"},
                "row_generation": {
                    "max_iters": 100,
                    "tolerance_optimality": 0.001,
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
                agent_modular = generator.generate_modular_agent(
                    (
                        params.num_agents,
                        params.num_items,
                        params.num_modular_agent_features,
                    ),
                    params.agent_config,
                )
                item_modular = generator.generate_modular_item(
                    (params.num_items, params.num_modular_item_features),
                    params.item_config,
                )
                item_quadratic = generator.generate_quadratic_item(
                    (
                        params.num_items,
                        params.num_items,
                        params.num_quadratic_item_features,
                    ),
                    params.quadratic_config,
                )
                errors = generator.generate_errors(
                    (params.num_simuls, params.num_agents, params.num_items), params.sigma
                )
                generation_data = {
                    "item_data": {
                        "modular": item_modular,
                        "quadratic": item_quadratic,
                    },
                    "agent_data": {
                        "modular": agent_modular,
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
                label=f"{params.num_agents}x{params.num_items}-quad-supermod",
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
            name="quadratic_supermodular",
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


def build() -> SupermodScenarioBuilder:
    return SupermodScenarioBuilder()


__all__ = ["SupermodScenarioBuilder", "build"]

