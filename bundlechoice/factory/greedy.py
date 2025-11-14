"""Scenario factory builder for the Greedy subproblem."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional

import numpy as np
from mpi4py import MPI

from bundlechoice.core import BundleChoice

from .base import FeatureSpec, SyntheticScenario
from .data_generator import DataGenerator, ModularAgentConfig
from . import utils


@dataclass(frozen=True)
class GreedyParams:
    num_agents: int = 200
    num_items: int = 150
    num_features: int = 4
    num_simuls: int = 1
    sigma: float = 1.0
    theta_star: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0, 0.1]))
    agent_config: ModularAgentConfig = field(default_factory=lambda: ModularAgentConfig(apply_abs=True))


class GreedyScenarioBuilder:
    """Builder-style interface for the Greedy scenario factory."""

    def __init__(self, params: Optional[GreedyParams] = None) -> None:
        self._params = params or GreedyParams()
        if self._params.theta_star.shape[0] != self._params.num_features:
            raise ValueError("theta_star length must match num_features")

    def with_dimensions(
        self, *, num_agents: Optional[int] = None, num_items: Optional[int] = None
    ) -> "GreedyScenarioBuilder":
        params = self._params
        if num_agents is not None:
            params = replace(params, num_agents=num_agents)
        if num_items is not None:
            params = replace(params, num_items=num_items)
        return GreedyScenarioBuilder(params)

    def with_num_features(self, num_features: int) -> "GreedyScenarioBuilder":
        theta = np.ones(num_features)
        theta[-1] = 0.1
        return GreedyScenarioBuilder(replace(self._params, num_features=num_features, theta_star=theta))

    def with_num_simuls(self, num_simuls: int) -> "GreedyScenarioBuilder":
        return GreedyScenarioBuilder(replace(self._params, num_simuls=num_simuls))

    def with_sigma(self, sigma: float) -> "GreedyScenarioBuilder":
        return GreedyScenarioBuilder(replace(self._params, sigma=sigma))

    def with_theta(self, theta_star: np.ndarray) -> "GreedyScenarioBuilder":
        if theta_star.ndim != 1:
            raise ValueError("theta_star must be 1-dimensional")
        return GreedyScenarioBuilder(replace(self._params, theta_star=theta_star))

    def with_agent_config(
        self,
        apply_abs: bool = True,
        multiplier: float = 1.0,
        mean: float = 0.0,
        std: float = 1.0,
    ) -> "GreedyScenarioBuilder":
        """Configure modular agent feature generation."""
        agent_config = ModularAgentConfig(
            apply_abs=apply_abs, multiplier=multiplier, mean=mean, std=std
        )
        return GreedyScenarioBuilder(replace(self._params, agent_config=agent_config))

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
                "subproblem": {"name": "Greedy"},
                "row_generation": {
                    "max_iters": 100,
                    "tolerance_optimality": 0.001,
                    "min_iters": 1,
                    "gurobi_settings": {"OutputFlag": 0},
                    "theta_ubs": 100,
                },
            }

        feature_spec = FeatureSpec.oracle(_greedy_features_oracle)

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
                modular = generator.generate_modular_agent(
                    (params.num_agents, params.num_items, params.num_features - 1),
                    params.agent_config,
                )
                errors = generator.generate_errors(
                    (params.num_agents, params.num_items), params.sigma
                )
                generation_data = {
                    "agent_data": {"modular": modular},
                    "errors": errors,
                }

            bc.data.load_and_scatter(generation_data if rank == 0 else None)
            spec.initializer(bc)

            def solve() -> np.ndarray:
                return bc.subproblems.init_and_solve(params.theta_star)

            obs_bundles = utils.mpi_call_with_timeout(
                comm, solve, timeout, label=f"{params.num_agents}x{params.num_items}-greedy"
            )

            estimation_data = None
            if rank == 0:
                estimation_errors = generator.generate_errors(
                    (params.num_simuls, params.num_agents, params.num_items), params.sigma
                )
                estimation_data = {
                    "agent_data": generation_data["agent_data"],
                    "errors": estimation_errors,
                    "obs_bundle": obs_bundles,
                }

            return {
                "generation": utils.root_dict(comm, generation_data),
                "estimation": utils.root_dict(comm, estimation_data),
            }

        return SyntheticScenario(
            name="greedy",
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


def _greedy_features_oracle(agent_idx: int, bundles: np.ndarray, data: Dict[str, Any]) -> np.ndarray:
    modular_agent = data["agent_data"]["modular"][agent_idx]

    if bundles.ndim == 1:
        return np.concatenate((modular_agent.T @ bundles, [-bundles.sum() ** 2]))
    else:
        return np.concatenate((modular_agent.T @ bundles, -np.sum(bundles, axis=0, keepdims=True) ** 2), axis=0)


def build() -> GreedyScenarioBuilder:
    return GreedyScenarioBuilder()


__all__ = ["GreedyScenarioBuilder", "build"]

