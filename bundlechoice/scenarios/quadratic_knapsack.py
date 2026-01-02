"""Scenario factory builder for the QuadraticKnapsack subproblem."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, Union

import numpy as np
from mpi4py import MPI

from bundlechoice.core import BundleChoice

from .base import FeatureSpec, SyntheticScenario
from .data_generator import (
    CapacityConfig,
    DataGenerator,
    EndogeneityConfig,
    ModularAgentConfig,
    ModularItemConfig,
    QuadraticGenerationMethod,
    QuadraticItemConfig,
    WeightConfig,
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
    num_simulations: int = 1
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
    weight_config: WeightConfig = field(default_factory=WeightConfig)
    capacity_config: CapacityConfig = field(default_factory=CapacityConfig)
    subproblem_settings: Dict[str, Any] = field(default_factory=dict)
    endogeneity_config: Optional[EndogeneityConfig] = None

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

    def with_num_simulations(self, num_simulations: int) -> "QuadraticKnapsackScenarioBuilder":
        return QuadraticKnapsackScenarioBuilder(
            replace(self._params, num_simulations=num_simulations)
        )

    def with_weight_config(
        self,
        distribution: str = "uniform",
        low: int = 1,
        high: int = 10,
        log_mean: float = 0.0,
        log_std: float = 1.0,
        exp_scale: float = 2.0,
    ) -> "QuadraticKnapsackScenarioBuilder":
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
        return QuadraticKnapsackScenarioBuilder(
            replace(self._params, weight_config=weight_config)
        )

    def with_capacity_config(
        self,
        mean_multiplier: float = 0.45,
        lower_multiplier: float = 0.85,
        upper_multiplier: float = 1.15,
    ) -> "QuadraticKnapsackScenarioBuilder":
        """Configure capacity generation using variance-based method (legacy)."""
        capacity_config = CapacityConfig(
            mean_multiplier=mean_multiplier,
            lower_multiplier=lower_multiplier,
            upper_multiplier=upper_multiplier,
        )
        return QuadraticKnapsackScenarioBuilder(
            replace(self._params, capacity_config=capacity_config)
        )

    def with_capacity_fraction(
        self,
        fraction: float,
    ) -> "QuadraticKnapsackScenarioBuilder":
        """Configure capacity as fixed fraction of total weight (most standard method).
        
        Args:
            fraction: Fraction of total weight sum (e.g., 0.5 for 50%).
        """
        capacity_config = CapacityConfig(fraction=fraction)
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

    def with_subproblem_settings(
        self, **settings: Any
    ) -> "QuadraticKnapsackScenarioBuilder":
        """Configure subproblem solver settings (e.g., OutputFlag, TimeLimit)."""
        # Merge with existing settings
        merged_settings = {**self._params.subproblem_settings, **settings}
        return QuadraticKnapsackScenarioBuilder(
            replace(self._params, subproblem_settings=merged_settings)
        )

    def with_endogeneity(
        self,
        endogenous_feature_indices: list[int],
        num_instruments: int,
        *,
        pi_matrix: Optional[np.ndarray] = None,
        pi_config: Optional[dict] = None,
        lambda_matrix: Optional[np.ndarray] = None,
        lambda_config: Optional[dict] = None,
        xi_cov: Optional[Union[float, np.ndarray]] = None,
        instrument_cov: Optional[np.ndarray] = None,
        instrument_config: Optional[ModularItemConfig] = None,
        ensure_full_rank: bool = True,
    ) -> "QuadraticKnapsackScenarioBuilder":
        """Enable endogeneity in modular item features following BLP structure."""
        endogeneity_config = EndogeneityConfig(
            endogenous_feature_indices=endogenous_feature_indices,
            num_instruments=num_instruments,
            pi_matrix=pi_matrix,
            pi_config=pi_config,
            lambda_matrix=lambda_matrix,
            lambda_config=lambda_config,
            xi_cov=xi_cov,
            instrument_cov=instrument_cov,
            instrument_config=instrument_config,
            ensure_full_rank=ensure_full_rank,
        )
        return QuadraticKnapsackScenarioBuilder(
            replace(self._params, endogeneity_config=endogeneity_config)
        )

    def build(self) -> SyntheticScenario:
        params = self._params

        def config_factory() -> Dict[str, Any]:
            return {
                "dimensions": {
                    "num_agents": params.num_agents,
                    "num_items": params.num_items,
                    "num_features": params.num_features,
                    "num_simulations": 1,  # Always 1 for generation stage
                },
                "subproblem": {
                    "name": "QuadKnapsack",
                    "settings": params.subproblem_settings,
                },
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
            theta: Any,
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
                # Generate base modular item features
                base_item_modular = generator.generate_modular_item(
                    (params.num_items, params.num_item_modular_features),
                    params.item_config,
                )
                
                # Apply endogeneity if configured
                endogeneity_metadata = {}
                if params.endogeneity_config:
                    item_modular, instruments, xi, v = generator.generate_endogenous_modular_item(
                        base_item_modular, params.endogeneity_config
                    )
                    errors = generator.generate_errors_with_endogeneity(
                        (params.num_agents, params.num_items), params.sigma, xi
                    )
                    # Store for IV regression
                    endogeneity_metadata = {
                        "instruments": instruments,
                        "xi": xi,
                        "original_modular_item": base_item_modular.copy(),
                    }
                else:
                    item_modular = base_item_modular
                    errors = generator.generate_errors(
                        (params.num_agents, params.num_items), params.sigma
                    )
                
                item_quadratic = generator.generate_quadratic_item(
                    (
                        params.num_items,
                        params.num_items,
                        params.num_item_quadratic_features,
                    ),
                    params.item_quadratic_config,
                )
                item_weights = generator.generate_weights(params.num_items, params.weight_config)
                agent_capacity = generator.generate_capacities(
                    params.num_agents, item_weights, params.capacity_config
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
                # Add endogeneity metadata if present
                generation_data.update(endogeneity_metadata)

            bc.data.load_and_scatter(generation_data if rank == 0 else None)
            spec.initializer(bc)

            obs_bundles = bc.subproblems.init_and_solve(theta)

            estimation_data = None
            if rank == 0:
                estimation_errors = generator.generate_errors(
                    (params.num_simulations, params.num_agents, params.num_items), params.sigma
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
                "num_simulations": params.num_simulations,
                "sigma": params.sigma,
            },
        )


def build() -> QuadraticKnapsackScenarioBuilder:
    return QuadraticKnapsackScenarioBuilder()


__all__ = ["QuadraticKnapsackScenarioBuilder", "build"]

