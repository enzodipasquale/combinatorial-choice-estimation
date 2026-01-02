"""Scenario factory builder for the Greedy subproblem."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, Union

import numpy as np
from mpi4py import MPI

from bundlechoice.core import BundleChoice

from .base import FeatureSpec, SyntheticScenario
from .data_generator import (
    DataGenerator,
    EndogeneityConfig,
    ModularAgentConfig,
    ModularItemConfig,
)
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
    item_config: Optional[ModularItemConfig] = None
    endogeneity_config: Optional[EndogeneityConfig] = None


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

    def with_item_config(
        self,
        apply_abs: bool = False,
        multiplier: float = 1.0,
        mean: float = 0.0,
        std: float = 1.0,
    ) -> "GreedyScenarioBuilder":
        """Configure modular item feature generation."""
        item_config = ModularItemConfig(
            apply_abs=apply_abs, multiplier=multiplier, mean=mean, std=std
        )
        return GreedyScenarioBuilder(replace(self._params, item_config=item_config))

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
    ) -> "GreedyScenarioBuilder":
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
        return GreedyScenarioBuilder(
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
                    "num_simuls": 1,  # Always 1 for generation stage
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

        def _greedy_feature_initializer(bc: BundleChoice) -> None:
            """Initialize features for greedy factory."""
            bc.features.set_oracle(_greedy_features_oracle)
            # Install find_best_item after subproblems are loaded
            # This will be called from PreparedScenario.apply() after subproblems.load()
            # Also called from payload_factory after subproblems.load() for data generation
            if bc.subproblems.subproblem_instance is not None:
                from bundlechoice.subproblems.registry.greedy import GreedySubproblem
                if isinstance(bc.subproblems.subproblem_instance, GreedySubproblem):
                    _install_find_best_item(bc.subproblems.subproblem_instance)
        
        feature_spec = FeatureSpec(mode="oracle", initializer=_greedy_feature_initializer)

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
                # Generate item features if endogeneity is enabled or item_config is provided
                if params.endogeneity_config is not None or params.item_config is not None:
                    # Determine number of item features needed
                    if params.endogeneity_config:
                        # Need at least max(endogenous_feature_indices) + 1 features
                        num_item_features = max(params.endogeneity_config.endogenous_feature_indices) + 1
                    else:
                        # Default to 1 if only item_config provided
                        num_item_features = 1
                    
                    # When item features exist, split modular features: agent + item + quadratic
                    # num_features = num_agent_features + num_item_features + 1 (quadratic)
                    num_agent_features = params.num_features - num_item_features - 1
                    modular_agent = generator.generate_modular_agent(
                        (params.num_agents, params.num_items, num_agent_features),
                        params.agent_config,
                    )
                    
                    item_config = params.item_config or ModularItemConfig(apply_abs=True)
                    base_modular_item = generator.generate_modular_item(
                        (params.num_items, num_item_features), item_config
                    )
                    
                    # Apply endogeneity if configured
                    endogeneity_metadata = {}
                    if params.endogeneity_config:
                        modular_item, instruments, xi, v = generator.generate_endogenous_modular_item(
                            base_modular_item, params.endogeneity_config
                        )
                        errors = generator.generate_errors_with_endogeneity(
                            (params.num_agents, params.num_items), params.sigma, xi
                        )
                        # Store for IV regression
                        endogeneity_metadata = {
                            "instruments": instruments,
                            "xi": xi,
                            "original_modular_item": base_modular_item.copy(),
                        }
                    else:
                        modular_item = base_modular_item
                        errors = generator.generate_errors(
                            (params.num_agents, params.num_items), params.sigma
                        )
                    
                    generation_data = {
                        "agent_data": {"modular": modular_agent},
                        "item_data": {"modular": modular_item},
                        "errors": errors,
                    }
                    # Add endogeneity metadata if present
                    generation_data.update(endogeneity_metadata)
                else:
                    # No item features (original behavior)
                    # num_features = num_agent_features + 1 (quadratic)
                    modular_agent = generator.generate_modular_agent(
                        (params.num_agents, params.num_items, params.num_features - 1),
                        params.agent_config,
                    )
                    errors = generator.generate_errors(
                        (params.num_agents, params.num_items), params.sigma
                    )
                    generation_data = {
                        "agent_data": {"modular": modular_agent},
                        "errors": errors,
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
                    (params.num_simuls, params.num_agents, params.num_items), params.sigma
                )
                estimation_data = {
                    "agent_data": generation_data["agent_data"],
                    "errors": estimation_errors,
                    "obs_bundle": obs_bundles,
                }
                # Include item_data if it was generated
                if "item_data" in generation_data:
                    estimation_data["item_data"] = generation_data["item_data"]

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
    """
    Compute features for greedy factory.
    Without item features: [modular_agent.T @ bundle, -|bundle|^2]
    With item features: [modular_agent.T @ bundle, modular_item.T @ bundle, -|bundle|^2]
    """
    modular_agent = data["agent_data"]["modular"][agent_idx]
    
    # Check if item features exist (when endogeneity is enabled)
    has_item_features = (
        "item_data" in data 
        and data["item_data"] is not None 
        and "modular" in data["item_data"]
    )
    
    if bundles.ndim == 1:
        agent_features = modular_agent.T @ bundles
        if has_item_features:
            modular_item = data["item_data"]["modular"]
            item_features = modular_item.T @ bundles
            return np.concatenate((agent_features, item_features, [-bundles.sum() ** 2]))
        else:
            return np.concatenate((agent_features, [-bundles.sum() ** 2]))
    else:
        agent_features = modular_agent.T @ bundles
        if has_item_features:
            modular_item = data["item_data"]["modular"]
            item_features = modular_item.T @ bundles
            return np.concatenate((agent_features, item_features, -np.sum(bundles, axis=0, keepdims=True) ** 2), axis=0)
        else:
            return np.concatenate((agent_features, -np.sum(bundles, axis=0, keepdims=True) ** 2), axis=0)


def _find_best_item(
    self, local_id: int, base_bundle: np.ndarray, items_left: np.ndarray,
    theta: np.ndarray, error_j: np.ndarray
) -> tuple[int, float]:
    """
    find_best_item for modular + quadratic cost structure.
    
    For each j in items_left, computes the full value of bundle = base + {j}:
    value = modular_features(base + j) @ theta + error_j[j] + quadratic_term
    where quadratic_term = -|base + j|^2 = -(base_size + 1)^2 (same for all j)
    """
    modular_agent = self.local_data["agent_data"]["modular"][local_id]
    base_size = base_bundle.sum()
    new_size = base_size + 1
    theta_quadratic = theta[-1]
    theta_modular = theta[:-1]
    
    # Quadratic term is the same for all candidate bundles (base + j)
    quadratic_term = theta_quadratic * (-new_size ** 2)
    
    # Check if item features exist (when endogeneity is enabled)
    has_item_features = (
        "item_data" in self.local_data 
        and self.local_data["item_data"] is not None
        and "modular" in self.local_data["item_data"]
    )
    
    if has_item_features:
        modular_item = self.local_data["item_data"]["modular"]
        num_agent_features = modular_agent.shape[1]
        num_item_features = modular_item.shape[1]
        
        # Split theta into agent and item parts
        theta_agent = theta_modular[:num_agent_features]
        theta_item = theta_modular[num_agent_features:num_agent_features + num_item_features]
        
        # For each j in items_left, compute modular features of bundle = base + {j}
        # modular_agent_features = modular_agent.T @ (base + e_j) = modular_agent.T @ base + modular_agent[j, :]
        base_modular_agent = modular_agent.T @ base_bundle  # (num_agent_features,)
        base_modular_item = modular_item.T @ base_bundle  # (num_item_features,)
        
        # Vectorized: compute modular features for all candidate bundles
        candidate_modular_agent = base_modular_agent[None, :] + modular_agent[items_left, :]  # (len(items_left), num_agent_features)
        candidate_modular_item = base_modular_item[None, :] + modular_item[items_left, :]  # (len(items_left), num_item_features)
        
        # Compute modular values for each candidate bundle
        modular_values = (
            candidate_modular_agent @ theta_agent +  # (len(items_left),)
            candidate_modular_item @ theta_item      # (len(items_left),)
        )
    else:
        # No item features: only agent features
        base_modular = modular_agent.T @ base_bundle  # (num_agent_features,)
        
        # For each j: modular_features = modular_agent.T @ (base + e_j) = base_modular + modular_agent[j, :]
        candidate_modular = base_modular[None, :] + modular_agent[items_left, :]  # (len(items_left), num_agent_features)
        modular_values = candidate_modular @ theta_modular  # (len(items_left),)
    
    # Full values: modular + errors + quadratic (quadratic is same for all, doesn't affect argmax)
    full_values = modular_values + error_j[items_left] + quadratic_term
    
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


def build() -> GreedyScenarioBuilder:
    return GreedyScenarioBuilder()


__all__ = ["GreedyScenarioBuilder", "build"]

