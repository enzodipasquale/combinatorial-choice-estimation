"""Scenario factory builder for the Gentzkow two-period setting with block diagonal quadratic features."""

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
class GentzkowParams:
    num_agents: int = 20
    num_items_per_period: int = 25  # Actual items per period
    num_modular_agent_features: int = 2
    num_modular_item_features: int = 2
    num_quadratic_item_features: int = 2
    num_simulations: int = 1
    sigma: float = 5.0
    sigma_time_invariant: float = 2.0  # Standard deviation for time-invariant error component
    time_invariant_alpha: float = 1.0  # Diffusion parameter for graph Laplacian correlation (alpha > 0)
    correlation_method: str = "laplacian"  # "laplacian" or "rho"
    time_invariant_rho: float = 1.0  # Correlation parameter for rho method (rho in [0,1])
    num_periods: int = 2
    # Modular features: time-invariant vs time-varying
    num_modular_agent_features_time_invariant: int = 0  # Time-invariant agent features
    num_modular_item_features_time_invariant: int = 0  # Time-invariant item features
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
    def num_items(self) -> int:
        """Total number of items (num_periods * num_items_per_period)."""
        return self.num_periods * self.num_items_per_period

    @property
    def num_features(self) -> int:
        return (
            self.num_modular_agent_features
            + self.num_modular_item_features
            + self.num_quadratic_item_features
        )
    
    @property
    def num_modular_agent_features_time_varying(self) -> int:
        """Time-varying agent features."""
        return self.num_modular_agent_features - self.num_modular_agent_features_time_invariant
    
    @property
    def num_modular_item_features_time_varying(self) -> int:
        """Time-varying item features."""
        return self.num_modular_item_features - self.num_modular_item_features_time_invariant

    @property
    def theta_star(self) -> np.ndarray:
        return np.ones(self.num_features)


class GentzkowScenarioBuilder:
    def __init__(self, params: Optional[GentzkowParams] = None, custom_theta: Optional[np.ndarray] = None) -> None:
        self._params = params or GentzkowParams()
        self._custom_theta = custom_theta

    def with_dimensions(
        self, *, num_agents: Optional[int] = None, num_items_per_period: Optional[int] = None
    ) -> "GentzkowScenarioBuilder":
        params = self._params
        if num_agents is not None:
            params = replace(params, num_agents=num_agents)
        if num_items_per_period is not None:
            params = replace(params, num_items_per_period=num_items_per_period)
        return GentzkowScenarioBuilder(params, self._custom_theta)

    def with_feature_counts(
        self,
        *,
        num_mod_agent: Optional[int] = None,
        num_mod_item: Optional[int] = None,
        num_quad_item: Optional[int] = None,
    ) -> "GentzkowScenarioBuilder":
        params = self._params
        if num_mod_agent is not None:
            params = replace(params, num_modular_agent_features=num_mod_agent)
        if num_mod_item is not None:
            params = replace(params, num_modular_item_features=num_mod_item)
        if num_quad_item is not None:
            params = replace(params, num_quadratic_item_features=num_quad_item)
        return GentzkowScenarioBuilder(params, self._custom_theta)

    def with_sigma(self, sigma: float) -> "GentzkowScenarioBuilder":
        return GentzkowScenarioBuilder(replace(self._params, sigma=sigma), self._custom_theta)

    def with_sigma_time_invariant(self, sigma_time_invariant: float) -> "GentzkowScenarioBuilder":
        """Set standard deviation for time-invariant error component. Set to 0 to disable correlated errors."""
        return GentzkowScenarioBuilder(replace(self._params, sigma_time_invariant=sigma_time_invariant), self._custom_theta)

    def with_time_invariant_alpha(self, alpha: float) -> "GentzkowScenarioBuilder":
        """Set diffusion parameter for graph Laplacian correlation. Small alpha = weak correlations, large alpha = strong correlations."""
        return GentzkowScenarioBuilder(replace(self._params, time_invariant_alpha=alpha), self._custom_theta)
    
    def with_correlation_method(self, method: str) -> "GentzkowScenarioBuilder":
        """Set correlation method: 'laplacian' or 'rho'."""
        if method not in ["laplacian", "rho"]:
            raise ValueError(f"correlation_method must be 'laplacian' or 'rho', got '{method}'")
        return GentzkowScenarioBuilder(replace(self._params, correlation_method=method), self._custom_theta)
    
    def with_time_invariant_rho(self, rho: float) -> "GentzkowScenarioBuilder":
        """Set rho parameter for rho correlation method (rho in [0,1])."""
        if not 0 <= rho <= 1:
            raise ValueError(f"rho must be in [0,1], got {rho}")
        return GentzkowScenarioBuilder(replace(self._params, time_invariant_rho=rho), self._custom_theta)

    def with_num_simulations(self, num_simulations: int) -> "GentzkowScenarioBuilder":
        return GentzkowScenarioBuilder(replace(self._params, num_simulations=num_simulations), self._custom_theta)

    def with_theta(self, theta: np.ndarray) -> "GentzkowScenarioBuilder":
        """Set custom theta vector for bundle generation."""
        return GentzkowScenarioBuilder(self._params, custom_theta=theta.copy())

    def with_agent_modular_config(
        self,
        multiplier: float = -2.0,
        mean: float = 2.0,
        std: float = 1.0,
    ) -> "GentzkowScenarioBuilder":
        """Configure modular agent feature generation."""
        agent_config = ModularAgentConfig(
            multiplier=multiplier, mean=mean, std=std, apply_abs=True
        )
        return GentzkowScenarioBuilder(replace(self._params, agent_config=agent_config), self._custom_theta)

    def with_item_modular_config(
        self,
        multiplier: float = -2.0,
        mean: float = 2.0,
        std: float = 1.0,
    ) -> "GentzkowScenarioBuilder":
        """Configure modular item feature generation."""
        item_config = ModularItemConfig(
            multiplier=multiplier, mean=mean, std=std, apply_abs=True
        )
        return GentzkowScenarioBuilder(replace(self._params, item_config=item_config), self._custom_theta)

    def with_time_invariant_modular_features(
        self,
        num_agent_time_invariant: Optional[int] = None,
        num_item_time_invariant: Optional[int] = None,
    ) -> "GentzkowScenarioBuilder":
        """Set number of time-invariant modular features. If None, all features are time-invariant."""
        params = self._params
        # Handle agent features
        if num_agent_time_invariant is not None:
            params = replace(params, num_modular_agent_features_time_invariant=num_agent_time_invariant)
        elif params.num_modular_agent_features > 0:
            # If not specified but we have agent features, make all time-invariant
            params = replace(params, num_modular_agent_features_time_invariant=params.num_modular_agent_features)
        # Handle item features
        if num_item_time_invariant is not None:
            params = replace(params, num_modular_item_features_time_invariant=num_item_time_invariant)
        elif params.num_modular_item_features > 0:
            # If not specified but we have item features, make all time-invariant
            params = replace(params, num_modular_item_features_time_invariant=params.num_modular_item_features)
        return GentzkowScenarioBuilder(params, self._custom_theta)

    def with_quadratic_method(
        self,
        method: QuadraticGenerationMethod,
        binary_prob: float = 0.2,
        binary_value: float = 1.0,
        mask_threshold: float = 0.3,
    ) -> "GentzkowScenarioBuilder":
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
        return GentzkowScenarioBuilder(
            replace(self._params, quadratic_config=quadratic_config), self._custom_theta
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
                "subproblem": {"name": "QuadSupermodularNetwork"},
                "row_generation": {
                    "max_iters": 100,
                    "tolerance_optimality": 0.001,
                    "min_iters": 1,
                    "gurobi_settings": {"OutputFlag": 0},
                    "theta_lbs": (
                        [-100.0] * params.num_modular_agent_features
                        + [-100.0] * params.num_modular_item_features
                        + [0.0] * params.num_quadratic_item_features
                    ),
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
                # Generate modular agent features
                # If time-invariant features are specified, generate only for period 0 and copy to period 1
                if params.num_modular_agent_features_time_invariant > 0:
                    # Generate for period 0 only
                    agent_modular_period0 = generator.generate_modular_agent(
                        (
                            params.num_agents,
                            params.num_items_per_period,
                            params.num_modular_agent_features_time_invariant,
                        ),
                        params.agent_config,
                    )
                    # Copy to all periods
                    agent_modular = np.zeros((params.num_agents, params.num_items, params.num_modular_agent_features))
                    for period in range(params.num_periods):
                        start_idx = period * params.num_items_per_period
                        end_idx = (period + 1) * params.num_items_per_period
                        agent_modular[:, start_idx:end_idx, :params.num_modular_agent_features_time_invariant] = agent_modular_period0
                    
                    # Generate time-varying features if any
                    if params.num_modular_agent_features_time_varying > 0:
                        agent_modular_time_varying = generator.generate_modular_agent(
                            (
                                params.num_agents,
                                params.num_items,
                                params.num_modular_agent_features_time_varying,
                            ),
                            params.agent_config,
                        )
                        agent_modular[:, :, params.num_modular_agent_features_time_invariant:] = agent_modular_time_varying
                else:
                    # All features are time-varying
                    agent_modular = generator.generate_modular_agent(
                        (
                            params.num_agents,
                            params.num_items,
                            params.num_modular_agent_features,
                        ),
                        params.agent_config,
                    )

                # Generate modular item features
                # If time-invariant features are specified, generate only for period 0 and copy to period 1
                if params.num_modular_item_features_time_invariant > 0:
                    # Generate for period 0 only
                    item_modular_period0 = generator.generate_modular_item(
                        (params.num_items_per_period, params.num_modular_item_features_time_invariant),
                        params.item_config,
                    )
                    # Copy to all periods
                    item_modular = np.zeros((params.num_items, params.num_modular_item_features))
                    for period in range(params.num_periods):
                        start_idx = period * params.num_items_per_period
                        end_idx = (period + 1) * params.num_items_per_period
                        item_modular[start_idx:end_idx, :params.num_modular_item_features_time_invariant] = item_modular_period0
                    
                    # Generate time-varying features if any
                    if params.num_modular_item_features_time_varying > 0:
                        item_modular_time_varying = generator.generate_modular_item(
                            (params.num_items, params.num_modular_item_features_time_varying),
                            params.item_config,
                        )
                        item_modular[:, params.num_modular_item_features_time_invariant:] = item_modular_time_varying
                else:
                    # All features are time-varying
                    item_modular = generator.generate_modular_item(
                        (params.num_items, params.num_modular_item_features),
                        params.item_config,
                    )

                # Generate time-invariant block diagonal quadratic item features
                # Q_11 = Q_22 = Q (time-invariant complementarities)
                item_quadratic = self._generate_time_invariant_block_diagonal_quadratic(
                    generator,
                    params.num_items_per_period,
                    params.num_periods,
                    params.num_quadratic_item_features,
                    params.quadratic_config,
                )

                # Generate errors with time-invariant component
                errors = self._generate_gentzkow_errors(
                    generator,
                    params,
                    item_quadratic,  # Use Q structure to determine which items correlate
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

            obs_bundles = bc.subproblems.init_and_solve(theta)

            estimation_data = None
            if rank == 0:
                # For estimation, generate errors with same structure
                # Shape is (num_simuls, num_agents, num_items)
                estimation_errors = np.zeros((params.num_simulations, params.num_agents, params.num_items))
                
                # Time-invariant errors are the same across all simulations (they don't depend on time)
                # Generate once and reuse (skip if sigma_time_invariant=0)
                if params.sigma_time_invariant > 0:
                    if params.correlation_method == "rho":
                        time_invariant_errors = self._generate_time_invariant_errors_with_rho(
                            generator,
                            params.num_agents,
                            params.num_items,
                            params.num_items_per_period,
                            params.num_periods,
                            params.sigma_time_invariant,
                            params.time_invariant_rho,
                            item_quadratic,
                        )
                    else:  # laplacian (default)
                        time_invariant_errors = self._generate_time_invariant_errors_with_q_structure(
                            generator,
                            params.num_agents,
                            params.num_items,
                            params.num_items_per_period,
                            params.num_periods,
                            params.sigma_time_invariant,
                            params.time_invariant_alpha,
                            item_quadratic,
                        )
                    
                    # Generate i.i.d. errors for each simulation (these vary across simulations)
                    for simul in range(params.num_simulations):
                        iid_errors = generator.generate_errors(
                            (params.num_agents, params.num_items), params.sigma
                        )
                        estimation_errors[simul] = iid_errors + time_invariant_errors
                else:
                    # No time-invariant component, just i.i.d. errors
                    for simul in range(params.num_simulations):
                        iid_errors = generator.generate_errors(
                            (params.num_agents, params.num_items), params.sigma
                        )
                        estimation_errors[simul] = iid_errors
                
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

        # Use custom theta if provided, otherwise use default
        if self._custom_theta is not None:
            theta_factory = lambda: self._custom_theta.copy()
        else:
            theta_factory = lambda: params.theta_star.copy()
        
        return SyntheticScenario(
            name="gentzkow",
            config_factory=config_factory,
            feature_spec=feature_spec,
            payload_factory=payload_factory,
            theta_factory=theta_factory,
            metadata={
                "num_agents": params.num_agents,
                "num_items": params.num_items,
                "num_items_per_period": params.num_items_per_period,
                "num_periods": params.num_periods,
                "num_features": params.num_features,
                "num_simulations": params.num_simulations,
                "sigma": params.sigma,
            },
        )

    def _generate_time_invariant_block_diagonal_quadratic(
        self,
        generator: DataGenerator,
        num_items_per_period: int,
        num_periods: int,
        num_features: int,
        config: QuadraticItemConfig,
    ) -> np.ndarray:
        """
        Generate time-invariant block diagonal quadratic item features.
        
        Creates a (num_items, num_items, num_features) array where:
        - Q_11 = Q_22 = ... = Q (time-invariant complementarities)
        - Each feature has non-zero entries only within each period (block diagonal structure).
        """
        total_items = num_periods * num_items_per_period
        data = np.zeros((total_items, total_items, num_features))

        # Generate Q once (time-invariant)
        Q = generator.generate_quadratic_item(
            (num_items_per_period, num_items_per_period, num_features),
            config,
        )

        # Place Q in each period's block (Q_11 = Q_22 = Q)
        for period in range(num_periods):
            start_idx = period * num_items_per_period
            end_idx = (period + 1) * num_items_per_period
            data[start_idx:end_idx, start_idx:end_idx, :] = Q

        return data

    def _generate_gentzkow_errors(
        self,
        generator: DataGenerator,
        params: GentzkowParams,
        item_quadratic: np.ndarray,
    ) -> np.ndarray:
        """
        Generate errors with time-invariant component having Q-matching correlation structure.
        
        Error structure:
        - epsilon_ijt = iid_error_ijt + time_invariant_error_ij
        - iid_error_ijt: i.i.d. across all items and times
        - time_invariant_error_ij: does NOT depend on time, has correlation structure matching Q
          (block diagonal: items in same period are correlated, different periods are not)
        
        Args:
            params: GentzkowParams containing all parameters
            item_quadratic: (num_items, num_items, num_features) - used to determine correlation structure
        """
        # Generate i.i.d. errors (for generation stage, shape is (num_agents, num_items))
        iid_errors = generator.generate_errors(
            (params.num_agents, params.num_items), params.sigma
        )
        
        # Generate time-invariant errors with Q-matching correlation structure
        # The correlation structure matches Q: block diagonal
        # Skip if sigma_time_invariant=0 (no time-invariant component)
        if params.sigma_time_invariant > 0:
            if params.correlation_method == "rho":
                time_invariant_errors = self._generate_time_invariant_errors_with_rho(
                    generator,
                    params.num_agents,
                    params.num_items,
                    params.num_items_per_period,
                    params.num_periods,
                    params.sigma_time_invariant,
                    params.time_invariant_rho,
                    item_quadratic,
                )
            else:  # laplacian (default)
                time_invariant_errors = self._generate_time_invariant_errors_with_q_structure(
                    generator,
                    params.num_agents,
                    params.num_items,
                    params.num_items_per_period,
                    params.num_periods,
                    params.sigma_time_invariant,
                    params.time_invariant_alpha,
                    item_quadratic,
                )
            # Total errors = iid + time-invariant
            errors = iid_errors + time_invariant_errors
        else:
            # No time-invariant component, just i.i.d. errors
            errors = iid_errors
        
        return errors

    def _generate_time_invariant_errors_with_q_structure(
        self,
        generator: DataGenerator,
        num_agents: int,
        num_items: int,
        num_items_per_period: int,
        num_periods: int,
        sigma: float,
        alpha: float,
        item_quadratic: np.ndarray,
    ) -> np.ndarray:
        """
        Generate time-invariant errors with correlation structure from graph Laplacian of Q.
        
        Given Q (quadratic item features), construct correlation matrix using graph Laplacian:
        1. Compute degree matrix D_ii = sum_j Q_ij
        2. Form graph Laplacian L = D - Q
        3. Compute kernel K = exp(-alpha * L)
        4. Normalize to correlation matrix R_ij = K_ij / sqrt(K_ii * K_jj)
        
        The correlation structure is block diagonal matching Q:
        - Items in same period are correlated (based on Q structure)
        - Items in different periods are independent (correlation = 0)
        
        Args:
            alpha: Diffusion parameter (alpha > 0). Small alpha = weak local correlations,
                   large alpha = stronger global correlations.
        """
        from scipy.linalg import expm
        
        errors = np.zeros((num_agents, num_items))
        
        # IMPORTANT: Time-invariant errors are the SAME across periods.
        # Generate errors once for period 0, then copy to all periods.
        period = 0
        start_idx = period * num_items_per_period
        end_idx = (period + 1) * num_items_per_period
        period_items = num_items_per_period
        
        # Extract Q block for period 0 (same for all periods since Q is time-invariant)
        # Sum Q across all features to get connection strength matrix
        Q_period = item_quadratic[start_idx:end_idx, start_idx:end_idx, :]
        Q_sum = Q_period.sum(axis=2)  # Sum across features: (period_items, period_items)
        
        # Make Q_sum symmetric (since it's upper triangular, reflect it)
        Q_symmetric = Q_sum + Q_sum.T - np.diag(np.diag(Q_sum))
        
        # Step 1: Compute degree matrix D_ii = sum_j Q_ij
        D = np.diag(Q_symmetric.sum(axis=1))
        
        # Step 2: Form graph Laplacian L = D - Q
        L = D - Q_symmetric
        
        # Step 3: Compute kernel K = exp(-alpha * L)
        K = expm(-alpha * L)
        
        # Step 4: Normalize to correlation matrix R_ij = K_ij / sqrt(K_ii * K_jj)
        K_diag_sqrt = np.sqrt(np.diag(K))
        corr_matrix = K / np.outer(K_diag_sqrt, K_diag_sqrt)
        
        # Ensure positive semidefinite (should be by construction, but add small regularization if needed)
        corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Ensure symmetry
        eigenvals = np.linalg.eigvals(corr_matrix)
        if np.any(eigenvals < -1e-10):
            # Add small regularization to ensure positive semidefinite
            corr_matrix += np.eye(period_items) * (1e-10 - np.min(eigenvals))
        
        # Generate correlated errors ONCE for period 0
        # For each agent, generate period_items correlated errors
        time_invariant_errors = np.zeros((num_agents, period_items))
        for agent in range(num_agents):
            try:
                period_errors = generator.rng.multivariate_normal(
                    np.zeros(period_items),
                    sigma**2 * corr_matrix,
                )
            except np.linalg.LinAlgError:
                # If correlation matrix is not positive definite, use independent errors
                period_errors = generator.rng.normal(0, sigma, period_items)
            time_invariant_errors[agent, :] = period_errors
        
        # Copy the SAME time-invariant errors to all periods
        for period in range(num_periods):
            start_idx = period * num_items_per_period
            end_idx = (period + 1) * num_items_per_period
            errors[:, start_idx:end_idx] = time_invariant_errors
        
        return errors
    
    def _generate_time_invariant_errors_with_rho(
        self,
        generator: DataGenerator,
        num_agents: int,
        num_items: int,
        num_items_per_period: int,
        num_periods: int,
        sigma: float,
        rho: float,
        item_quadratic: np.ndarray,
    ) -> np.ndarray:
        """
        Generate time-invariant errors with rho-based correlation structure.
        
        Given:
        - A is an n x n matrix with unit Euclidean norm rows (derived from Q)
        - rho is a parameter in [0,1]
        - x ~ N(0, I_n) is a standard normal vector
        - z ~ N(0, I_n) is an independent standard normal vector
        
        Define: y = sqrt(rho) * (A @ x) + sqrt(1 - rho) * z
        
        Properties:
        - Each entry of y has variance 1
        - Cov(y) = rho * (A @ A.T) + (1 - rho) * I
        - rho = 1 gives fully correlated, rho = 0 gives independent
        
        Args:
            rho: Correlation parameter in [0,1]
        """
        errors = np.zeros((num_agents, num_items))
        
        # Time-invariant errors are the SAME across periods.
        # Generate errors once for period 0, then copy to all periods.
        period = 0
        start_idx = period * num_items_per_period
        end_idx = (period + 1) * num_items_per_period
        period_items = num_items_per_period
        
        # Extract Q block for period 0
        Q_period = item_quadratic[start_idx:end_idx, start_idx:end_idx, :]
        Q_sum = Q_period.sum(axis=2)  # Sum across features: (period_items, period_items)
        
        # Make Q_sum symmetric
        Q_symmetric = Q_sum + Q_sum.T - np.diag(np.diag(Q_sum))
        
        # Create matrix A from Q: normalize rows to unit Euclidean norm
        # A will be used to define the correlation structure
        A = Q_symmetric.copy()
        row_norms = np.linalg.norm(A, axis=1, keepdims=True)
        # Avoid division by zero (if a row is all zeros, keep it as zeros)
        row_norms = np.where(row_norms > 1e-10, row_norms, 1.0)
        A = A / row_norms  # Now each row has unit norm
        
        # Generate correlated errors ONCE for period 0
        time_invariant_errors = np.zeros((num_agents, period_items))
        for agent in range(num_agents):
            # Generate independent standard normal vectors
            x = generator.rng.normal(0, 1, period_items)  # x ~ N(0, I)
            z = generator.rng.normal(0, 1, period_items)  # z ~ N(0, I), independent of x
            
            # Compute: y = sqrt(rho) * (A @ x) + sqrt(1 - rho) * z
            y = np.sqrt(rho) * (A @ x) + np.sqrt(1 - rho) * z
            
            # Scale by sigma to get desired variance
            time_invariant_errors[agent, :] = sigma * y
        
        # Copy the SAME time-invariant errors to all periods
        for period in range(num_periods):
            start_idx = period * num_items_per_period
            end_idx = (period + 1) * num_items_per_period
            errors[:, start_idx:end_idx] = time_invariant_errors
        
        return errors


def build() -> GentzkowScenarioBuilder:
    return GentzkowScenarioBuilder()


__all__ = ["GentzkowScenarioBuilder", "build"]

