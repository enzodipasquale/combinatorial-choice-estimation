"""Base data generator for exogenous data generation (before BundleChoice)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional

import numpy as np

from . import utils


class QuadraticGenerationMethod(Enum):
    """Methods for generating quadratic item features."""

    BINARY_CHOICE = "binary_choice"  # random.choice([0, 1])
    EXPONENTIAL = "exponential"  # exp(-abs(normal))


@dataclass
class CorrelationConfig:
    """Configuration for feature correlation via matrix transformation."""

    enabled: bool = False
    matrix_range: tuple[int, int] = (0, 4)  # Range for random integers
    normalize: bool = True  # Whether to normalize columns


@dataclass
class ModularAgentConfig:
    """Configuration for modular agent feature generation."""

    multiplier: float = 1.0  # Multiplier applied to abs(normal)
    mean: float = 0.0  # Mean for normal distribution
    std: float = 1.0  # Std for normal distribution
    apply_abs: bool = False  # Whether to apply abs()
    correlation: CorrelationConfig = field(default_factory=CorrelationConfig)


@dataclass
class ModularItemConfig:
    """Configuration for modular item feature generation."""

    multiplier: float = 1.0
    mean: float = 0.0
    std: float = 1.0
    apply_abs: bool = False


@dataclass
class QuadraticItemConfig:
    """Configuration for quadratic item feature generation."""

    method: QuadraticGenerationMethod = QuadraticGenerationMethod.EXPONENTIAL
    # For BINARY_CHOICE:
    binary_prob: float = 0.2  # Probability of 1
    binary_value: float = 1.0  # Value when chosen
    # For EXPONENTIAL:
    exp_mean: float = 0.0
    exp_std: float = 1.0
    # Common:
    mask_threshold: float = 0.3  # Threshold for masking (for exponential method)
    zero_diagonal: bool = True  # Whether to zero out diagonal


@dataclass
class CapacityConfig:
    """Configuration for agent capacity generation (knapsack)."""

    mean_multiplier: float = 0.45  # Multiplier for mean capacity (0.5 in manual, 0.45 in factory)
    lower_multiplier: float = 0.85  # Lower bound multiplier
    upper_multiplier: float = 1.15  # Upper bound multiplier


class DataGenerator:
    """Base class for generating exogenous data before BundleChoice."""

    def __init__(self, seed: Optional[int] = None):
        """Initialize with optional seed."""
        self.rng = utils.rng(seed)

    def generate_modular_agent(
        self,
        shape: tuple[int, ...],
        config: ModularAgentConfig,
    ) -> np.ndarray:
        """Generate modular agent features with optional correlation."""
        data = self.rng.normal(config.mean, config.std, shape)
        
        # Apply correlation matrix BEFORE multiplier/abs (to match manual generation order)
        if config.correlation.enabled and len(shape) >= 2:
            data = self._apply_correlation_matrix(data, config.correlation)
        
        if config.apply_abs:
            data = np.abs(data)
        data = config.multiplier * data

        return data

    def generate_modular_item(
        self,
        shape: tuple[int, ...],
        config: ModularItemConfig,
    ) -> np.ndarray:
        """Generate modular item features."""
        data = self.rng.normal(config.mean, config.std, shape)
        if config.apply_abs:
            data = np.abs(data)
        return config.multiplier * data

    def generate_quadratic_item(
        self,
        shape: tuple[int, int, int],  # (num_items, num_items, num_features)
        config: QuadraticItemConfig,
    ) -> np.ndarray:
        """Generate quadratic item features."""
        num_items, _, num_features = shape

        if config.method == QuadraticGenerationMethod.BINARY_CHOICE:
            data = np.zeros(shape)
            for k in range(num_features):
                # Generate upper triangular matrix for supermodularity
                # For supermodularity: off-diagonal must be >= 0, upper triangular
                # First create upper triangular mask
                upper_mask = np.triu(np.ones((num_items, num_items)), k=1)
                # Generate binary values only in upper triangular part
                upper_tri = (
                    config.binary_value
                    * self.rng.choice(
                        [0, 1],
                        size=(num_items, num_items),
                        p=[1 - config.binary_prob, config.binary_prob],
                    )
                    * upper_mask  # Only upper triangular part
                )
                # Zero out diagonal (always for supermodularity)
                np.fill_diagonal(upper_tri, 0.0)
                data[:, :, k] = upper_tri

        elif config.method == QuadraticGenerationMethod.EXPONENTIAL:
            data = np.zeros(shape)
            for k in range(num_features):
                # Generate upper triangular matrix for supermodularity
                # For supermodularity: off-diagonal must be >= 0, upper triangular
                upper_tri = np.exp(
                    -np.abs(
                        self.rng.normal(
                            config.exp_mean,
                            config.exp_std,
                            (num_items, num_items),
                        )
                    )
                )
                # Zero out diagonal and lower triangular part
                if config.zero_diagonal:
                    np.fill_diagonal(upper_tri, 0.0)
                upper_tri *= np.triu(np.ones((num_items, num_items)), k=1)
                # Apply masking (also upper triangular)
                mask = self.rng.random((num_items, num_items)) < config.mask_threshold
                mask *= np.triu(np.ones((num_items, num_items)), k=1)  # Ensure mask is upper triangular
                upper_tri *= mask
                data[:, :, k] = upper_tri

        else:
            raise ValueError(f"Unknown quadratic generation method: {config.method}")

        return data

    def generate_quadratic_agent(
        self,
        shape: tuple[int, int, int, int],  # (num_agents, num_items, num_items, num_features)
        config: QuadraticItemConfig,
    ) -> np.ndarray:
        """Generate quadratic agent features."""
        num_agents, num_items, _, num_features = shape
        data = np.zeros(shape)
        
        for i in range(num_agents):
            # Generate quadratic features for each agent using the same logic as items
            agent_quad = self.generate_quadratic_item(
                (num_items, num_items, num_features), config
            )
            data[i] = agent_quad
        
        return data

    def generate_capacities(
        self,
        num_agents: int,
        weights: np.ndarray,
        config: CapacityConfig,
    ) -> np.ndarray:
        """Generate agent capacities for knapsack."""
        mean_capacity = int(config.mean_multiplier * weights.sum())
        lower = max(1, int(config.lower_multiplier * mean_capacity))
        upper = max(lower + 1, int(config.upper_multiplier * mean_capacity))
        return self.rng.integers(lower, upper + 1, size=num_agents)

    def generate_random_capacities(
        self,
        num_agents: int,
        low: int = 1,
        high: int = 100,
    ) -> np.ndarray:
        """Generate random agent capacities (not based on weights)."""
        return self.rng.integers(low, high, size=num_agents)

    def generate_weights(
        self,
        num_items: int,
        low: int = 1,
        high: int = 11,
    ) -> np.ndarray:
        """Generate item weights."""
        return self.rng.integers(low, high, size=num_items)

    def generate_errors(
        self,
        shape: tuple[int, ...],
        sigma: float,
    ) -> np.ndarray:
        """Generate error terms."""
        return sigma * self.rng.normal(0.0, 1.0, shape)

    def _apply_correlation_matrix(
        self,
        data: np.ndarray,
        config: CorrelationConfig,
    ) -> np.ndarray:
        """Apply correlation matrix transformation to data."""
        # Reshape to (..., num_features) for matrix multiplication
        original_shape = data.shape
        if len(original_shape) < 2:
            return data

        num_features = original_shape[-1]
        data_2d = data.reshape(-1, num_features)

        # Generate full-rank matrix
        while True:
            matrix = self.rng.integers(
                config.matrix_range[0],
                config.matrix_range[1],  # Exclusive upper bound to match numpy.random.integers behavior
                size=(num_features, num_features),
            )
            if np.any(matrix.sum(0) == 0):
                continue
            if np.linalg.matrix_rank(matrix) == num_features:
                if config.normalize:
                    matrix = matrix / matrix.sum(0)
                break

        # Apply transformation
        transformed = data_2d @ matrix
        return transformed.reshape(original_shape)


__all__ = [
    "DataGenerator",
    "ModularAgentConfig",
    "ModularItemConfig",
    "QuadraticItemConfig",
    "QuadraticGenerationMethod",
    "CorrelationConfig",
    "CapacityConfig",
]

