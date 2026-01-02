"""Base data generator for exogenous data generation (before BundleChoice)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional, Union

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
class WeightConfig:
    """Configuration for item weight generation (knapsack).
    
    Supports different distributions for heterogeneity, all producing integers:
    - 'uniform': Uniform random integers [low, high] (default, least heterogeneous)
    - 'lognormal': Log-normal distribution, truncated and rounded to integers (more heterogeneous)
    - 'exponential': Exponential distribution, truncated and rounded to integers
    """

    distribution: str = "uniform"  # 'uniform', 'lognormal', or 'exponential'
    low: int = 1  # Lower bound for weight range (min value)
    high: int = 10  # Upper bound for weight range (max value)
    # For lognormal:
    log_mean: float = 0.0  # Mean of underlying normal distribution
    log_std: float = 1.0  # Std of underlying normal distribution
    # For exponential:
    exp_scale: float = 2.0  # Scale parameter (mean = scale)


@dataclass
class CapacityConfig:
    """Configuration for agent capacity generation (knapsack).
    
    Supports two methods:
    1. Fixed fraction: capacity = int(fraction * sum(weights)) - deterministic, same for all agents
    2. Variance-based (default): capacity = random in [lower_mult * mean, upper_mult * mean]
       where mean = mean_multiplier * sum(weights) - random per agent
    
    Default uses variance-based method with mean_multiplier=0.45 for randomness.
    """

    # Method 1: Fixed fraction (deterministic)
    fraction: Optional[float] = None  # If set, use fixed fraction of total weight
    
    # Method 2: Variance-based (default, random)
    mean_multiplier: Optional[float] = None  # Multiplier for mean capacity (default 0.45 if fraction not set)
    lower_multiplier: float = 0.85  # Lower bound multiplier
    upper_multiplier: float = 1.15  # Upper bound multiplier
    
    def __post_init__(self):
        """Set default mean_multiplier if fraction not specified."""
        if self.fraction is None and self.mean_multiplier is None:
            # Default to variance-based with 0.45 for randomness
            object.__setattr__(self, 'mean_multiplier', 0.45)
        elif self.fraction is not None and self.mean_multiplier is not None:
            raise ValueError("Cannot specify both 'fraction' and 'mean_multiplier'. Choose one method.")


@dataclass
class EndogeneityConfig:
    """Configuration for endogenous product characteristics following BLP structure.
    
    Implements the structure:
    - ξ ~ N(0, Σ_ξ): unobserved product characteristics
    - Z ~ N(0, Σ_z): instruments
    - v_j = Λ ξ_j: correlation component
    - x_j = z_j @ Π + v_j: endogenous features
    
    This creates Cov(x_j, ξ_j) = Λ Σ_ξ (endogeneity) and Cov(z_j, ξ_j) = 0 (valid instruments).
    """

    # Which features are endogenous (column indices in modular_item)
    endogenous_feature_indices: list[int]  # K features
    
    # Instrument specification
    num_instruments: int  # L instruments
    instrument_cov: Optional[np.ndarray] = None  # Σ_z (L×L), defaults to I
    instrument_config: Optional[ModularItemConfig] = None  # Alternative generator config
    
    # First-stage coefficients Π (L×K)
    pi_matrix: Optional[np.ndarray] = None  # Explicit matrix (L×K)
    pi_config: Optional[dict] = None  # {"mean": 0, "std": 1} for random generation
    
    # Correlation structure Λ (K×1 or K×K)
    lambda_matrix: Optional[np.ndarray] = None  # Explicit matrix
    lambda_config: Optional[dict] = None  # {"mean": 0, "std": 1} for random generation
    
    # Error covariance Σ_ξ
    xi_cov: Optional[Union[float, np.ndarray]] = None  # Scalar or matrix, defaults to 1.0
    
    # Rank condition
    ensure_full_rank: bool = True
    min_rank_tolerance: float = 1e-10


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
                mask = mask & np.triu(np.ones((num_items, num_items), dtype=bool), k=1)  # Ensure mask is upper triangular
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
        """Generate agent capacities for knapsack.
        
        Supports two methods:
        1. Fixed fraction (most standard): capacity = int(fraction * sum(weights))
        2. Variance-based: capacity = random in [lower_mult * mean, upper_mult * mean]
        
        Args:
            num_agents: Number of agents
            weights: Array of item weights
            config: CapacityConfig specifying generation method
        
        Returns:
            Array of integer capacities.
        """
        if config.fraction is not None:
            # Method 1: Fixed fraction (most standard)
            capacity = int(config.fraction * weights.sum())
            return np.full(num_agents, capacity, dtype=np.int64)
        else:
            # Method 2: Variance-based (legacy)
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
        config: Optional[WeightConfig] = None,
    ) -> np.ndarray:
        """Generate item weights as random integers.
        
        Supports different distributions for heterogeneity, all producing integers:
        - 'uniform': Uniform random integers [low, high]
        - 'lognormal': Log-normal distribution, truncated to [low, high] and rounded
        - 'exponential': Exponential distribution, truncated to [low, high] and rounded
        
        Args:
            num_items: Number of items
            config: WeightConfig specifying distribution and parameters. If None, uses default (uniform 1-10).
        
        Returns:
            Array of integer weights in range [config.low, config.high] (inclusive).
        """
        if config is None:
            config = WeightConfig()
        
        if config.distribution == "uniform":
            weights = self.rng.integers(config.low, config.high + 1, size=num_items)
        elif config.distribution == "lognormal":
            # Generate log-normal, scale to [low, high], round to integers
            log_weights = self.rng.lognormal(config.log_mean, config.log_std, size=num_items)
            # Scale to [low, high] range
            log_min, log_max = log_weights.min(), log_weights.max()
            if log_max > log_min:
                weights = config.low + (log_weights - log_min) / (log_max - log_min) * (config.high - config.low)
            else:
                weights = np.full(num_items, config.low)
            weights = np.clip(np.round(weights).astype(int), config.low, config.high)
        elif config.distribution == "exponential":
            # Generate exponential, scale to [low, high], round to integers
            exp_weights = self.rng.exponential(config.exp_scale, size=num_items)
            # Scale to [low, high] range
            exp_min, exp_max = exp_weights.min(), exp_weights.max()
            if exp_max > exp_min:
                weights = config.low + (exp_weights - exp_min) / (exp_max - exp_min) * (config.high - config.low)
            else:
                weights = np.full(num_items, config.low)
            weights = np.clip(np.round(weights).astype(int), config.low, config.high)
        else:
            raise ValueError(f"Unknown weight distribution: {config.distribution}")
        
        return weights

    def generate_errors(
        self,
        shape: tuple[int, ...],
        sigma: float,
    ) -> np.ndarray:
        """Generate error terms."""
        return sigma * self.rng.normal(0.0, 1.0, shape)

    def generate_instruments(
        self,
        num_items: int,
        num_instruments: int,
        instrument_cov: Optional[np.ndarray] = None,
        instrument_config: Optional[ModularItemConfig] = None,
    ) -> np.ndarray:
        """Generate instrument matrix Z ~ N(0, Σ_z) of shape (num_items, num_instruments).
        
        IMPORTANT: Instruments are i.i.d. across items (j). Each row z_j is independent.
        The instrument_cov (L×L) specifies covariance across instruments within an item,
        not across items.
        
        Args:
            num_items: Number of items
            num_instruments: Number of instruments (L)
            instrument_cov: Covariance matrix Σ_z (L×L) for instruments within an item, defaults to I
            instrument_config: Alternative config for using ModularItemConfig generator
        
        Returns:
            Instrument matrix Z (num_items, num_instruments) where rows are i.i.d.
        """
        if instrument_config is not None:
            # Use existing modular item generator
            return self.generate_modular_item(
                (num_items, num_instruments), instrument_config
            )
        
        if instrument_cov is None:
            # Default: independent standard normal - i.i.d. across items ✓
            return self.rng.normal(0, 1, size=(num_items, num_instruments))
        else:
            # Multivariate normal with specified covariance
            # size=num_items generates num_items independent draws (i.i.d. across items) ✓
            return self.rng.multivariate_normal(
                np.zeros(num_instruments), instrument_cov, size=num_items
            )

    def generate_endogenous_modular_item(
        self,
        base_modular_item: np.ndarray,  # (num_items, num_features)
        config: EndogeneityConfig,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate endogenous features following BLP structure.
        
        Implements:
        - ξ ~ N(0, Σ_ξ): unobserved product characteristics (i.i.d. across items j)
        - Z ~ N(0, Σ_z): instruments (i.i.d. across items j)
        - v_j = Λ ξ_j: correlation component (i.i.d. across items j since ξ is i.i.d.)
        - x_j = z_j @ Π + v_j: endogenous features (i.i.d. across items j)
        
        IMPORTANT: All components (ξ, Z, v, x) are i.i.d. across items (j) to ensure
        valid IV regression. This requires xi_cov to be scalar or diagonal.
        
        Args:
            base_modular_item: Base modular item features (num_items, num_features)
            config: EndogeneityConfig specifying the structure
        
        Returns:
            Tuple of:
            - modified_modular_item: with endogenous columns replaced (i.i.d. across items)
            - instruments: Z (num_items, num_instruments) - i.i.d. across items
            - xi: ξ (num_items,) - unobserved product characteristics (i.i.d. across items)
            - v: v (num_items, num_endogenous) - correlation component (i.i.d. across items)
        """
        num_items = base_modular_item.shape[0]
        K = len(config.endogenous_feature_indices)
        L = config.num_instruments
        
        # Generate ξ ~ N(0, Σ_ξ)
        # IMPORTANT: ξ must be i.i.d. across items (j) for IV regression to be valid
        # Therefore, we only support scalar variance (diagonal covariance)
        if config.xi_cov is None:
            xi = self.rng.normal(0, 1, size=(num_items,))
        else:
            if isinstance(config.xi_cov, (int, float)):
                # Scalar variance - i.i.d. across items ✓
                xi = self.rng.normal(0, np.sqrt(float(config.xi_cov)), size=(num_items,))
            elif isinstance(config.xi_cov, np.ndarray):
                if config.xi_cov.ndim == 0 or config.xi_cov.shape == (1, 1):
                    # Scalar variance (1D array) - i.i.d. across items ✓
                    xi = self.rng.normal(0, np.sqrt(float(config.xi_cov)), size=(num_items,))
                elif config.xi_cov.ndim == 2 and config.xi_cov.shape[0] == config.xi_cov.shape[1]:
                    # Full covariance matrix - check if diagonal
                    if np.allclose(config.xi_cov, np.diag(np.diag(config.xi_cov))):
                        # Diagonal covariance - extract diagonal and use as scalar variances
                        diag_var = np.diag(config.xi_cov)
                        if np.allclose(diag_var, diag_var[0]):
                            # All diagonal elements are the same - i.i.d. ✓
                            xi = self.rng.normal(0, np.sqrt(diag_var[0]), size=(num_items,))
                        else:
                            # Different variances per item - still i.i.d. but with different variances
                            xi = np.array([self.rng.normal(0, np.sqrt(var)) for var in diag_var])
                    else:
                        # Non-diagonal covariance - NOT i.i.d. across items ✗
                        raise ValueError(
                            "xi_cov must be a scalar or diagonal matrix for i.i.d. ξ across items. "
                            "Full covariance matrices create correlation across items, which violates "
                            "the i.i.d. assumption required for IV regression. "
                            f"Got non-diagonal matrix of shape {config.xi_cov.shape}."
                        )
                else:
                    raise ValueError(
                        f"xi_cov must be a scalar or square matrix, got shape {config.xi_cov.shape}"
                    )
            else:
                raise ValueError(f"xi_cov must be a scalar, numpy array, or None, got {type(config.xi_cov)}")
        
        # Generate Z ~ N(0, Σ_z)
        instruments = self.generate_instruments(
            num_items, L, config.instrument_cov, config.instrument_config
        )
        
        # Generate or use provided Π (L×K)
        if config.pi_matrix is not None:
            Pi = config.pi_matrix
            if Pi.shape != (L, K):
                raise ValueError(f"pi_matrix must have shape (L={L}, K={K}), got {Pi.shape}")
        elif config.pi_config:
            Pi = self.rng.normal(
                config.pi_config.get("mean", 0),
                config.pi_config.get("std", 1),
                size=(L, K)
            )
        else:
            # Default: identity-like for exactly identified case
            if L >= K:
                Pi = np.eye(L, K)
            else:
                Pi = np.eye(L, K).T
        
        # Generate Λ (K×1 or K×K)
        if config.lambda_matrix is not None:
            Lambda = config.lambda_matrix
        elif config.lambda_config:
            Lambda = self.rng.normal(
                config.lambda_config.get("mean", 0),
                config.lambda_config.get("std", 1),
                size=(K, 1)  # Default K×1
            )
        else:
            # Default: ones vector (simple correlation)
            Lambda = np.ones((K, 1))
        
        # Check rank condition: rank(Z @ Π) should be K
        if config.ensure_full_rank:
            ZPi = instruments @ Pi
            rank_ZPi = np.linalg.matrix_rank(ZPi, tol=config.min_rank_tolerance)
            if rank_ZPi < K:
                raise ValueError(
                    f"Rank condition violated: rank(Z@Π)={rank_ZPi} < K={K}. "
                    f"Need rank(Z@Π) >= {K} for identification."
                )
        
        # Generate v_j = Λ ξ_j
        if Lambda.shape == (K, 1):
            # v_j is K×1 for each j: v_j = Lambda * xi_j (scalar multiplication)
            # Lambda is (K, 1), xi is (num_items,)
            # We want v[j, :] = Lambda[:, 0] * xi[j] for each j
            # So v = (Lambda * xi[None, :]).T -> (K, num_items) -> (num_items, K)
            v = (Lambda * xi[None, :]).T  # (K, 1) * (1, num_items) -> (K, num_items) -> (num_items, K)
        elif Lambda.shape == (K, K):
            # General case: Λ is K×K, v_j = Λ @ [ξ_j, ...] (if ξ is vector)
            # For scalar ξ, we treat it as v_j = Λ @ [ξ_j, 0, ..., 0]
            xi_expanded = np.zeros((num_items, K))
            xi_expanded[:, 0] = xi
            v = (Lambda @ xi_expanded.T).T  # (num_items, K)
        else:
            raise ValueError(
                f"Lambda must have shape (K, 1) or (K, K), got {Lambda.shape}"
            )
        
        # Generate x_j = z_j @ Π + v_j
        x_endog = instruments @ Pi + v  # (num_items, K)
        
        # Replace endogenous columns in base_modular_item
        modified_modular_item = base_modular_item.copy()
        for idx, endog_idx in enumerate(config.endogenous_feature_indices):
            if endog_idx >= base_modular_item.shape[1]:
                raise ValueError(
                    f"endogenous_feature_indices contains {endog_idx} but "
                    f"modular_item only has {base_modular_item.shape[1]} columns"
                )
            modified_modular_item[:, endog_idx] = x_endog[:, idx]
        
        return modified_modular_item, instruments, xi, v

    def generate_errors_with_endogeneity(
        self,
        shape: tuple[int, ...],
        sigma: float,
        xi: np.ndarray,  # ξ (num_items,)
    ) -> np.ndarray:
        """Generate errors with endogenous component ξ.
        
        IMPORTANT: Errors are i.i.d. across items (j) conditional on ξ.
        The base errors are i.i.d., and ξ is i.i.d. across items (enforced in generate_endogenous_modular_item).
        Therefore, the final errors are i.i.d. across items.
        
        Args:
            shape: Shape for errors, e.g., (num_agents, num_items) or (num_simulations, num_agents, num_items)
            sigma: Standard deviation for base errors
            xi: Endogenous error component ξ (num_items,) - must be i.i.d. across items
        
        Returns:
            Errors with shape matching input, including ξ component. Errors are i.i.d. across items.
        """
        base_errors = self.generate_errors(shape, sigma)  # i.i.d. across items and agents
        # Add ξ component: errors[j,i] includes ξ[i]
        # Since xi is i.i.d. across items, final errors remain i.i.d. across items
        if len(shape) == 2:
            # (num_agents, num_items)
            return base_errors + xi[None, :]
        elif len(shape) == 3:
            # (num_simulations, num_agents, num_items)
            return base_errors + xi[None, None, :]
        else:
            raise ValueError(f"Unsupported shape for errors: {shape}")

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
    "WeightConfig",
    "CapacityConfig",
    "EndogeneityConfig",
]

