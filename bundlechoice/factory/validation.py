"""Validation script for factory data generation.

This script verifies that factory-generated data matches manual generation
for all scenarios when using the same seed. Run this when adding new factories
or modifying existing ones.
"""

from __future__ import annotations

import sys
from typing import Dict, Tuple

import numpy as np

from .data_generator import (
    CapacityConfig,
    CorrelationConfig,
    DataGenerator,
    ModularAgentConfig,
    ModularItemConfig,
    QuadraticGenerationMethod,
    QuadraticItemConfig,
)


def validate_greedy(seed: int) -> Tuple[bool, str]:
    """Validate Greedy factory data generation."""
    rng_manual = np.random.default_rng(seed)
    num_agents, num_items, num_features = 300, 100, 4
    sigma = 1.0

    # Manual generation
    modular_agent_manual = np.abs(rng_manual.normal(0, 1, (num_agents, num_items, num_features - 1)))
    errors_manual = sigma * rng_manual.normal(0, 1, size=(num_agents, num_items))

    # Factory generation
    generator = DataGenerator(seed=seed)
    agent_config = ModularAgentConfig(apply_abs=True)
    modular_agent_factory = generator.generate_modular_agent(
        (num_agents, num_items, num_features - 1), agent_config
    )
    errors_factory = generator.generate_errors((num_agents, num_items), sigma)

    agent_match = np.allclose(modular_agent_manual, modular_agent_factory, rtol=1e-10)
    errors_match = np.allclose(errors_manual, errors_factory, rtol=1e-10)

    if agent_match and errors_match:
        return True, "✅ Greedy matches"
    return False, f"❌ Greedy mismatch: agent={agent_match}, errors={errors_match}"


def validate_plain_single_item(seed: int) -> Tuple[bool, str]:
    """Validate PlainSingleItem factory data generation."""
    rng_manual = np.random.default_rng(seed)
    num_agents, num_items, num_features = 300, 100, 4
    sigma = 1.0

    # Manual generation
    modular_agent_manual = rng_manual.normal(0, 1, (num_agents, num_items, num_features))
    while True:
        full_rank = rng_manual.integers(0, 4, size=(num_features, num_features))
        if np.any(full_rank.sum(0) == 0):
            continue
        if np.linalg.matrix_rank(full_rank) == num_features:
            full_rank = full_rank / full_rank.sum(0)
            break
    modular_agent_manual = modular_agent_manual @ full_rank
    errors_manual = sigma * rng_manual.normal(0, 1, size=(num_agents, num_items))

    # Factory generation
    generator = DataGenerator(seed=seed)
    agent_config = ModularAgentConfig(
        correlation=CorrelationConfig(enabled=True, matrix_range=(0, 4), normalize=True)
    )
    modular_agent_factory = generator.generate_modular_agent(
        (num_agents, num_items, num_features), agent_config
    )
    errors_factory = generator.generate_errors((num_agents, num_items), sigma)

    agent_match = np.allclose(modular_agent_manual, modular_agent_factory, rtol=1e-10)
    errors_match = np.allclose(errors_manual, errors_factory, rtol=1e-10)

    if agent_match and errors_match:
        return True, "✅ PlainSingleItem matches"
    return False, f"❌ PlainSingleItem mismatch: agent={agent_match}, errors={errors_match}"


def validate_linear_knapsack(seed: int) -> Tuple[bool, str]:
    """Validate LinearKnapsack factory data generation."""
    rng_manual = np.random.default_rng(seed)
    num_agents, num_items = 300, 100
    modular_agent_features, modular_item_features = 2, 1
    sigma = 1.0

    # Manual generation
    modular_agent_manual = np.abs(rng_manual.normal(0, 1, (num_agents, num_items, modular_agent_features)))
    modular_item_manual = np.abs(rng_manual.normal(0, 1, (num_items, modular_item_features)))
    weights_manual = rng_manual.integers(1, 11, num_items)
    mean_capacity = int(0.5 * weights_manual.sum())
    lo = int(0.85 * mean_capacity)
    hi = int(1.15 * mean_capacity)
    capacity_manual = rng_manual.integers(lo, hi + 1, size=num_agents)
    errors_manual = sigma * rng_manual.normal(0, 1, size=(num_agents, num_items))

    # Factory generation
    generator = DataGenerator(seed=seed)
    agent_config = ModularAgentConfig(apply_abs=True)
    item_config = ModularItemConfig(apply_abs=True)
    capacity_config = CapacityConfig(mean_multiplier=0.5, lower_multiplier=0.85, upper_multiplier=1.15)

    modular_agent_factory = generator.generate_modular_agent(
        (num_agents, num_items, modular_agent_features), agent_config
    )
    modular_item_factory = generator.generate_modular_item(
        (num_items, modular_item_features), item_config
    )
    weights_factory = generator.generate_weights(num_items)  # Uses default WeightConfig
    capacity_factory = generator.generate_capacities(num_agents, weights_factory, capacity_config)
    errors_factory = generator.generate_errors((num_agents, num_items), sigma)

    checks = {
        "agent": np.allclose(modular_agent_manual, modular_agent_factory, rtol=1e-10),
        "item": np.allclose(modular_item_manual, modular_item_factory, rtol=1e-10),
        "weights": np.array_equal(weights_manual, weights_factory),
        "capacity": np.array_equal(capacity_manual, capacity_factory),
        "errors": np.allclose(errors_manual, errors_factory, rtol=1e-10),
    }

    if all(checks.values()):
        return True, "✅ LinearKnapsack matches"
    failed = [k for k, v in checks.items() if not v]
    return False, f"❌ LinearKnapsack mismatch: {failed}"


def validate_quadratic_supermodular(seed: int) -> Tuple[bool, str]:
    """Validate QuadraticSupermodular factory data generation."""
    rng_manual = np.random.default_rng(seed)
    num_agents, num_items = 1000, 100
    modular_agent_features, quadratic_item_features = 5, 1
    sigma = 5.0

    # Manual generation
    modular_agent_manual = -5 * np.abs(rng_manual.normal(0, 1, (num_agents, num_items, modular_agent_features)))
    quadratic_item_manual = rng_manual.choice(
        [0, 1], size=(num_items, num_items, quadratic_item_features), p=[0.8, 0.2]
    )
    quadratic_item_manual *= (1 - np.eye(num_items, dtype=int))[:, :, None]
    errors_manual = sigma * rng_manual.normal(0, 1, size=(num_agents, num_items))

    # Factory generation
    generator = DataGenerator(seed=seed)
    agent_config = ModularAgentConfig(multiplier=-5.0, mean=0.0, std=1.0, apply_abs=True)
    quadratic_config = QuadraticItemConfig(
        method=QuadraticGenerationMethod.BINARY_CHOICE,
        binary_prob=0.2,
        binary_value=1.0,
    )

    modular_agent_factory = generator.generate_modular_agent(
        (num_agents, num_items, modular_agent_features), agent_config
    )
    quadratic_item_factory = generator.generate_quadratic_item(
        (num_items, num_items, quadratic_item_features), quadratic_config
    )
    errors_factory = generator.generate_errors((num_agents, num_items), sigma)

    checks = {
        "agent": np.allclose(modular_agent_manual, modular_agent_factory, rtol=1e-10),
        "quadratic": np.allclose(quadratic_item_manual, quadratic_item_factory, rtol=1e-10),
        "errors": np.allclose(errors_manual, errors_factory, rtol=1e-10),
    }

    if all(checks.values()):
        return True, "✅ QuadraticSupermodular matches"
    failed = [k for k, v in checks.items() if not v]
    return False, f"❌ QuadraticSupermodular mismatch: {failed}"


VALIDATORS: Dict[str, callable] = {
    "greedy": validate_greedy,
    "plain_single_item": validate_plain_single_item,
    "linear_knapsack": validate_linear_knapsack,
    "quadratic_supermodular": validate_quadratic_supermodular,
}


def validate_all(seed: int = 42) -> bool:
    """Validate all factory data generation."""
    print(f"Validating factory data generation with seed={seed}\n")
    print("=" * 60)

    all_passed = True
    for name, validator in VALIDATORS.items():
        print(f"\n{name}:")
        passed, message = validator(seed)
        print(f"  {message}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All validations passed!")
    else:
        print("❌ Some validations failed!")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    success = validate_all(seed)
    sys.exit(0 if success else 1)

