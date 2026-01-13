"""
Pytest configuration and shared fixtures for BundleChoice tests.

This module provides:
- Shared fixtures for common test scenarios
- Pytest markers for test categorization
- Helper functions for test data generation
"""

import pytest
import numpy as np
from typing import Dict, Any, Tuple
from mpi4py import MPI

from bundlechoice.core import BundleChoice
from bundlechoice.config import BundleChoiceConfig, DimensionsConfig
from bundlechoice.comm_manager import CommManager


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "mpi: test requires MPI (run with mpirun)")
    config.addinivalue_line("markers", "slow: test takes a long time to run")
    config.addinivalue_line("markers", "integration: integration test")
    config.addinivalue_line("markers", "unit: unit test")


@pytest.fixture(scope="function")
def comm():
    """MPI communicator fixture."""
    return MPI.COMM_WORLD


@pytest.fixture(scope="function")
def comm_manager(comm):
    """CommManager fixture."""
    return CommManager(comm)


@pytest.fixture(scope="function")
def rank(comm):
    """Current MPI rank."""
    return comm.Get_rank()


@pytest.fixture(scope="function")
def size(comm):
    """Total number of MPI processes."""
    return comm.Get_size()


@pytest.fixture(scope="function")
def is_root(rank):
    """Check if current rank is root."""
    return rank == 0


@pytest.fixture(scope="function")
def rng():
    """Seeded random number generator for reproducible tests."""
    return np.random.RandomState(42)


def generate_simple_test_data(
    num_agents: int = 20,
    num_items: int = 10,
    num_features: int = 3,
    num_simulations: int = 1,
    seed: int = 42,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Generate simple test data for bundle choice problems.
    
    Returns:
        Tuple of (input_data dict, config dict)
    """
    rng = np.random.RandomState(seed)
    
    agent_data = {
        "modular": rng.normal(0, 1, (num_agents, num_items, num_features))
    }
    item_data = {}
    errors = rng.normal(0, 0.5, size=(num_simulations, num_agents, num_items))
    
    input_data = {
        "agent_data": agent_data,
        "item_data": item_data,
        "errors": errors,
    }
    
    config = {
        "dimensions": {
            "num_agents": num_agents,
            "num_items": num_items,
            "num_features": num_features,
            "num_simulations": num_simulations,
        },
        "subproblem": {
            "name": "Greedy",
            "settings": {}
        }
    }
    
    return input_data, config


def generate_knapsack_test_data(
    num_agents: int = 20,
    num_items: int = 10,
    agent_modular_dim: int = 2,
    item_modular_dim: int = 1,
    num_simulations: int = 1,
    seed: int = 42,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate test data for knapsack problems."""
    rng = np.random.RandomState(seed)
    
    item_data = {
        "weights": rng.randint(1, 10, size=num_items),
        "modular": rng.normal(0, 1, (num_items, item_modular_dim))
    }
    agent_data = {
        "modular": rng.normal(0, 1, (num_agents, num_items, agent_modular_dim)),
        "capacity": rng.randint(10, 50, size=num_agents),
    }
    errors = rng.normal(0, 1, size=(num_simulations, num_agents, num_items))
    
    input_data = {
        "item_data": item_data,
        "agent_data": agent_data,
        "errors": errors,
    }
    
    num_features = agent_modular_dim + item_modular_dim
    config = {
        "dimensions": {
            "num_agents": num_agents,
            "num_items": num_items,
            "num_features": num_features,
            "num_simulations": num_simulations,
        },
        "subproblem": {
            "name": "LinearKnapsack",
            "settings": {"TimeLimit": 10, "MIPGap_tol": 0.01}
        }
    }
    
    return input_data, config


@pytest.fixture
def simple_bundlechoice(comm, rng):
    """Fixture providing a configured BundleChoice instance with simple test data."""
    input_data, config = generate_simple_test_data(seed=rng.randint(0, 10000))
    
    bc = BundleChoice()
    bc.load_config(config)
    bc.data.load_and_scatter(input_data)
    bc.oracles.build_from_data()
    
    return bc


def assert_bundles_valid(bundles: np.ndarray, num_agents: int, num_items: int, is_root: bool):
    """Assert that bundles array has correct shape and dtype."""
    if is_root:
        assert bundles is not None, "Bundles should not be None on root"
        assert isinstance(bundles, np.ndarray), "Bundles should be numpy array"
        assert bundles.dtype == np.bool_, f"Bundles should be bool, got {bundles.dtype}"
        assert bundles.shape == (num_agents, num_items), \
            f"Expected shape ({num_agents}, {num_items}), got {bundles.shape}"
    else:
        assert bundles is None, "Bundles should be None on non-root ranks"


def assert_theta_valid(theta: np.ndarray, num_features: int, is_root: bool):
    """Assert that theta array is valid."""
    if is_root:
        assert theta is not None, "Theta should not be None on root"
        assert isinstance(theta, np.ndarray), "Theta should be numpy array"
        assert theta.shape == (num_features,), \
            f"Expected shape ({num_features},), got {theta.shape}"
        assert not np.any(np.isnan(theta)), "Theta should not contain NaN"
        assert not np.any(np.isinf(theta)), "Theta should not contain Inf"
    else:
        assert theta is None, "Theta should be None on non-root ranks"

