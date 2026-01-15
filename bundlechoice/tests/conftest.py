import pytest
import numpy as np
from typing import Dict, Any, Tuple
from mpi4py import MPI
from bundlechoice.core import BundleChoice
from bundlechoice.config import BundleChoiceConfig, DimensionsConfig
from bundlechoice.comm_manager import CommManager

def pytest_configure(config):
    config.addinivalue_line('markers', 'mpi: test requires MPI (run with mpirun)')
    config.addinivalue_line('markers', 'slow: test takes a long time to run')
    config.addinivalue_line('markers', 'integration: integration test')
    config.addinivalue_line('markers', 'unit: unit test')

@pytest.fixture(scope='function')
def comm():
    return MPI.COMM_WORLD

@pytest.fixture(scope='function')
def comm_manager(comm):
    return CommManager(comm)

@pytest.fixture(scope='function')
def rank(comm):
    return comm.Get_rank()

@pytest.fixture(scope='function')
def size(comm):
    return comm.Get_size()

@pytest.fixture(scope='function')
def _is_root(rank):
    return rank == 0

@pytest.fixture(scope='function')
def rng():
    return np.random.RandomState(42)

def generate_simple_test_data(num_obs: int=20, num_items: int=10, num_features: int=3, num_simulations: int=1, seed: int=42):
    rng = np.random.RandomState(seed)
    agent_data = {'modular': rng.normal(0, 1, (num_obs, num_items, num_features))}
    item_data = {}
    errors = rng.normal(0, 0.5, size=(num_simulations, num_obs, num_items))
    input_data = {'agent_data': agent_data, 'item_data': item_data, 'errors': errors}
    config = {'dimensions': {'num_obs': num_obs, 'num_items': num_items, 'num_features': num_features, 'num_simulations': num_simulations}, 'subproblem': {'name': 'Greedy', 'settings': {}}}
    return (input_data, config)

def generate_knapsack_test_data(num_obs: int=20, num_items: int=10, agent_modular_dim: int=2, item_modular_dim: int=1, num_simulations: int=1, seed: int=42):
    rng = np.random.RandomState(seed)
    item_data = {'weights': rng.randint(1, 10, size=num_items), 'modular': rng.normal(0, 1, (num_items, item_modular_dim))}
    agent_data = {'modular': rng.normal(0, 1, (num_obs, num_items, agent_modular_dim)), 'capacity': rng.randint(10, 50, size=num_obs)}
    errors = rng.normal(0, 1, size=(num_simulations, num_obs, num_items))
    input_data = {'item_data': item_data, 'agent_data': agent_data, 'errors': errors}
    num_features = agent_modular_dim + item_modular_dim
    config = {'dimensions': {'num_obs': num_obs, 'num_items': num_items, 'num_features': num_features, 'num_simulations': num_simulations}, 'subproblem': {'name': 'LinearKnapsack', 'settings': {'TimeLimit': 10, 'MIPGap_tol': 0.01}}}
    return (input_data, config)

@pytest.fixture
def simple_bundlechoice(comm, rng):
    input_data, config = generate_simple_test_data(seed=rng.randint(0, 10000))
    bc = BundleChoice()
    bc.load_config(config)
    bc.data.load_input_data(input_data)
    bc.oracles.build_quadratic_features_from_data()
    return bc

def assert_bundles_valid(bundles: np.ndarray, num_obs: int, num_items: int, is_root: bool):
    if is_root:
        assert bundles is not None, 'Bundles should not be None on root'
        assert isinstance(bundles, np.ndarray), 'Bundles should be numpy array'
        assert bundles.dtype == np.bool_, f'Bundles should be bool, got {bundles.dtype}'
        assert bundles.shape == (num_obs, num_items), f'Expected shape ({num_obs}, {num_items}), got {bundles.shape}'
    else:
        assert bundles is None, 'Bundles should be None on non-root ranks'

def assert_theta_valid(theta: np.ndarray, num_features: int, is_root: bool):
    if is_root:
        assert theta is not None, 'Theta should not be None on root'
        assert isinstance(theta, np.ndarray), 'Theta should be numpy array'
        assert theta.shape == (num_features,), f'Expected shape ({num_features},), got {theta.shape}'
        assert not np.any(np.isnan(theta)), 'Theta should not contain NaN'
        assert not np.any(np.isinf(theta)), 'Theta should not contain Inf'
    else:
        assert theta is None, 'Theta should be None on non-root ranks'