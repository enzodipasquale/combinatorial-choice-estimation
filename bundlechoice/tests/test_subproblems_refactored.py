import numpy as np
from mpi4py import MPI
import pytest
from bundlechoice.config import BundleChoiceConfig
from bundlechoice.comm_manager import CommManager
from bundlechoice.data_manager import DataManager
from bundlechoice.oracles_manager import OraclesManager
from bundlechoice.subproblems.subproblem_manager import SubproblemManager

def setup_test_env(num_obs=10, num_items=5, num_features=3, subproblem_name='Greedy', settings=None):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    cfg = BundleChoiceConfig()
    cfg.dimensions.num_obs = num_obs
    cfg.dimensions.num_items = num_items
    cfg.dimensions.num_features = num_features
    cfg.dimensions.num_simulations = 1
    cfg.subproblem.name = subproblem_name
    cfg.subproblem.settings = settings or {}
    comm_manager = CommManager(comm)
    data_manager = DataManager(cfg.dimensions, comm_manager)
    oracles_manager = OraclesManager(cfg.dimensions, comm_manager, data_manager)
    input_data = {
        'agent_data': {'modular': np.random.randn(num_obs, num_items, 2)},
        'item_data': {'modular': np.random.randn(num_items, 1)}
    } if rank == 0 else {'agent_data': {}, 'item_data': {}}
    data_manager.load_input_data(input_data)
    def features_oracle(bundles, local_id, data):
        return np.sum(bundles, axis=-1, keepdims=True).repeat(num_features, axis=-1).astype(float)
    oracles_manager.set_features_oracle(features_oracle)
    oracles_manager.build_local_modular_error_oracle(seed=42)
    return cfg, comm_manager, data_manager, oracles_manager

def test_greedy_subproblem():
    cfg, comm, data, oracles = setup_test_env(subproblem_name='Greedy')
    sub = SubproblemManager(comm, cfg, data, oracles)
    sub.load()
    sub.initialize_subproblems()
    theta = np.ones(cfg.dimensions.num_features)
    result = sub.solve_subproblems(theta)
    assert result is not None
    assert result.shape[0] == data.num_local_agent
    assert result.shape[1] == cfg.dimensions.num_items
    assert result.dtype == bool

def test_brute_force_subproblem():
    cfg, comm, data, oracles = setup_test_env(num_items=4, subproblem_name='BruteForce')
    sub = SubproblemManager(comm, cfg, data, oracles)
    sub.load()
    sub.initialize_subproblems()
    theta = np.ones(cfg.dimensions.num_features)
    result = sub.solve_subproblems(theta)
    assert result is not None
    assert result.shape[0] == data.num_local_agent
    assert result.shape[1] == cfg.dimensions.num_items

def test_plain_single_item_subproblem():
    cfg, comm, data, oracles = setup_test_env(subproblem_name='PlainSingleItem')
    sub = SubproblemManager(comm, cfg, data, oracles)
    sub.load()
    sub.initialize_subproblems()
    theta = np.ones(cfg.dimensions.num_features)
    result = sub.solve_subproblems(theta)
    assert result is not None
    assert result.shape[0] == data.num_local_agent
    assert result.shape[1] == cfg.dimensions.num_items
    assert result.sum(axis=1).max() <= 1  # at most 1 item per agent

def test_subproblem_manager_load_unknown():
    cfg, comm, data, oracles = setup_test_env()
    sub = SubproblemManager(comm, cfg, data, oracles)
    with pytest.raises(ValueError, match="Unknown subproblem"):
        sub.load('NonExistentSubproblem')

def test_subproblem_multiple_solves():
    cfg, comm, data, oracles = setup_test_env(subproblem_name='Greedy')
    sub = SubproblemManager(comm, cfg, data, oracles)
    sub.load()
    sub.initialize_subproblems()
    theta1 = np.ones(cfg.dimensions.num_features)
    theta2 = np.ones(cfg.dimensions.num_features) * 2
    result1 = sub.solve_subproblems(theta1)
    result2 = sub.solve_subproblems(theta2)
    assert result1 is not None
    assert result2 is not None

def test_greedy_different_theta():
    cfg, comm, data, oracles = setup_test_env(subproblem_name='Greedy')
    sub = SubproblemManager(comm, cfg, data, oracles)
    sub.load()
    sub.initialize_subproblems()
    theta_pos = np.array([1.0, 1.0, 1.0])
    theta_neg = np.array([-1.0, -1.0, -1.0])
    result_pos = sub.solve_subproblems(theta_pos)
    result_neg = sub.solve_subproblems(theta_neg)
    assert result_pos is not None
    assert result_neg is not None

def setup_quadratic_test_env(num_obs=10, num_items=5, subproblem_name='QuadSupermodularLovasz', settings=None):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_features = 2 + 1  # modular_item + quadratic_item
    cfg = BundleChoiceConfig()
    cfg.dimensions.num_obs = num_obs
    cfg.dimensions.num_items = num_items
    cfg.dimensions.num_features = num_features
    cfg.dimensions.num_simulations = 1
    cfg.subproblem.name = subproblem_name
    cfg.subproblem.settings = settings or {}
    comm_manager = CommManager(comm)
    data_manager = DataManager(cfg.dimensions, comm_manager)
    oracles_manager = OraclesManager(cfg.dimensions, comm_manager, data_manager)
    # Create quadratic features with zero diagonal and non-negative (for supermodular)
    quad_item = np.abs(np.random.randn(num_items, num_items, 1)) * 0.1
    for i in range(num_items):
        quad_item[i, i, :] = 0
    input_data = {
        'agent_data': {},
        'item_data': {'modular': np.random.randn(num_items, 2), 'quadratic': quad_item}
    } if rank == 0 else {'agent_data': {}, 'item_data': {}}
    data_manager.load_input_data(input_data)
    oracles_manager.build_quadratic_features_from_data()
    oracles_manager.build_local_modular_error_oracle(seed=42)
    return cfg, comm_manager, data_manager, oracles_manager

def test_quad_supermodular_lovasz():
    cfg, comm, data, oracles = setup_quadratic_test_env(subproblem_name='QuadSupermodularLovasz', settings={'num_iters_SGM': 100})
    sub = SubproblemManager(comm, cfg, data, oracles)
    sub.load()
    sub.initialize_subproblems()
    theta = np.array([0.5, 0.5, -0.05])
    result = sub.solve_subproblems(theta)
    assert result is not None
    assert result.shape[0] == data.num_local_agent
    assert result.shape[1] == cfg.dimensions.num_items

def test_quad_supermodular_network():
    cfg, comm, data, oracles = setup_quadratic_test_env(subproblem_name='QuadSupermodularNetwork')
    sub = SubproblemManager(comm, cfg, data, oracles)
    sub.load()
    sub.initialize_subproblems()
    theta = np.array([0.5, 0.5, -0.05])
    result = sub.solve_subproblems(theta)
    assert result is not None
    assert result.shape[0] == data.num_local_agent
    assert result.shape[1] == cfg.dimensions.num_items

def setup_knapsack_test_env(num_obs=10, num_items=5, capacity=3):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_features = 2 + 1  # modular_item + quadratic_item
    cfg = BundleChoiceConfig()
    cfg.dimensions.num_obs = num_obs
    cfg.dimensions.num_items = num_items
    cfg.dimensions.num_features = num_features
    cfg.dimensions.num_simulations = 1
    cfg.subproblem.name = 'QuadKnapsack'
    comm_manager = CommManager(comm)
    data_manager = DataManager(cfg.dimensions, comm_manager)
    oracles_manager = OraclesManager(cfg.dimensions, comm_manager, data_manager)
    quad_item = np.random.randn(num_items, num_items, 1) * 0.1
    for i in range(num_items):
        quad_item[i, i, :] = 0
    input_data = {
        'agent_data': {'capacity': np.ones(num_obs) * capacity},
        'item_data': {'modular': np.random.randn(num_items, 2), 'quadratic': quad_item, 'weights': np.ones(num_items)}
    } if rank == 0 else {'agent_data': {}, 'item_data': {}}
    data_manager.load_input_data(input_data)
    oracles_manager.build_quadratic_features_from_data()
    oracles_manager.build_local_modular_error_oracle(seed=42)
    return cfg, comm_manager, data_manager, oracles_manager

def test_quad_knapsack():
    cfg, comm, data, oracles = setup_knapsack_test_env(capacity=3)
    sub = SubproblemManager(comm, cfg, data, oracles)
    sub.load()
    sub.initialize_subproblems()
    theta = np.array([0.5, 0.5, -0.05])
    result = sub.solve_subproblems(theta)
    assert result is not None
    assert result.shape[0] == data.num_local_agent
    assert result.shape[1] == cfg.dimensions.num_items
    assert all(result.sum(axis=1) <= 3)  # capacity constraint

def test_linear_knapsack():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_obs, num_items, capacity = 10, 5, 3
    cfg = BundleChoiceConfig()
    cfg.dimensions.num_obs = num_obs
    cfg.dimensions.num_items = num_items
    cfg.dimensions.num_features = 3
    cfg.subproblem.name = 'LinearKnapsack'
    comm_mgr = CommManager(comm)
    data_mgr = DataManager(cfg.dimensions, comm_mgr)
    oracles_mgr = OraclesManager(cfg.dimensions, comm_mgr, data_mgr)
    input_data = {
        'agent_data': {'capacity': np.ones(num_obs) * capacity},
        'item_data': {'modular': np.random.randn(num_items, 3), 'weights': np.ones(num_items)}
    } if rank == 0 else {'agent_data': {}, 'item_data': {}}
    data_mgr.load_input_data(input_data)
    def features_oracle(bundles, local_id, data):
        return np.sum(bundles, axis=-1, keepdims=True).repeat(3, axis=-1).astype(float)
    oracles_mgr.set_features_oracle(features_oracle)
    oracles_mgr.build_local_modular_error_oracle(seed=42)
    sub = SubproblemManager(comm_mgr, cfg, data_mgr, oracles_mgr)
    sub.load()
    sub.initialize_subproblems()
    theta = np.array([1.0, 0.5, -0.5])
    result = sub.solve_subproblems(theta)
    assert result is not None
    assert result.shape[0] == data_mgr.num_local_agent
    assert result.shape[1] == num_items
    assert all(result.sum(axis=1) <= capacity)
