import numpy as np
import pytest
from bundlechoice.core import BundleChoice

def test_generate_data_linear():
    num_obs, num_items, num_simulations = (32, 6, 1)
    agent_modular_dim = 3
    item_modular_dim = 2
    np.random.seed(123)
    item_data = {'weights': np.random.randint(1, 10, size=num_items), 'modular': np.random.normal(0, 1, (num_items, item_modular_dim))}
    agent_data = {'modular': np.random.normal(0, 1, (num_obs, num_items, agent_modular_dim)), 'capacity': np.random.randint(1, 100, size=num_obs)}
    errors = np.random.normal(0, 1, size=(num_simulations, num_obs, num_items))
    input_data = {'item_data': item_data, 'agent_data': agent_data, 'errors': errors}
    num_features = 0
    if 'modular' in agent_data:
        num_features += agent_data['modular'].shape[-1]
    if 'modular' in item_data:
        num_features += item_data['modular'].shape[-1]
    cfg = {'dimensions': {'num_obs': num_obs, 'num_items': num_items, 'num_features': num_features, 'num_simulations': num_simulations}, 'subproblem': {'name': 'LinearKnapsack', 'settings': {'TimeLimit': 10, 'MIPGap_tol': 0.01}}}
    bc = BundleChoice()
    bc.load_config(cfg)
    bc.data.load_input_data(input_data)
    bc.oracles.build_quadratic_features_from_data()
    theta_0 = np.ones(num_features)
    results = bc.subproblems.initialize_and_solve_subproblems(theta_0)
    if bc._is_root():
        assert isinstance(results, np.ndarray)
        assert results.dtype == bool
        assert results.shape == (num_obs * bc.config.dimensions.num_simulations, num_items)
    else:
        assert results is None