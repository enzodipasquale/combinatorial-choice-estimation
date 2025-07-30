import numpy as np
import pytest
from bundlechoice.core import BundleChoice

def test_generate_data_linear():
    # Simulate config and data
    num_agents, num_items, num_simuls = 3, 6, 1
    agent_modular_dim = 3
    item_modular_dim = 2
    np.random.seed(123)
    item_data = {
        "weights": np.random.randint(1, 10, size=num_items),
        "modular": np.random.normal(0, 1, (num_items, item_modular_dim))
    }
    agent_data = {
        "modular": np.random.normal(0, 1, (num_agents, num_items, agent_modular_dim)),
        "capacity": np.random.randint(1, 100, size=num_agents),
    }
    errors = np.random.normal(0, 1, size=(num_simuls, num_agents, num_items))
    input_data = {  
        "item_data": item_data, 
        "agent_data": agent_data, 
        "errors": errors
    }

    # Compute num_features automatically
    num_features = 0
    if "modular" in agent_data:
        num_features += agent_data["modular"].shape[-1]
    if "modular" in item_data:
        num_features += item_data["modular"].shape[-1]

    cfg = {
        "dimensions": {
            "num_agents": num_agents,
            "num_items": num_items,
            "num_features": num_features,
            "num_simuls": num_simuls
        },
        "subproblem": {
            "name": "LinearKnapsack",
            "settings": {"TimeLimit": 10, "MIPGap_tol": 0.01}
        }
    }

    bc = BundleChoice()
    bc.load_config(cfg)
    bc.data.load_and_scatter(input_data)
    bc.features.build_from_data()

    # Solve pricing
    lambda_k = np.ones(num_features)
    results = bc.subproblems.init_and_solve(lambda_k)

    if bc.rank == 0:
        assert isinstance(results, np.ndarray)
        assert results.dtype == bool
        assert results.shape == (num_agents * cfg["dimensions"]["num_simuls"], num_items)
    else:
        assert results is None