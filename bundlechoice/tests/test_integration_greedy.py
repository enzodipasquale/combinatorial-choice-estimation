import numpy as np
import pytest
import time
from bundlechoice.core import BundleChoice

def generate_test_data(num_agents, num_items, agent_modular_dim, item_modular_dim, num_simuls):
    """Generate test data for agents and items."""
    item_data = {"modular": np.random.normal(0, 1, (num_items, item_modular_dim))}
    agent_data = {
        "modular": np.random.normal(0, 1, (num_agents, num_items, agent_modular_dim)),
        "capacity": np.random.randint(1, 100, size=num_agents),
    }
    errors = np.random.normal(0, 1, size=(num_simuls, num_agents, num_items))
    input_data = {"item_data": item_data, "agent_data": agent_data, "errors": errors}
    
    num_features = agent_modular_dim + item_modular_dim + 1
    dimensions_cfg = {
        "num_agents": num_agents,
        "num_items": num_items,
        "num_features": num_features,
        "num_simuls": num_simuls
    }
    
    return input_data, dimensions_cfg

def get_features(i_id, B_j, data):
    """
    Compute features for a given agent and bundle(s).
    Supports both single (1D) and multiple (2D) bundles.
    Returns array of shape (num_features,) for a single bundle,
    or (num_features, m) for m bundles.
    """
    modular_agent = data["agent_data"]["modular"][i_id]
    modular_item = data["item_data"]["modular"]

    modular_agent = np.atleast_2d(modular_agent)
    modular_item = np.atleast_2d(modular_item)

    single_bundle = False
    if B_j.ndim == 1:
        B_j = B_j[:, None]
        single_bundle = True

    agent_sum = modular_agent.T @ B_j
    item_sum = modular_item.T @ B_j
    neg_sq = -np.sum(B_j, axis=0, keepdims=True) ** 2

    features = np.vstack((agent_sum, item_sum, neg_sq))
    if single_bundle:
        return features[:, 0]  # Return as 1D array for a single bundle
    return features

def test_greedy_vs_bruteforce():
    """Test that greedy solver finds the same optimal bundles as brute force."""
    num_agents, num_items, num_simuls = 20, 12, 2
    agent_modular_dim, item_modular_dim = 2, 3
    
    input_data, dimensions_cfg = generate_test_data(
        num_agents, num_items, agent_modular_dim, item_modular_dim, num_simuls
    )

    cfg = {
        "dimensions": dimensions_cfg,
        "subproblem": {"name": "Greedy", "settings": {}}
    }

    bc = BundleChoice()
    bc.load_config(cfg)
    bc.load_data(input_data, scatter=True)
    bc.load_features(get_features)

    test_lambdas = [
        np.ones(dimensions_cfg["num_features"]),
        np.random.normal(0, 1, dimensions_cfg["num_features"]),
        np.array([1] * (dimensions_cfg["num_features"] - 1) + [0.1]),
    ]
    for i in range(len(test_lambdas)):
        if test_lambdas[i][-1] <= 0:
            test_lambdas[i][-1] = abs(test_lambdas[i][-1]) + 0.1

    for idx, lambda_k in enumerate(test_lambdas):
        if bc.rank == 0:
            print(f"\nTesting lambda_k {idx+1}: {lambda_k}")
        t0 = time.time()
        greedy_bundles = bc.init_and_solve_subproblems(lambda_k)
        t1 = time.time()
        assert bc.subproblem_manager is not None, "Subproblem manager should be initialized"
        result = bc.subproblem_manager.find_max_bundle_bruteforce(lambda_k)
        t2 = time.time()
        if bc.rank == 0:
            assert greedy_bundles is not None, "Greedy results should not be None at rank 0"
            assert result is not None, "Brute force results should not be None at rank 0"
            brute_vals, brute_bundles = result
            print(f"Greedy time: {t1-t0:.3f}s | Brute force: {t2-t1:.3f}s | Speedup: {(t2-t1)/(t1-t0):.1f}x")
            # Print bundle sizes for greedy and brute force
            # print("Greedy bundle sizes:", [int(np.sum(bundle)) for bundle in greedy_bundles])
            # print("Brute force bundle sizes:", [int(np.sum(bundle)) for bundle in brute_bundles])
            assert np.array_equal(greedy_bundles, brute_bundles), f"Mismatch for lambda_k {idx+1}"
            print(f"✅ Test {idx+1} passed: Greedy and brute force results match")
        else:
            assert greedy_bundles is None, "Greedy results should be None at non-root ranks"
            assert result == (None,None), "Brute force results should be None at non-root ranks " 

