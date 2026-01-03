"""
Comprehensive SE debugging for greedy and knapsack.
Tests: low-dim and fixed effects settings.
"""

import numpy as np
from mpi4py import MPI
from bundlechoice.core import BundleChoice
from bundlechoice.scenarios import ScenarioLibrary


def greedy_oracle(agent_idx: int, bundles: np.ndarray, data: dict) -> np.ndarray:
    """Greedy features: modular + quadratic."""
    modular = data["agent_data"]["modular"][agent_idx]
    modular = np.atleast_2d(modular)
    single = bundles.ndim == 1
    if single:
        bundles = bundles[:, None]
    modular_feat = modular.T @ bundles
    quad_feat = -np.sum(bundles, axis=0, keepdims=True) ** 2
    features = np.vstack((modular_feat, quad_feat))
    return features[:, 0] if single else features


def greedy_fe_oracle(agent_idx: int, bundles: np.ndarray, data: dict) -> np.ndarray:
    """Greedy features with fixed effects: modular_agent + FE + quadratic."""
    modular_agent = data["agent_data"]["modular"][agent_idx]
    modular_item = data["item_data"]["modular"]  # Identity matrix for FE
    modular_agent = np.atleast_2d(modular_agent)
    modular_item = np.atleast_2d(modular_item)
    modular = np.concatenate([modular_agent, modular_item], axis=1)
    
    single = bundles.ndim == 1
    if single:
        bundles = bundles[:, None]
    
    modular_feat = modular.T @ bundles
    quad_feat = -np.sum(bundles, axis=0, keepdims=True) ** 2
    features = np.vstack((modular_feat, quad_feat))
    return features[:, 0] if single else features


def run_greedy_lowdim(comm):
    """Test 1: Greedy low-dimensional (3 features)."""
    rank = comm.Get_rank()
    if rank == 0:
        print("\n" + "="*70)
        print("TEST 1: GREEDY LOW-DIM (3 features)")
        print("="*70)
    
    NUM_FEATURES = 3
    scenario = (
        ScenarioLibrary.greedy()
        .with_dimensions(num_agents=200, num_items=30)
        .with_num_features(NUM_FEATURES)
        .build()
    )
    prepared = scenario.prepare(comm=comm, timeout_seconds=60, seed=42)
    
    bc = BundleChoice()
    config = {
        "dimensions": {"num_agents": 200, "num_items": 30, "num_features": NUM_FEATURES, "num_simuls": 1},
        "subproblem": {"name": "Greedy"},
        "row_generation": {"max_iters": 100, "theta_ubs": 100},
    }
    bc.load_config(config)
    bc.data.load_and_scatter(prepared.estimation_data if rank == 0 else None)
    bc.features.set_oracle(greedy_oracle)
    bc.subproblems.load()
    
    from bundlechoice.subproblems.registry.greedy import GreedySubproblem
    from bundlechoice.scenarios.greedy import _install_find_best_item
    if isinstance(bc.subproblems.subproblem_instance, GreedySubproblem):
        _install_find_best_item(bc.subproblems.subproblem_instance)
    
    bc.subproblems.initialize_local()
    result = bc.row_generation.solve()
    
    # SE computation
    se_result = bc.standard_errors.compute(
        theta_hat=result.theta_hat if rank == 0 else None,
        num_simulations=10,
        step_size=0.01,
        seed=1995,
    )
    
    if rank == 0:
        if se_result is None:
            print("❌ SE computation failed!")
            return False
        print(f"\n✓ SUCCESS: SE computed for {NUM_FEATURES} params")
        print(f"  A cond: {np.linalg.cond(se_result.A_matrix):.2e}")
        print(f"  SE range: [{se_result.se.min():.4f}, {se_result.se.max():.4f}]")
        return True
    return None


def run_greedy_fe(comm):
    """Test 2: Greedy with Fixed Effects (33 features = 3 agent + 30 FE)."""
    rank = comm.Get_rank()
    if rank == 0:
        print("\n" + "="*70)
        print("TEST 2: GREEDY WITH FE (33 features)")
        print("="*70)
    
    NUM_ITEMS = 30
    NUM_AGENT_FEATURES = 2
    NUM_FEATURES = NUM_AGENT_FEATURES + NUM_ITEMS + 1  # agent + FE + quad
    
    scenario = (
        ScenarioLibrary.greedy()
        .with_dimensions(num_agents=200, num_items=NUM_ITEMS)
        .with_num_features(NUM_AGENT_FEATURES + 1)  # Scenario builds without FE
        .build()
    )
    prepared = scenario.prepare(comm=comm, timeout_seconds=60, seed=42)
    
    # Modify for FE
    if rank == 0:
        est_data = prepared.estimation_data.copy()
        est_data["item_data"] = {"modular": np.eye(NUM_ITEMS)}
    else:
        est_data = None
    
    bc = BundleChoice()
    config = {
        "dimensions": {"num_agents": 200, "num_items": NUM_ITEMS, "num_features": NUM_FEATURES, "num_simuls": 1},
        "subproblem": {"name": "Greedy"},
        "row_generation": {"max_iters": 100, "theta_ubs": 100},
    }
    bc.load_config(config)
    bc.data.load_and_scatter(est_data)
    bc.features.set_oracle(greedy_fe_oracle)
    bc.subproblems.load()
    
    from bundlechoice.subproblems.registry.greedy import GreedySubproblem
    from bundlechoice.scenarios.greedy import _install_find_best_item
    if isinstance(bc.subproblems.subproblem_instance, GreedySubproblem):
        _install_find_best_item(bc.subproblems.subproblem_instance)
    
    bc.subproblems.initialize_local()
    result = bc.row_generation.solve()
    
    # SE computation - subset for non-FE params
    beta_indices = np.array([0, 1, NUM_FEATURES - 1], dtype=np.int64)  # Agent + quad
    
    se_result = bc.standard_errors.compute(
        theta_hat=result.theta_hat if rank == 0 else None,
        num_simulations=10,
        step_size=0.01,
        beta_indices=beta_indices,
        seed=1995,
    )
    
    if rank == 0:
        if se_result is None:
            print("❌ SE computation failed!")
            return False
        print(f"\n✓ SUCCESS: SE computed for non-FE params")
        print(f"  A cond: {np.linalg.cond(se_result.A_matrix):.2e}")
        print(f"  SE: {se_result.se}")
        return True
    return None


def run_knapsack_lowdim(comm):
    """Test 3: Linear Knapsack low-dimensional (4 features)."""
    rank = comm.Get_rank()
    if rank == 0:
        print("\n" + "="*70)
        print("TEST 3: KNAPSACK LOW-DIM (4 features)")
        print("="*70)
    
    scenario = (
        ScenarioLibrary.linear_knapsack()
        .with_dimensions(num_agents=150, num_items=20)
        .with_feature_counts(num_agent_features=2, num_item_features=2)
        .build()
    )
    prepared = scenario.prepare(comm=comm, timeout_seconds=120, seed=42)
    
    bc = BundleChoice()
    config = {
        "dimensions": {"num_agents": 150, "num_items": 20, "num_features": 4, "num_simuls": 1},
        "subproblem": {"name": "LinearKnapsack", "settings": {"TimeLimit": 1.0}},
        "row_generation": {"max_iters": 100, "theta_ubs": 100},
    }
    bc.load_config(config)
    bc.data.load_and_scatter(prepared.estimation_data if rank == 0 else None)
    bc.features.build_from_data()
    bc.subproblems.load()
    bc.subproblems.initialize_local()
    
    result = bc.row_generation.solve()
    
    se_result = bc.standard_errors.compute(
        theta_hat=result.theta_hat if rank == 0 else None,
        num_simulations=10,
        step_size=0.01,
        seed=1995,
    )
    
    if rank == 0:
        if se_result is None:
            print("❌ SE computation failed!")
            return False
        print(f"\n✓ SUCCESS: SE computed for 4 params")
        print(f"  A cond: {np.linalg.cond(se_result.A_matrix):.2e}")
        print(f"  SE: {se_result.se}")
        return True
    return None


def run_knapsack_fe(comm):
    """Test 4: Linear Knapsack with Fixed Effects."""
    rank = comm.Get_rank()
    if rank == 0:
        print("\n" + "="*70)
        print("TEST 4: KNAPSACK WITH FE (24 features)")
        print("="*70)
    
    NUM_ITEMS = 20
    NUM_AGENT_FEATURES = 2
    NUM_FEATURES = NUM_AGENT_FEATURES + NUM_ITEMS + 2  # agent + FE + item
    
    scenario = (
        ScenarioLibrary.linear_knapsack()
        .with_dimensions(num_agents=150, num_items=NUM_ITEMS)
        .with_feature_counts(num_agent_features=NUM_AGENT_FEATURES, num_item_features=2)
        .build()
    )
    prepared = scenario.prepare(comm=comm, timeout_seconds=120, seed=42)
    
    # Modify for FE
    if rank == 0:
        est_data = prepared.estimation_data.copy()
        orig_item = est_data["item_data"]["modular"]
        fe_matrix = np.eye(NUM_ITEMS)
        est_data["item_data"]["modular"] = np.hstack([fe_matrix, orig_item])
    else:
        est_data = None
    
    bc = BundleChoice()
    config = {
        "dimensions": {"num_agents": 150, "num_items": NUM_ITEMS, "num_features": NUM_FEATURES, "num_simuls": 1},
        "subproblem": {"name": "LinearKnapsack", "settings": {"TimeLimit": 1.0}},
        "row_generation": {"max_iters": 100, "theta_ubs": 100},
    }
    bc.load_config(config)
    bc.data.load_and_scatter(est_data)
    bc.features.build_from_data()
    bc.subproblems.load()
    bc.subproblems.initialize_local()
    
    result = bc.row_generation.solve()
    
    # SE for non-FE params only
    beta_indices = np.concatenate([
        np.arange(NUM_AGENT_FEATURES),
        np.arange(NUM_AGENT_FEATURES + NUM_ITEMS, NUM_FEATURES)
    ]).astype(np.int64)
    
    se_result = bc.standard_errors.compute(
        theta_hat=result.theta_hat if rank == 0 else None,
        num_simulations=10,
        step_size=0.01,
        beta_indices=beta_indices,
        seed=1995,
    )
    
    if rank == 0:
        if se_result is None:
            print("❌ SE computation failed!")
            return False
        print(f"\n✓ SUCCESS: SE computed for non-FE params")
        print(f"  A cond: {np.linalg.cond(se_result.A_matrix):.2e}")
        print(f"  SE: {se_result.se}")
        return True
    return None


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("\n" + "="*70)
        print("COMPREHENSIVE SE DEBUGGING")
        print("="*70)
    
    results = {}
    
    # Run all tests
    results["greedy_lowdim"] = run_greedy_lowdim(comm)
    comm.Barrier()
    
    results["greedy_fe"] = run_greedy_fe(comm)
    comm.Barrier()
    
    results["knapsack_lowdim"] = run_knapsack_lowdim(comm)
    comm.Barrier()
    
    results["knapsack_fe"] = run_knapsack_fe(comm)
    comm.Barrier()
    
    # Summary
    if rank == 0:
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        for name, success in results.items():
            status = "✓ PASS" if success else "❌ FAIL"
            print(f"  {name}: {status}")


if __name__ == "__main__":
    main()

