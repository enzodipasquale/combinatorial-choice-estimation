"""Quick SE debug test for greedy and knapsack."""

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


def test_greedy():
    """Test SE with greedy scenario."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("\n" + "="*60)
        print("TEST: Greedy SE")
        print("="*60)
    
    NUM_FEATURES = 3
    scenario = (
        ScenarioLibrary.greedy()
        .with_dimensions(num_agents=300, num_items=40)
        .with_num_features(NUM_FEATURES)
        .build()
    )
    prepared = scenario.prepare(comm=comm, timeout_seconds=60, seed=42)
    
    bc = BundleChoice()
    config = {
        "dimensions": {"num_agents": 300, "num_items": 40, "num_features": NUM_FEATURES, "num_simuls": 1},
        "subproblem": {"name": "Greedy"},
        "row_generation": {"max_iters": 200, "theta_ubs": 100},
        "standard_errors": {"num_simulations": 10, "step_size": 1e-2},
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
    
    if rank == 0:
        print(f"\nEstimation done: theta = {result.theta_hat}")
    
    # Compute SE
    se_result = bc.standard_errors.compute(
        theta_hat=result.theta_hat if rank == 0 else None,
        num_simulations=10,
        seed=1995,
    )
    
    if rank == 0:
        print(f"\nSE Result:")
        print(f"  SE: {se_result.se}")
        print(f"  A cond: {np.linalg.cond(se_result.A_matrix):.2e}")
        print(f"  B cond: {np.linalg.cond(se_result.B_matrix):.2e}")
        assert np.all(se_result.se > 0), "SEs should be positive"
        print("  ✓ PASS")


def test_knapsack():
    """Test SE with knapsack scenario."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("\n" + "="*60)
        print("TEST: Knapsack SE")
        print("="*60)
    
    scenario = (
        ScenarioLibrary.linear_knapsack()
        .with_dimensions(num_agents=300, num_items=30)
        .with_feature_counts(num_agent_features=2, num_item_features=2)
        .build()
    )
    prepared = scenario.prepare(comm=comm, timeout_seconds=60, seed=42)
    
    bc = BundleChoice()
    config = {
        "dimensions": {"num_agents": 300, "num_items": 30, "num_features": 4, "num_simuls": 1},
        "subproblem": {"name": "LinearKnapsack", "settings": {"TimeLimit": 0.5}},
        "row_generation": {"max_iters": 200, "theta_ubs": 100},
        "standard_errors": {"num_simulations": 10, "step_size": 1e-2},
    }
    bc.load_config(config)
    bc.data.load_and_scatter(prepared.estimation_data if rank == 0 else None)
    bc.features.build_from_data()
    bc.subproblems.load()
    bc.subproblems.initialize_local()
    
    result = bc.row_generation.solve()
    
    if rank == 0:
        print(f"\nEstimation done: theta = {result.theta_hat}")
    
    # Compute SE
    se_result = bc.standard_errors.compute(
        theta_hat=result.theta_hat if rank == 0 else None,
        num_simulations=10,
        seed=1995,
    )
    
    if rank == 0:
        print(f"\nSE Result:")
        print(f"  SE: {se_result.se}")
        print(f"  A cond: {np.linalg.cond(se_result.A_matrix):.2e}")
        print(f"  B cond: {np.linalg.cond(se_result.B_matrix):.2e}")
        assert np.all(se_result.se > 0), "SEs should be positive"
        print("  ✓ PASS")


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    test_greedy()
    comm.Barrier()
    
    test_knapsack()
    comm.Barrier()
    
    if rank == 0:
        print("\n" + "="*60)
        print("ALL TESTS PASSED")
        print("="*60)


if __name__ == "__main__":
    main()



