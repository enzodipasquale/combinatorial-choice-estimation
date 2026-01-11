"""
Debug SE formula - test with and without 1/N factor.
"""
import numpy as np
from mpi4py import MPI
from bundlechoice.core import BundleChoice


def main():
    num_agents = 100
    num_items = 10
    num_agent_features = 1
    num_item_features = 2
    num_features = num_agent_features + num_item_features
    sigma = 2
    num_se_simulations = 100
    num_bootstrap = 100
    step_size = 1e-3
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    all_indices = np.arange(num_features, dtype=np.int64)
    theta_0 = np.array([2.0, 1.5, 0.8])
    
    if rank == 0:
        np.random.seed(43)
        modular_agent = np.abs(np.random.randn(num_agents, num_items, num_agent_features))
        modular_item = np.random.randn(num_items, num_item_features)
        weights = np.random.randint(1, 4, num_items).astype(float)
        capacity = np.full(num_agents, 0.5 * weights.sum())
        gen_errors = sigma * np.random.randn(num_agents, num_items)
        np.random.seed(2024)
        est_errors = sigma * np.random.randn(num_agents, num_items)
    else:
        modular_agent = modular_item = weights = capacity = gen_errors = est_errors = None
    
    modular_agent = comm.bcast(modular_agent, root=0)
    modular_item = comm.bcast(modular_item, root=0)
    weights = comm.bcast(weights, root=0)
    capacity = comm.bcast(capacity, root=0)
    gen_errors = comm.bcast(gen_errors, root=0)
    est_errors = comm.bcast(est_errors, root=0)
    
    config = {
        "dimensions": {"num_agents": num_agents, "num_items": num_items,
                      "num_features": num_features, "num_simulations": 1},
        "subproblem": {"name": "LinearKnapsack", "settings": {"TimeLimit": 5}},
        "row_generation": {"max_iters": 200, "tolerance_optimality": 0.001,
                          "theta_lbs": [0] * num_features, "theta_ubs": 100},
        "standard_errors": {"num_simulations": num_se_simulations, "step_size": step_size, "seed": 2024},
    }
    
    # Generate observed bundles
    gen_data = {
        "agent_data": {"modular": modular_agent, "capacity": capacity},
        "item_data": {"modular": modular_item, "weights": weights},
        "errors": gen_errors
    }
    bc_gen = BundleChoice()
    bc_gen.load_config(config)
    bc_gen.data.load_and_scatter(gen_data if rank == 0 else None)
    bc_gen.features.build_from_data()
    bc_gen.subproblems.load()
    obs_bundles = bc_gen.subproblems.init_and_solve(theta_0)
    obs_bundles = comm.bcast(obs_bundles, root=0)
    
    # Main estimation
    est_data = {
        "agent_data": {"modular": modular_agent, "capacity": capacity},
        "item_data": {"modular": modular_item, "weights": weights},
        "errors": est_errors, "obs_bundle": obs_bundles
    }
    bc = BundleChoice()
    bc.load_config(config)
    bc.data.load_and_scatter(est_data if rank == 0 else None)
    bc.features.build_from_data()
    bc.subproblems.load()
    
    if rank == 0:
        print("Main estimation...")
    result = bc.row_generation.solve()
    theta_hat = result.theta_hat
    
    if rank == 0:
        print(f"\nθ_hat: {theta_hat}")
    
    # Full sandwich SE
    if rank == 0:
        print("\nComputing SE matrices...")
    bc.standard_errors.clear_cache()
    se_result = bc.standard_errors.compute(
        theta_hat=theta_hat, num_simulations=num_se_simulations, step_size=step_size,
        beta_indices=all_indices, optimize_for_subset=False,
    )
    
    # Bootstrap
    if rank == 0:
        print(f"Bootstrap ({num_bootstrap} resamples)...")
    
    np.random.seed(999)
    theta_boots = []
    for b in range(num_bootstrap):
        if rank == 0:
            boot_idx = np.random.choice(num_agents, size=num_agents, replace=True)
        else:
            boot_idx = None
        boot_idx = comm.bcast(boot_idx, root=0)
        
        boot_data = {
            "agent_data": {"modular": modular_agent[boot_idx], "capacity": capacity[boot_idx]},
            "item_data": {"modular": modular_item, "weights": weights},
            "errors": est_errors[boot_idx],
            "obs_bundle": obs_bundles[boot_idx],
        }
        
        bc_b = BundleChoice()
        bc_b.load_config(config)
        bc_b.data.load_and_scatter(boot_data if rank == 0 else None)
        bc_b.features.build_from_data()
        bc_b.subproblems.load()
        
        res_b = bc_b.row_generation.solve()
        if rank == 0:
            theta_boots.append(res_b.theta_hat)
    
    if rank == 0 and se_result:
        theta_boots_arr = np.array(theta_boots)
        
        A = se_result.A_matrix
        B = se_result.B_matrix
        
        print("\n" + "=" * 80)
        print("MATRIX ANALYSIS")
        print("=" * 80)
        
        print(f"\nA matrix:\n{A}")
        print(f"\nB matrix:\n{B}")
        
        # Different variance formulas
        A_inv = np.linalg.inv(A)
        print(f"\nA_inv:\n{A_inv}")
        
        # Current formula: V = (1/N) A^{-1} B A^{-1}
        V_current = (1.0 / num_agents) * (A_inv @ B @ A_inv.T)
        
        # Alternative 1: V = A^{-1} B A^{-1} (remove outer 1/N)
        V_alt1 = A_inv @ B @ A_inv.T
        
        # Alternative 2: V = A^{-1} (N*B) A^{-1} / N = A^{-1} B A^{-1} (same as alt1)
        # This is if we define B_unscaled = sum_i g_i g_i^T (without 1/N)
        
        # Bootstrap variance
        V_boot = np.cov(theta_boots_arr.T, ddof=1)
        
        se_current = np.sqrt(np.diag(V_current))
        se_alt1 = np.sqrt(np.diag(V_alt1))
        se_boot = np.std(theta_boots_arr, axis=0, ddof=1)
        
        print("\n" + "=" * 80)
        print("VARIANCE COMPARISON")
        print("=" * 80)
        print(f"\nVariance (current): V = (1/N) * A^{-1} B A^{-1}")
        print(f"  diag(V) = {np.diag(V_current)}")
        print(f"  SE = {se_current}")
        
        print(f"\nVariance (alt1): V = A^{-1} B A^{-1}")
        print(f"  diag(V) = {np.diag(V_alt1)}")
        print(f"  SE = {se_alt1}")
        
        print(f"\nVariance (bootstrap):")
        print(f"  diag(V) = {np.diag(V_boot)}")
        print(f"  SE = {se_boot}")
        
        print("\n" + "=" * 80)
        print("RATIOS")
        print("=" * 80)
        print(f"SE_current / SE_boot = {se_current / se_boot}")
        print(f"SE_alt1 / SE_boot = {se_alt1 / se_boot}")
        
        # Check eigenvalues of A
        eig_A = np.linalg.eigvals(A)
        eig_B = np.linalg.eigvals(B)
        print(f"\nEigenvalues of A: {eig_A}")
        print(f"Eigenvalues of B: {eig_B}")
        
        # What if we use just B^{-1}?
        B_inv = np.linalg.inv(B)
        V_binv = B_inv / num_agents
        se_binv = np.sqrt(np.diag(V_binv))
        print(f"\nSE (B^{-1} only): {se_binv}")
        print(f"SE_binv / SE_boot = {se_binv / se_boot}")


if __name__ == "__main__":
    main()
