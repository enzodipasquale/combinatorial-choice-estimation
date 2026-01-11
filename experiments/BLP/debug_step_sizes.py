"""
Debug: How does step size affect A matrix and SE?
"""
import numpy as np
from mpi4py import MPI
from bundlechoice.core import BundleChoice


def main():
    num_agents = 100
    num_items = 10
    num_features = 3
    sigma = 2
    num_se_simulations = 100
    num_bootstrap = 50
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    all_indices = np.arange(num_features, dtype=np.int64)
    theta_0 = np.array([2.0, 1.5, 0.8])
    
    if rank == 0:
        np.random.seed(43)
        modular_agent = np.abs(np.random.randn(num_agents, num_items, 1))
        modular_item = np.random.randn(num_items, 2)
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
        "standard_errors": {"num_simulations": num_se_simulations, "step_size": 1e-3, "seed": 2024, "error_sigma": sigma},
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
    
    # Bootstrap first to get ground truth
    if rank == 0:
        print(f"\nBootstrap ({num_bootstrap} resamples)...")
    
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
    
    if rank == 0:
        theta_boots_arr = np.array(theta_boots)
        se_boot = np.std(theta_boots_arr, axis=0, ddof=1)
        print(f"\nSE(Boot): {se_boot}")
    
    # Test different step sizes
    step_sizes = [1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 0.1, 0.2, 0.5]
    
    if rank == 0:
        print("\n" + "=" * 100)
        print("STEP SIZE ANALYSIS")
        print("=" * 100)
        print(f"\n{'Step':<10} {'A_eig_min':<12} {'A_cond':<12} {'SE[0]':<12} {'SE[1]':<12} {'SE[2]':<12} {'Ratio[0]':<12}")
        print("-" * 100)
    
    for step in step_sizes:
        bc.standard_errors.clear_cache()
        se_result = bc.standard_errors.compute(
            theta_hat=theta_hat, num_simulations=num_se_simulations, step_size=step,
            beta_indices=all_indices, optimize_for_subset=False,
        )
        
        if rank == 0 and se_result:
            A = se_result.A_matrix
            eig_A = np.abs(np.linalg.eigvals(A))
            cond_A = np.max(eig_A) / np.min(eig_A)
            se_full = se_result.se_all
            ratio = se_full[0] / se_boot[0] if se_boot[0] > 0 else np.nan
            
            print(f"{step:<10.0e} {np.min(eig_A):<12.6f} {cond_A:<12.1f} {se_full[0]:<12.6f} {se_full[1]:<12.6f} {se_full[2]:<12.6f} {ratio:<12.2f}")


if __name__ == "__main__":
    main()
