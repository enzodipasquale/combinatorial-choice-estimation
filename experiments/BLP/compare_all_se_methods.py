"""
Compare ALL SE methods: Full, Subset, B^{-1}, Bootstrap, Subsampling
Using Knapsack experiment (simpler setup)
"""
import numpy as np
from mpi4py import MPI
from bundlechoice.core import BundleChoice


def run_knapsack_experiment(num_se_sims=250, num_bootstrap=100):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    num_agents = 100
    num_items = 10
    num_agent_features = 1
    num_item_features = 2
    num_features = num_agent_features + num_item_features
    sigma = 2
    step_size = 1e-2
    
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
        "standard_errors": {"num_simulations": num_se_sims, "step_size": step_size, "seed": 2024, "error_sigma": sigma},
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
        "errors": est_errors,
        "obs_bundle": obs_bundles
    }
    bc = BundleChoice()
    bc.load_config(config)
    bc.data.load_and_scatter(est_data if rank == 0 else None)
    bc.features.build_from_data()
    bc.subproblems.load()
    
    if rank == 0:
        print("=" * 100)
        print("KNAPSACK EXPERIMENT - Comparing All SE Methods")
        print("=" * 100)
        print(f"Agents: {num_agents}, Items: {num_items}")
        print(f"SE simulations: {num_se_sims}, Bootstrap/Subsample resamples: {num_bootstrap}")
        print(f"Step size: {step_size}, Error sigma: {sigma}")
        print("\nEstimating theta...")
    
    result = bc.row_generation.solve()
    theta_hat = result.theta_hat
    
    if rank == 0:
        print(f"\nθ_hat: {theta_hat}")
        print(f"True:  {theta_0}")
    
    # 1. Full sandwich SE
    if rank == 0:
        print("\n" + "-" * 50)
        print("[1] Full Sandwich SE (A^{-1} B A^{-1} / N)...")
    bc.standard_errors.clear_cache()
    se_full = bc.standard_errors.compute(
        theta_hat=theta_hat, num_simulations=num_se_sims, step_size=step_size,
        beta_indices=all_indices, optimize_for_subset=False,
    )
    
    # 2. Subset sandwich SE  
    if rank == 0:
        print("\n[2] Subset Sandwich SE (same formula, subset indices)...")
    bc.standard_errors.clear_cache()
    se_subset = bc.standard_errors.compute(
        theta_hat=theta_hat, num_simulations=num_se_sims, step_size=step_size,
        beta_indices=all_indices, optimize_for_subset=True,
    )
    
    # 3. B^{-1} SE
    if rank == 0:
        print("\n[3] B^{-1} SE (B^{-1} / N, no finite differences)...")
    bc.standard_errors.clear_cache()
    se_binv = bc.standard_errors.compute_B_inverse(
        theta_hat=theta_hat, num_simulations=num_se_sims,
        beta_indices=all_indices,
    )
    
    # 4. Bootstrap
    if rank == 0:
        print(f"\n[4] Bootstrap ({num_bootstrap} resamples)...")
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
    
    # 5. Subsampling
    subsamp_size = int(num_agents ** 0.7)
    if rank == 0:
        print(f"\n[5] Subsampling ({num_bootstrap} resamples, subsample size={subsamp_size})...")
    np.random.seed(888)
    theta_subs = []
    for s in range(num_bootstrap):
        if rank == 0:
            sub_idx = np.random.choice(num_agents, size=subsamp_size, replace=False)
        else:
            sub_idx = None
        sub_idx = comm.bcast(sub_idx, root=0)
        
        sub_config = config.copy()
        sub_config["dimensions"] = {"num_agents": subsamp_size, "num_items": num_items,
                                    "num_features": num_features, "num_simulations": 1}
        
        sub_data = {
            "agent_data": {"modular": modular_agent[sub_idx], "capacity": capacity[sub_idx]},
            "item_data": {"modular": modular_item, "weights": weights},
            "errors": est_errors[sub_idx],
            "obs_bundle": obs_bundles[sub_idx],
        }
        bc_s = BundleChoice()
        bc_s.load_config(sub_config)
        bc_s.data.load_and_scatter(sub_data if rank == 0 else None)
        bc_s.features.build_from_data()
        bc_s.subproblems.load()
        res_s = bc_s.row_generation.solve()
        if rank == 0:
            theta_subs.append(res_s.theta_hat)
    
    if rank == 0:
        theta_boots_arr = np.array(theta_boots)
        theta_subs_arr = np.array(theta_subs)
        
        se_from_full = se_full.se_all
        se_from_subset = se_subset.se_all
        se_from_binv = se_binv.se
        se_boot = np.std(theta_boots_arr, axis=0, ddof=1)
        se_subs = np.sqrt(subsamp_size / num_agents) * np.std(theta_subs_arr, axis=0, ddof=1)
        
        print("\n" + "=" * 110)
        print("RESULTS: ALL SE METHODS COMPARISON")
        print("=" * 110)
        params = ["Agent", "Item1", "Item2"]
        print(f"\n{'Param':<10} {'θ_hat':<10} {'SE(Full)':<14} {'SE(Subset)':<14} {'SE(B⁻¹)':<14} {'SE(Boot)':<14} {'SE(Subs)':<14}")
        print("-" * 110)
        for i, p in enumerate(params):
            print(f"{p:<10} {theta_hat[i]:<10.4f} {se_from_full[i]:<14.6f} {se_from_subset[i]:<14.6f} {se_from_binv[i]:<14.6f} {se_boot[i]:<14.6f} {se_subs[i]:<14.6f}")
        
        print("\n" + "=" * 110)
        print("RATIOS (relative to Full Sandwich)")
        print("=" * 110)
        print(f"{'Param':<10} {'Subset/Full':<14} {'B⁻¹/Full':<14} {'Boot/Full':<14} {'Subs/Full':<14}")
        print("-" * 110)
        for i, p in enumerate(params):
            r_sub = se_from_subset[i]/se_from_full[i]
            r_binv = se_from_binv[i]/se_from_full[i]
            r_boot = se_boot[i]/se_from_full[i]
            r_subs = se_subs[i]/se_from_full[i]
            print(f"{p:<10} {r_sub:<14.3f} {r_binv:<14.3f} {r_boot:<14.3f} {r_subs:<14.3f}")
        
        # Summary statistics
        print("\n" + "=" * 110)
        print("SUMMARY (Average ratio across parameters)")
        print("=" * 110)
        avg_sub = np.mean([se_from_subset[i]/se_from_full[i] for i in range(len(params))])
        avg_binv = np.mean([se_from_binv[i]/se_from_full[i] for i in range(len(params))])
        avg_boot = np.mean([se_boot[i]/se_from_full[i] for i in range(len(params))])
        avg_subs = np.mean([se_subs[i]/se_from_full[i] for i in range(len(params))])
        print(f"{'Average':<10} {avg_sub:<14.3f} {avg_binv:<14.3f} {avg_boot:<14.3f} {avg_subs:<14.3f}")


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("\n" + "#" * 110)
        print("  COMPARING ALL SE METHODS")
        print("  Methods: Full Sandwich, Subset Sandwich, B^{-1}, Bootstrap, Subsampling")
        print("#" * 110 + "\n")
    
    run_knapsack_experiment(num_se_sims=250, num_bootstrap=100)
