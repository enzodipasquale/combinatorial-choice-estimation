"""
Debug numerics: Check if B matrix computation has issues.
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
        "standard_errors": {"num_simulations": num_se_simulations, "step_size": 0.1, "seed": 2024},
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
    bc_gen.oracles.build_from_data()
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
    bc.oracles.build_from_data()
    bc.subproblems.load()
    
    if rank == 0:
        print("Main estimation...")
    result = bc.row_generation.solve()
    theta_hat = result.theta_hat
    
    if rank == 0:
        print(f"θ_hat: {theta_hat}")
    
    # Now manually compute SE components to debug
    if rank == 0:
        print("\n" + "=" * 80)
        print("NUMERICAL DEBUG")
        print("=" * 80)
    
    # Get observed features
    obs_features = bc.oracles.compute_gathered_features(obs_bundles)
    
    # Compute per-agent subgradients with multiple simulations
    all_g_i = []  # List of (N, K) arrays, one per simulation
    
    for s in range(num_se_simulations):
        # Generate new errors for this simulation
        if rank == 0:
            sim_errors = sigma * np.random.randn(num_agents, num_items)
        else:
            sim_errors = None
        sim_errors = comm.bcast(sim_errors, root=0)
        
        bc.data.update_errors(sim_errors if rank == 0 else None)
        sim_bundles = bc.subproblems.solve_local(theta_hat)
        sim_features = bc.oracles.compute_gathered_features(sim_bundles)
        
        if rank == 0:
            g_i_s = sim_features - obs_features  # (N, K)
            all_g_i.append(g_i_s)
    
    if rank == 0:
        all_g_i = np.array(all_g_i)  # (S, N, K)
        
        # Average over simulations for each agent
        g_i_avg = all_g_i.mean(axis=0)  # (N, K)
        
        print(f"\nPer-agent subgradient g_i (averaged over {num_se_simulations} sims):")
        print(f"  Shape: {g_i_avg.shape}")
        print(f"  Mean (should be ~0): {g_i_avg.mean(axis=0)}")
        print(f"  Std: {g_i_avg.std(axis=0)}")
        
        # B matrix using second moment
        B_second_moment = (g_i_avg.T @ g_i_avg) / num_agents
        
        # B matrix using centered covariance
        g_i_centered = g_i_avg - g_i_avg.mean(axis=0)
        B_covariance = (g_i_centered.T @ g_i_centered) / num_agents
        
        print(f"\nB matrix (second moment):")
        print(f"  diag = {np.diag(B_second_moment)}")
        
        print(f"\nB matrix (covariance, centered):")
        print(f"  diag = {np.diag(B_covariance)}")
        
        # Compare
        print(f"\nRatio (second moment / covariance):")
        print(f"  {np.diag(B_second_moment) / np.diag(B_covariance)}")
        
        # Now compute A matrix manually
        print("\n" + "-" * 40)
        print("A matrix computation")
        
        step_size = 0.1
        A = np.zeros((num_features, num_features))
        
        # For each parameter, compute finite difference
        for k in range(num_features):
            h = step_size * abs(theta_hat[k]) if abs(theta_hat[k]) > 0.1 else step_size * 0.1
            
            theta_plus = theta_hat.copy()
            theta_plus[k] += h
            theta_minus = theta_hat.copy()
            theta_minus[k] -= h
            
            # Compute g_bar at theta_plus and theta_minus using SAME errors as B matrix
            g_bar_plus = np.zeros(num_features)
            g_bar_minus = np.zeros(num_features)
            
            for s in range(min(20, num_se_simulations)):  # Use fewer sims for speed
                np.random.seed(2024 + s)  # Same seed as B matrix computation
                sim_errors = sigma * np.random.randn(num_agents, num_items)
                
                bc.data.update_errors(sim_errors if rank == 0 else None)
                
                sim_bundles_plus = bc.subproblems.solve_local(theta_plus)
                sim_features_plus = bc.oracles.compute_gathered_features(sim_bundles_plus)
                g_bar_plus += (sim_features_plus - obs_features).mean(axis=0)
                
                sim_bundles_minus = bc.subproblems.solve_local(theta_minus)
                sim_features_minus = bc.oracles.compute_gathered_features(sim_bundles_minus)
                g_bar_minus += (sim_features_minus - obs_features).mean(axis=0)
            
            g_bar_plus /= min(20, num_se_simulations)
            g_bar_minus /= min(20, num_se_simulations)
            
            A[:, k] = (g_bar_plus - g_bar_minus) / (2 * h)
            print(f"  Column {k}: h={h:.4f}, g_bar_plus={g_bar_plus}, g_bar_minus={g_bar_minus}")
        
        print(f"\nA matrix (manual):")
        print(A)
        print(f"  diag = {np.diag(A)}")
        print(f"  eigenvalues = {np.linalg.eigvals(A)}")
        
        # Compute variance with both B versions
        A_inv = np.linalg.inv(A)
        
        V_second = (1.0 / num_agents) * (A_inv @ B_second_moment @ A_inv.T)
        V_cov = (1.0 / num_agents) * (A_inv @ B_covariance @ A_inv.T)
        
        se_second = np.sqrt(np.diag(V_second))
        se_cov = np.sqrt(np.diag(V_cov))
        
        print(f"\nSE comparison:")
        print(f"  Using second moment B: {se_second}")
        print(f"  Using covariance B:    {se_cov}")
        
        # Bootstrap for reference
        print("\nRunning bootstrap for reference...")
    
    # Bootstrap
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
        bc_b.oracles.build_from_data()
        bc_b.subproblems.load()
        
        res_b = bc_b.row_generation.solve()
        if rank == 0:
            theta_boots.append(res_b.theta_hat)
    
    if rank == 0:
        theta_boots_arr = np.array(theta_boots)
        se_boot = np.std(theta_boots_arr, axis=0, ddof=1)
        
        print(f"\nSE (bootstrap): {se_boot}")
        print(f"\nRatios to bootstrap:")
        print(f"  SE_second / SE_boot: {se_second / se_boot}")
        print(f"  SE_cov / SE_boot:    {se_cov / se_boot}")


if __name__ == "__main__":
    main()
