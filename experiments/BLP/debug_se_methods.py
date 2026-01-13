"""
Debug SE methods - small setting to identify discrepancy.
100 agents, 10 items, no FE.
"""
import numpy as np
from mpi4py import MPI
from bundlechoice.core import BundleChoice


def debug_greedy():
    """Debug greedy with detailed output."""
    num_agents = 100
    num_items = 10
    num_agent_features = 1
    num_item_features = 2
    num_features = num_agent_features + num_item_features + 1  # +1 for quadratic
    sigma = 2
    num_se_simulations = 100
    num_bootstrap = 100
    step_size = 1e-2
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("=" * 80)
        print("DEBUG GREEDY: 100 agents, 10 items, no FE")
        print("=" * 80)
    
    all_indices = np.arange(num_features, dtype=np.int64)
    theta_0 = np.array([2.0, 1.5, 0.8, 0.1])  # Agent, Item1, Item2, Quadratic
    
    # Generate data
    if rank == 0:
        np.random.seed(42)
        modular_agent = np.abs(np.random.randn(num_agents, num_items, num_agent_features))
        modular_item = np.random.randn(num_items, num_item_features)
        gen_errors = sigma * np.random.randn(num_agents, num_items)
        np.random.seed(2024)
        est_errors = sigma * np.random.randn(num_agents, num_items)
    else:
        modular_agent = modular_item = gen_errors = est_errors = None
    
    modular_agent = comm.bcast(modular_agent, root=0)
    modular_item = comm.bcast(modular_item, root=0)
    gen_errors = comm.bcast(gen_errors, root=0)
    est_errors = comm.bcast(est_errors, root=0)
    
    config = {
        "dimensions": {"num_agents": num_agents, "num_items": num_items, 
                      "num_features": num_features, "num_simulations": 1},
        "subproblem": {"name": "Greedy"},
        "row_generation": {"max_iters": 200, "tolerance_optimality": 0.001,
                          "theta_lbs": [0] * num_features, "theta_ubs": 100},
        "standard_errors": {"num_simulations": num_se_simulations, "step_size": step_size, "seed": 2024, "error_sigma": sigma},
    }
    
    # Generate observed bundles
    gen_data = {"agent_data": {"modular": modular_agent}, "item_data": {"modular": modular_item}, "errors": gen_errors}
    bc_gen = BundleChoice()
    bc_gen.load_config(config)
    bc_gen.data.load_and_scatter(gen_data if rank == 0 else None)
    bc_gen.oracles.set_features_oracle(_greedy_oracle)
    bc_gen.subproblems.load()
    _install_greedy_find_best(bc_gen.subproblems.subproblem_instance)
    obs_bundles = bc_gen.subproblems.init_and_solve(theta_0)
    obs_bundles = comm.bcast(obs_bundles, root=0)
    
    # Main estimation
    est_data = {"agent_data": {"modular": modular_agent}, "item_data": {"modular": modular_item}, 
                "errors": est_errors, "obs_bundle": obs_bundles}
    bc = BundleChoice()
    bc.load_config(config)
    bc.data.load_and_scatter(est_data if rank == 0 else None)
    bc.oracles.set_features_oracle(_greedy_oracle)
    bc.subproblems.load()
    _install_greedy_find_best(bc.subproblems.subproblem_instance)
    
    if rank == 0:
        print("\nMain estimation...")
    result = bc.row_generation.solve()
    theta_hat = result.theta_hat
    
    if rank == 0:
        print(f"\nθ_hat: {theta_hat}")
        print(f"True:  {theta_0}")
    
    # Full sandwich SE
    if rank == 0:
        print("\n[1] Full Sandwich SE...")
    bc.standard_errors.clear_cache()
    se_full = bc.standard_errors.compute(
        theta_hat=theta_hat, num_simulations=num_se_simulations, step_size=step_size,
        beta_indices=all_indices, optimize_for_subset=False,
    )
    
    # Bootstrap
    if rank == 0:
        print(f"\n[2] Bootstrap ({num_bootstrap} resamples)...")
    
    np.random.seed(999)
    theta_boots = []
    for b in range(num_bootstrap):
        if rank == 0:
            boot_idx = np.random.choice(num_agents, size=num_agents, replace=True)
        else:
            boot_idx = None
        boot_idx = comm.bcast(boot_idx, root=0)
        
        boot_data = {
            "agent_data": {"modular": modular_agent[boot_idx]},
            "item_data": {"modular": modular_item},
            "errors": est_errors[boot_idx],
            "obs_bundle": obs_bundles[boot_idx],
        }
        
        bc_b = BundleChoice()
        bc_b.load_config(config)
        bc_b.data.load_and_scatter(boot_data if rank == 0 else None)
        bc_b.oracles.set_features_oracle(_greedy_oracle)
        bc_b.subproblems.load()
        _install_greedy_find_best(bc_b.subproblems.subproblem_instance)
        
        res_b = bc_b.row_generation.solve()
        if rank == 0:
            theta_boots.append(res_b.theta_hat)
            if (b + 1) % 20 == 0:
                print(f"    Bootstrap {b+1}/{num_bootstrap}")
    
    # Subsampling
    subsamp_size = int(num_agents ** 0.7)
    if rank == 0:
        print(f"\n[3] Subsampling ({num_bootstrap} subsamples, size={subsamp_size})...")
    
    theta_subs = []
    for s in range(num_bootstrap):
        if rank == 0:
            sub_idx = np.random.choice(num_agents, size=subsamp_size, replace=False)
        else:
            sub_idx = None
        sub_idx = comm.bcast(sub_idx, root=0)
        
        sub_data = {
            "agent_data": {"modular": modular_agent[sub_idx]},
            "item_data": {"modular": modular_item},
            "errors": est_errors[sub_idx],
            "obs_bundle": obs_bundles[sub_idx],
        }
        
        bc_s = BundleChoice()
        cfg_s = dict(config)
        cfg_s["dimensions"] = dict(config["dimensions"])
        cfg_s["dimensions"]["num_agents"] = subsamp_size
        bc_s.load_config(cfg_s)
        bc_s.data.load_and_scatter(sub_data if rank == 0 else None)
        bc_s.oracles.set_features_oracle(_greedy_oracle)
        bc_s.subproblems.load()
        _install_greedy_find_best(bc_s.subproblems.subproblem_instance)
        
        res_s = bc_s.row_generation.solve()
        if rank == 0:
            theta_subs.append(res_s.theta_hat)
            if (s + 1) % 20 == 0:
                print(f"    Subsample {s+1}/{num_bootstrap}")
    
    if rank == 0 and se_full:
        theta_boots_arr = np.array(theta_boots)
        theta_subs_arr = np.array(theta_subs)
        
        se_full_arr = se_full.se_all
        se_boot = np.std(theta_boots_arr, axis=0, ddof=1)
        se_subs = np.sqrt(subsamp_size / num_agents) * np.std(theta_subs_arr, axis=0, ddof=1)
        
        # Debug: Print A and B matrix info
        print("\n" + "=" * 80)
        print("DEBUG INFO")
        print("=" * 80)
        print(f"A matrix condition number: {np.linalg.cond(se_full.A_matrix):.2e}")
        print(f"B matrix condition number: {np.linalg.cond(se_full.B_matrix):.2e}")
        print(f"A matrix diagonal: {np.diag(se_full.A_matrix)}")
        print(f"B matrix diagonal: {np.diag(se_full.B_matrix)}")
        print(f"Variance diagonal: {np.diag(se_full.variance)}")
        
        print("\n" + "=" * 80)
        print("RESULTS TABLE")
        print("=" * 80)
        print(f"\n{'Param':<12} {'θ_hat':<10} {'SE(Full)':<12} {'SE(Boot)':<12} {'SE(Subs)':<12}")
        print("-" * 60)
        names = ["Agent", "Item1", "Item2", "Quadratic"]
        for i in range(num_features):
            print(f"{names[i]:<12} {theta_hat[i]:>8.4f}  {se_full_arr[i]:>10.6f}  {se_boot[i]:>10.6f}  {se_subs[i]:>10.6f}")
        
        print("\n" + "=" * 80)
        print("RATIOS (Boot/Full, Subs/Full)")
        print("=" * 80)
        for i in range(num_features):
            r_boot = se_boot[i] / se_full_arr[i] if se_full_arr[i] > 0 else np.nan
            r_subs = se_subs[i] / se_full_arr[i] if se_full_arr[i] > 0 else np.nan
            print(f"{names[i]:<12} Boot/Full: {r_boot:.3f}, Subs/Full: {r_subs:.3f}")


def debug_knapsack():
    """Debug knapsack with detailed output."""
    num_agents = 100
    num_items = 10
    num_agent_features = 1
    num_item_features = 2
    num_features = num_agent_features + num_item_features
    sigma = 2
    num_se_simulations = 100
    num_bootstrap = 100
    step_size = 1e-2
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("\n" + "=" * 80)
        print("DEBUG KNAPSACK: 100 agents, 10 items, no FE")
        print("=" * 80)
    
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
        "standard_errors": {"num_simulations": num_se_simulations, "step_size": step_size, "seed": 2024, "error_sigma": sigma},
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
        print("\nMain estimation...")
    result = bc.row_generation.solve()
    theta_hat = result.theta_hat
    
    if rank == 0:
        print(f"\nθ_hat: {theta_hat}")
        print(f"True:  {theta_0}")
    
    # Full sandwich SE
    if rank == 0:
        print("\n[1] Full Sandwich SE...")
    bc.standard_errors.clear_cache()
    se_full = bc.standard_errors.compute(
        theta_hat=theta_hat, num_simulations=num_se_simulations, step_size=step_size,
        beta_indices=all_indices, optimize_for_subset=False,
    )
    
    # Bootstrap
    if rank == 0:
        print(f"\n[2] Bootstrap ({num_bootstrap} resamples)...")
    
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
            if (b + 1) % 20 == 0:
                print(f"    Bootstrap {b+1}/{num_bootstrap}")
    
    # Subsampling
    subsamp_size = int(num_agents ** 0.7)
    if rank == 0:
        print(f"\n[3] Subsampling ({num_bootstrap} subsamples, size={subsamp_size})...")
    
    theta_subs = []
    for s in range(num_bootstrap):
        if rank == 0:
            sub_idx = np.random.choice(num_agents, size=subsamp_size, replace=False)
        else:
            sub_idx = None
        sub_idx = comm.bcast(sub_idx, root=0)
        
        sub_data = {
            "agent_data": {"modular": modular_agent[sub_idx], "capacity": capacity[sub_idx]},
            "item_data": {"modular": modular_item, "weights": weights},
            "errors": est_errors[sub_idx],
            "obs_bundle": obs_bundles[sub_idx],
        }
        
        bc_s = BundleChoice()
        cfg_s = dict(config)
        cfg_s["dimensions"] = dict(config["dimensions"])
        cfg_s["dimensions"]["num_agents"] = subsamp_size
        bc_s.load_config(cfg_s)
        bc_s.data.load_and_scatter(sub_data if rank == 0 else None)
        bc_s.oracles.build_from_data()
        bc_s.subproblems.load()
        
        res_s = bc_s.row_generation.solve()
        if rank == 0:
            theta_subs.append(res_s.theta_hat)
            if (s + 1) % 20 == 0:
                print(f"    Subsample {s+1}/{num_bootstrap}")
    
    if rank == 0 and se_full:
        theta_boots_arr = np.array(theta_boots)
        theta_subs_arr = np.array(theta_subs)
        
        se_full_arr = se_full.se_all
        se_boot = np.std(theta_boots_arr, axis=0, ddof=1)
        se_subs = np.sqrt(subsamp_size / num_agents) * np.std(theta_subs_arr, axis=0, ddof=1)
        
        # Debug: Print A and B matrix info
        print("\n" + "=" * 80)
        print("DEBUG INFO")
        print("=" * 80)
        print(f"A matrix condition number: {np.linalg.cond(se_full.A_matrix):.2e}")
        print(f"B matrix condition number: {np.linalg.cond(se_full.B_matrix):.2e}")
        print(f"A matrix diagonal: {np.diag(se_full.A_matrix)}")
        print(f"B matrix diagonal: {np.diag(se_full.B_matrix)}")
        print(f"Variance diagonal: {np.diag(se_full.variance)}")
        
        print("\n" + "=" * 80)
        print("RESULTS TABLE")
        print("=" * 80)
        print(f"\n{'Param':<12} {'θ_hat':<10} {'SE(Full)':<12} {'SE(Boot)':<12} {'SE(Subs)':<12}")
        print("-" * 60)
        names = ["Agent", "Item1", "Item2"]
        for i in range(num_features):
            print(f"{names[i]:<12} {theta_hat[i]:>8.4f}  {se_full_arr[i]:>10.6f}  {se_boot[i]:>10.6f}  {se_subs[i]:>10.6f}")
        
        print("\n" + "=" * 80)
        print("RATIOS (Boot/Full, Subs/Full)")
        print("=" * 80)
        for i in range(num_features):
            r_boot = se_boot[i] / se_full_arr[i] if se_full_arr[i] > 0 else np.nan
            r_subs = se_subs[i] / se_full_arr[i] if se_full_arr[i] > 0 else np.nan
            print(f"{names[i]:<12} Boot/Full: {r_boot:.3f}, Subs/Full: {r_subs:.3f}")


def _greedy_oracle(agent_idx, bundles, data):
    modular_agent = data["agent_data"]["modular"][agent_idx]
    modular_item = data["item_data"]["modular"]
    if bundles.ndim == 1:
        return np.concatenate([modular_agent.T @ bundles, modular_item.T @ bundles, [-bundles.sum() ** 2]])
    else:
        return np.vstack([modular_agent.T @ bundles, modular_item.T @ bundles, -np.sum(bundles, axis=0, keepdims=True) ** 2])


def _install_greedy_find_best(solver):
    def find_best_item(local_id, base_bundle, items_left, theta, error_j):
        modular_agent = solver.local_data["agent_data"]["modular"][local_id]
        modular_item = solver.local_data["item_data"]["modular"]
        num_agent_f = modular_agent.shape[1]
        num_item_f = modular_item.shape[1]
        theta_agent = theta[:num_agent_f]
        theta_item = theta[num_agent_f:num_agent_f + num_item_f]
        theta_quad = theta[-1]
        new_size = base_bundle.sum() + 1
        quad_term = theta_quad * (-new_size ** 2)
        base_agent = modular_agent.T @ base_bundle
        base_item = modular_item.T @ base_bundle
        cand_agent = base_agent[None, :] + modular_agent[items_left, :]
        cand_item = base_item[None, :] + modular_item[items_left, :]
        values = cand_agent @ theta_agent + cand_item @ theta_item + error_j[items_left] + quad_term
        best_idx = np.argmax(values)
        return items_left[best_idx], values[best_idx]
    solver.find_best_item = find_best_item


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("\n" + "#" * 80)
        print("  DEBUG SE METHODS - SMALL SETTING")
        print("#" * 80)
    
    debug_greedy()
    comm.Barrier()
    debug_knapsack()


if __name__ == "__main__":
    main()
