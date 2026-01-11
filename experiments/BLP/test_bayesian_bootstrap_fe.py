"""
Test Bayesian Bootstrap with Fixed Effects.
Compare Standard vs Bayesian bootstrap for Greedy (FE case).
"""
import numpy as np
from mpi4py import MPI
from bundlechoice.core import BundleChoice


def run_greedy_fe_experiment(num_agents=200, num_bootstrap=50):
    """Run Greedy with FE experiment."""
    num_items = 5
    num_agent_features = 1
    num_fe = num_items
    num_features = num_agent_features + 1 + num_fe  # agent + quad + FE
    sigma = 1.5
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("=" * 70)
        print(f"GREEDY WITH FE: {num_agents} agents, {num_items} items")
        print("=" * 70)
    
    subset_indices = np.array([0, 1], dtype=np.int64)  # Agent, Quadratic
    
    # True parameters
    np.random.seed(123)
    theta_0 = np.zeros(num_features)
    theta_0[0] = 3.0   # Agent
    theta_0[1] = 0.2   # Quadratic
    theta_0[2:] = np.abs(np.random.randn(num_fe)) * 1.5 + 0.5  # Positive FE
    
    if rank == 0:
        np.random.seed(42)
        modular_agent = np.abs(np.random.randn(num_agents, num_items, num_agent_features))
        gen_errors = sigma * np.random.randn(num_agents, num_items)
        np.random.seed(2024)
        est_errors = sigma * np.random.randn(num_agents, num_items)
    else:
        modular_agent = gen_errors = est_errors = None
    
    modular_agent = comm.bcast(modular_agent, root=0)
    gen_errors = comm.bcast(gen_errors, root=0)
    est_errors = comm.bcast(est_errors, root=0)
    theta_0 = comm.bcast(theta_0, root=0)
    
    # Bounds: positive FE
    theta_lbs = [0.0, 0.0] + [0.0] * num_fe
    theta_ubs = [20.0, 5.0] + [10.0] * num_fe
    
    config = {
        "dimensions": {"num_agents": num_agents, "num_items": num_items, 
                      "num_features": num_features, "num_simulations": 1},
        "subproblem": {"name": "Greedy"},
        "row_generation": {"max_iters": 200, "tolerance_optimality": 0.001,
                          "theta_lbs": theta_lbs, "theta_ubs": theta_ubs},
    }
    
    # Oracle for greedy with FE
    def _oracle_fe(agent_idx, bundles, data):
        modular = data["agent_data"]["modular"][agent_idx]
        if bundles.ndim == 1:
            bundle_size = bundles.sum()
            agent_part = (modular[:, 0] * bundles).sum()
            quad_part = -bundle_size ** 2
            fe_part = bundles.astype(np.float64)
            return np.concatenate([[agent_part, quad_part], fe_part])
        else:
            bundle_sizes = bundles.sum(axis=1)
            agent_parts = (modular[:, 0] * bundles).sum(axis=1)
            quad_parts = -bundle_sizes ** 2
            fe_parts = bundles.astype(np.float64)
            return np.column_stack([agent_parts, quad_parts, fe_parts])
    
    # Generate observed bundles
    gen_data = {"agent_data": {"modular": modular_agent}, "item_data": {}, "errors": gen_errors}
    bc_gen = BundleChoice()
    bc_gen.load_config(config)
    bc_gen.data.load_and_scatter(gen_data if rank == 0 else None)
    bc_gen.features.set_oracle(_oracle_fe)
    bc_gen.subproblems.load()
    obs_bundles = bc_gen.subproblems.init_and_solve(theta_0)
    obs_bundles = comm.bcast(obs_bundles, root=0)
    
    # Validate bundles
    if rank == 0:
        bundle_sizes = obs_bundles.sum(axis=1)
        items_chosen = obs_bundles.sum(axis=0)
        print(f"  Bundles: min={bundle_sizes.min()}, max={bundle_sizes.max()}, mean={bundle_sizes.mean():.1f}")
        print(f"  Items chosen: min={items_chosen.min()}, max={items_chosen.max()}")
    
    # Main estimation
    est_data = {"agent_data": {"modular": modular_agent}, "item_data": {}, 
                "errors": est_errors, "obs_bundle": obs_bundles}
    bc = BundleChoice()
    bc.load_config(config)
    bc.data.load_and_scatter(est_data if rank == 0 else None)
    bc.features.set_oracle(_oracle_fe)
    bc.subproblems.load()
    
    result = bc.row_generation.solve()
    theta_hat = result.theta_hat
    
    if rank == 0:
        print(f"  θ_hat (non-FE): {theta_hat[subset_indices]}")
        print(f"  True (non-FE):  {theta_0[subset_indices]}")
    
    # Standard Bootstrap
    if rank == 0:
        print(f"\n  Standard Bootstrap ({num_bootstrap})...")
    theta_boots_std = []
    np.random.seed(999)
    boot_indices_list = [np.random.choice(num_agents, num_agents, replace=True) for _ in range(num_bootstrap)]
    boot_indices_list = comm.bcast(boot_indices_list, root=0)
    
    for b_idx, boot_indices in enumerate(boot_indices_list):
        if rank == 0 and (b_idx + 1) % 20 == 0:
            print(f"    Std Boot {b_idx + 1}/{num_bootstrap}")
        
        boot_data = {
            "agent_data": {"modular": modular_agent[boot_indices]}, 
            "item_data": {}, 
            "errors": est_errors[boot_indices], 
            "obs_bundle": obs_bundles[boot_indices]
        }
        bc_b = BundleChoice()
        bc_b.load_config(config)
        bc_b.data.load_and_scatter(boot_data if rank == 0 else None)
        bc_b.features.set_oracle(_oracle_fe)
        bc_b.subproblems.load()
        
        try:
            res_b = bc_b.row_generation.solve()
            if rank == 0:
                theta_boots_std.append(res_b.theta_hat)
        except:
            pass
    
    theta_boots_std = np.array(theta_boots_std) if rank == 0 and len(theta_boots_std) > 0 else None
    theta_boots_std = comm.bcast(theta_boots_std, root=0)
    
    # Bayesian Bootstrap
    if rank == 0:
        print(f"\n  Bayesian Bootstrap ({num_bootstrap})...")
    theta_boots_bayes = []
    np.random.seed(777)
    
    bc_bayes = BundleChoice()
    bc_bayes.load_config(config)
    bc_bayes.data.load_and_scatter(est_data if rank == 0 else None)
    bc_bayes.features.set_oracle(_oracle_fe)
    bc_bayes.subproblems.load()
    
    for b_idx in range(num_bootstrap):
        if rank == 0 and (b_idx + 1) % 20 == 0:
            print(f"    Bayes Boot {b_idx + 1}/{num_bootstrap}")
        
        if rank == 0:
            weights = np.random.exponential(1.0, num_agents)
            weights = weights / weights.mean()
        else:
            weights = None
        weights = comm.bcast(weights, root=0)
        
        try:
            res_b = bc_bayes.row_generation.solve(agent_weights=weights)
            if rank == 0:
                theta_boots_bayes.append(res_b.theta_hat)
        except:
            pass
    
    theta_boots_bayes = np.array(theta_boots_bayes) if rank == 0 and len(theta_boots_bayes) > 0 else None
    theta_boots_bayes = comm.bcast(theta_boots_bayes, root=0)
    
    # Compute results
    if rank == 0:
        n_std = len(theta_boots_std) if theta_boots_std is not None else 0
        n_bayes = len(theta_boots_bayes) if theta_boots_bayes is not None else 0
        
        se_std = np.std(theta_boots_std[:, subset_indices], axis=0, ddof=1) if n_std > 0 else np.full(2, np.nan)
        se_bayes = np.std(theta_boots_bayes[:, subset_indices], axis=0, ddof=1) if n_bayes > 0 else np.full(2, np.nan)
        
        return {
            "theta_0": theta_0[subset_indices],
            "theta_hat": theta_hat[subset_indices],
            "se_std": se_std,
            "se_bayes": se_bayes,
            "n_std": n_std,
            "n_bayes": n_bayes,
        }
    return None


def run_knapsack_fe_experiment(num_agents=200, num_bootstrap=50):
    """Run Knapsack with FE experiment."""
    num_items = 5
    capacity = 3
    num_agent_features = 1
    num_fe = num_items
    num_features = num_agent_features + num_fe  # agent + FE (no quadratic for knapsack)
    sigma = 1.5
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("=" * 70)
        print(f"KNAPSACK WITH FE: {num_agents} agents, {num_items} items, cap={capacity}")
        print("=" * 70)
    
    subset_indices = np.array([0], dtype=np.int64)  # Agent only
    
    # True parameters
    np.random.seed(123)
    theta_0 = np.zeros(num_features)
    theta_0[0] = 3.0   # Agent
    theta_0[1:] = np.abs(np.random.randn(num_fe)) * 1.0 + 0.5  # Positive FE
    
    if rank == 0:
        np.random.seed(42)
        modular_agent = np.abs(np.random.randn(num_agents, num_items, num_agent_features))
        item_weights = np.ones(num_items)
        agent_capacity = np.full(num_agents, capacity, dtype=np.float64)  # Same capacity for all agents
        gen_errors = sigma * np.random.randn(num_agents, num_items)
        np.random.seed(2024)
        est_errors = sigma * np.random.randn(num_agents, num_items)
    else:
        modular_agent = gen_errors = est_errors = item_weights = agent_capacity = None
    
    modular_agent = comm.bcast(modular_agent, root=0)
    gen_errors = comm.bcast(gen_errors, root=0)
    est_errors = comm.bcast(est_errors, root=0)
    item_weights = comm.bcast(item_weights, root=0)
    agent_capacity = comm.bcast(agent_capacity, root=0)
    theta_0 = comm.bcast(theta_0, root=0)
    
    # Bounds
    theta_lbs = [0.0] + [0.0] * num_fe
    theta_ubs = [20.0] + [10.0] * num_fe
    
    config = {
        "dimensions": {"num_agents": num_agents, "num_items": num_items, 
                      "num_features": num_features, "num_simulations": 1},
        "subproblem": {"name": "LinearKnapsack", "settings": {"capacity": capacity}},
        "row_generation": {"max_iters": 200, "tolerance_optimality": 0.001,
                          "theta_lbs": theta_lbs, "theta_ubs": theta_ubs},
    }
    
    # Oracle for knapsack with FE
    def _oracle_fe(agent_idx, bundles, data):
        modular = data["agent_data"]["modular"][agent_idx]
        if bundles.ndim == 1:
            agent_part = (modular[:, 0] * bundles).sum()
            fe_part = bundles.astype(np.float64)
            return np.concatenate([[agent_part], fe_part])
        else:
            agent_parts = (modular[:, 0] * bundles).sum(axis=1)
            fe_parts = bundles.astype(np.float64)
            return np.column_stack([agent_parts, fe_parts])
    
    # Generate observed bundles
    gen_data = {"agent_data": {"modular": modular_agent, "capacity": agent_capacity}, "item_data": {"weights": item_weights}, "errors": gen_errors}
    bc_gen = BundleChoice()
    bc_gen.load_config(config)
    bc_gen.data.load_and_scatter(gen_data if rank == 0 else None)
    bc_gen.features.set_oracle(_oracle_fe)
    bc_gen.subproblems.load()
    obs_bundles = bc_gen.subproblems.init_and_solve(theta_0)
    obs_bundles = comm.bcast(obs_bundles, root=0)
    
    # Validate bundles
    if rank == 0:
        bundle_sizes = obs_bundles.sum(axis=1)
        items_chosen = obs_bundles.sum(axis=0)
        print(f"  Bundles: min={bundle_sizes.min()}, max={bundle_sizes.max()}, mean={bundle_sizes.mean():.1f}")
        print(f"  Items chosen: min={items_chosen.min()}, max={items_chosen.max()}")
    
    # Main estimation
    est_data = {"agent_data": {"modular": modular_agent, "capacity": agent_capacity}, "item_data": {"weights": item_weights}, 
                "errors": est_errors, "obs_bundle": obs_bundles}
    bc = BundleChoice()
    bc.load_config(config)
    bc.data.load_and_scatter(est_data if rank == 0 else None)
    bc.features.set_oracle(_oracle_fe)
    bc.subproblems.load()
    
    result = bc.row_generation.solve()
    theta_hat = result.theta_hat
    
    if rank == 0:
        print(f"  θ_hat (non-FE): {theta_hat[subset_indices]}")
        print(f"  True (non-FE):  {theta_0[subset_indices]}")
    
    # Standard Bootstrap
    if rank == 0:
        print(f"\n  Standard Bootstrap ({num_bootstrap})...")
    theta_boots_std = []
    np.random.seed(999)
    boot_indices_list = [np.random.choice(num_agents, num_agents, replace=True) for _ in range(num_bootstrap)]
    boot_indices_list = comm.bcast(boot_indices_list, root=0)
    
    for b_idx, boot_indices in enumerate(boot_indices_list):
        if rank == 0 and (b_idx + 1) % 20 == 0:
            print(f"    Std Boot {b_idx + 1}/{num_bootstrap}")
        
        boot_data = {
            "agent_data": {"modular": modular_agent[boot_indices], "capacity": agent_capacity[boot_indices]}, 
            "item_data": {"weights": item_weights}, 
            "errors": est_errors[boot_indices], 
            "obs_bundle": obs_bundles[boot_indices]
        }
        bc_b = BundleChoice()
        bc_b.load_config(config)
        bc_b.data.load_and_scatter(boot_data if rank == 0 else None)
        bc_b.features.set_oracle(_oracle_fe)
        bc_b.subproblems.load()
        
        try:
            res_b = bc_b.row_generation.solve()
            if rank == 0:
                theta_boots_std.append(res_b.theta_hat)
        except:
            pass
    
    theta_boots_std = np.array(theta_boots_std) if rank == 0 and len(theta_boots_std) > 0 else None
    theta_boots_std = comm.bcast(theta_boots_std, root=0)
    
    # Bayesian Bootstrap
    if rank == 0:
        print(f"\n  Bayesian Bootstrap ({num_bootstrap})...")
    theta_boots_bayes = []
    np.random.seed(777)
    
    bc_bayes = BundleChoice()
    bc_bayes.load_config(config)
    bc_bayes.data.load_and_scatter(est_data if rank == 0 else None)
    bc_bayes.features.set_oracle(_oracle_fe)
    bc_bayes.subproblems.load()
    
    for b_idx in range(num_bootstrap):
        if rank == 0 and (b_idx + 1) % 20 == 0:
            print(f"    Bayes Boot {b_idx + 1}/{num_bootstrap}")
        
        if rank == 0:
            weights = np.random.exponential(1.0, num_agents)
            weights = weights / weights.mean()
        else:
            weights = None
        weights = comm.bcast(weights, root=0)
        
        try:
            res_b = bc_bayes.row_generation.solve(agent_weights=weights)
            if rank == 0:
                theta_boots_bayes.append(res_b.theta_hat)
        except:
            pass
    
    theta_boots_bayes = np.array(theta_boots_bayes) if rank == 0 and len(theta_boots_bayes) > 0 else None
    theta_boots_bayes = comm.bcast(theta_boots_bayes, root=0)
    
    # Compute results
    if rank == 0:
        n_std = len(theta_boots_std) if theta_boots_std is not None else 0
        n_bayes = len(theta_boots_bayes) if theta_boots_bayes is not None else 0
        
        se_std = np.std(theta_boots_std[:, subset_indices], axis=0, ddof=1) if n_std > 0 else np.full(1, np.nan)
        se_bayes = np.std(theta_boots_bayes[:, subset_indices], axis=0, ddof=1) if n_bayes > 0 else np.full(1, np.nan)
        
        return {
            "theta_0": theta_0[subset_indices],
            "theta_hat": theta_hat[subset_indices],
            "se_std": se_std,
            "se_bayes": se_bayes,
            "n_std": n_std,
            "n_bayes": n_bayes,
        }
    return None


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Run Knapsack FE
    res_knap = run_knapsack_fe_experiment(num_agents=200, num_bootstrap=50)
    comm.Barrier()
    
    # Run Greedy FE
    res_greedy = run_greedy_fe_experiment(num_agents=200, num_bootstrap=50)
    comm.Barrier()
    
    # Summary
    if rank == 0:
        print("\n" + "=" * 80)
        print("SUMMARY: Bayesian Bootstrap with Fixed Effects")
        print("=" * 80)
        
        if res_knap:
            print(f"\n[KNAPSACK]")
            print(f"{'Param':<12} {'True':<10} {'θ_hat':<10} {'SE(Std)':<12} {'SE(Bayes)':<12} {'Bayes/Std':<10}")
            print("-" * 70)
            names = ["Agent"]
            for i in range(len(names)):
                ratio = res_knap["se_bayes"][i] / res_knap["se_std"][i] if res_knap["se_std"][i] > 0 else np.nan
                print(f"{names[i]:<12} {res_knap['theta_0'][i]:>8.4f}  {res_knap['theta_hat'][i]:>8.4f}  "
                      f"{res_knap['se_std'][i]:>10.6f}  {res_knap['se_bayes'][i]:>10.6f}  {ratio:>8.3f}")
            print(f"Success: Std={res_knap['n_std']}/50, Bayes={res_knap['n_bayes']}/50")
        
        if res_greedy:
            print(f"\n[GREEDY]")
            print(f"{'Param':<12} {'True':<10} {'θ_hat':<10} {'SE(Std)':<12} {'SE(Bayes)':<12} {'Bayes/Std':<10}")
            print("-" * 70)
            names = ["Agent", "Quadratic"]
            for i in range(len(names)):
                ratio = res_greedy["se_bayes"][i] / res_greedy["se_std"][i] if res_greedy["se_std"][i] > 0 else np.nan
                print(f"{names[i]:<12} {res_greedy['theta_0'][i]:>8.4f}  {res_greedy['theta_hat'][i]:>8.4f}  "
                      f"{res_greedy['se_std'][i]:>10.6f}  {res_greedy['se_bayes'][i]:>10.6f}  {ratio:>8.3f}")
            print(f"Success: Std={res_greedy['n_std']}/50, Bayes={res_greedy['n_bayes']}/50")


if __name__ == "__main__":
    main()
