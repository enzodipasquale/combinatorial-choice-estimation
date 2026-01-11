"""
Experiment: Compare SE computation methods (Full vs Subset/Heuristic).

With many items/FE, computing and inverting the full matrices is expensive.
This tests whether restricting to non-FE indices gives similar results.
"""
import numpy as np
import time
from mpi4py import MPI
from bundlechoice.core import BundleChoice


def run_greedy_experiment():
    """Run greedy experiment with fixed effects and compare SE methods."""
    num_agent_features = 1
    num_items = 50
    num_agents = 100
    num_features = num_agent_features + num_items + 1  # agent + FE + quadratic
    sigma = 2
    num_se_simulations = 100
    step_size = 1e-2
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("=" * 70)
        print("GREEDY EXPERIMENT WITH FIXED EFFECTS")
        print("=" * 70)
        print(f"  {num_agents} agents, {num_items} items, {num_features} features")
        print(f"  SE simulations: {num_se_simulations}, step_size: {step_size}")
    
    non_fe_indices = np.array([0, num_features - 1], dtype=np.int64)
    
    # Retry loop to ensure good item demand distribution
    max_retries = 10
    for attempt in range(max_retries):
        seed = 42 + attempt
        np.random.seed(seed)
        theta_0 = np.concatenate([
            np.array([2.0]),
            np.random.uniform(-0.5, 1.5, num_items),  # Varied FE
            np.array([0.05])  # Smaller quadratic to allow more items
        ])
        
        if rank == 0:
            np.random.seed(seed)
            modular_agent = np.abs(np.random.randn(num_agents, num_items, num_agent_features))
            modular_item = np.eye(num_items)
            errors = sigma * np.random.randn(num_agents, num_items)
            input_data = {
                "agent_data": {"modular": modular_agent},
                "item_data": {"modular": modular_item},
                "errors": errors,
            }
        else:
            input_data = None
        
        config = {
            "dimensions": {"num_agents": num_agents, "num_items": num_items, 
                          "num_features": num_features, "num_simulations": 1},
            "subproblem": {"name": "Greedy"},
            "row_generation": {
                "max_iters": 300, "tolerance_optimality": 0.001,
                "theta_lbs": [0] * num_agent_features + [-1e10] * num_items + [0],
                "theta_ubs": 1e10,
            },
            "standard_errors": {"num_simulations": num_se_simulations, "step_size": step_size, "seed": 2024},
        }
        
        bc_gen = BundleChoice()
        bc_gen.load_config(config)
        bc_gen.data.load_and_scatter(input_data)
        bc_gen.features.set_oracle(_greedy_oracle)
        bc_gen.subproblems.load()
        _install_greedy_find_best(bc_gen.subproblems.subproblem_instance)
        obs_bundles = bc_gen.subproblems.init_and_solve(theta_0)
        
        # Check item demands
        item_demands = obs_bundles.sum(axis=0) if rank == 0 else None
        item_demands = comm.bcast(item_demands, root=0)
        never_chosen = np.where(item_demands == 0)[0]
        always_chosen = np.where(item_demands == num_agents)[0]
        
        if rank == 0:
            print(f"  [Attempt {attempt+1}] Item demands: min={item_demands.min()}, max={item_demands.max()}")
        
        if len(never_chosen) == 0 and len(always_chosen) == 0:
            if rank == 0:
                print(f"  ✓ All items chosen by some (not all) agents")
            break
        else:
            if rank == 0:
                if len(never_chosen) > 0:
                    print(f"    ⚠ {len(never_chosen)} never chosen")
                if len(always_chosen) > 0:
                    print(f"    ⚠ {len(always_chosen)} always chosen")
    
    # Prepare estimation data
    if rank == 0:
        np.random.seed(2024)
        est_data = {
            "agent_data": {"modular": modular_agent},
            "item_data": {"modular": modular_item},
            "errors": sigma * np.random.randn(num_agents, num_items),
            "obs_bundle": obs_bundles,
        }
    else:
        est_data = None
    
    bc = BundleChoice()
    bc.load_config(config)
    bc.data.load_and_scatter(est_data)
    bc.features.set_oracle(_greedy_oracle)
    bc.subproblems.load()
    _install_greedy_find_best(bc.subproblems.subproblem_instance)
    
    if rank == 0:
        print("\nRunning estimation...")
    result = bc.row_generation.solve()
    theta_hat = result.theta_hat
    
    if rank == 0:
        print(f"θ_hat: agent={theta_hat[0]:.4f}, quad={theta_hat[-1]:.6f}")
        print(f"True:  agent={theta_0[0]:.4f}, quad={theta_0[-1]:.6f}")
        # Check for FE hitting bounds
        fe_params = theta_hat[num_agent_features:-1]
        at_bounds = np.sum(np.abs(fe_params) > 1e9)
        if at_bounds > 0:
            print(f"  ⚠ {at_bounds} FE params at bounds!")
    
    # SE Method 1: Full matrices
    if rank == 0:
        print("\n[Method 1] Full SE computation...")
    bc.standard_errors.clear_cache()
    se_full = bc.standard_errors.compute(
        theta_hat=theta_hat, num_simulations=num_se_simulations,
        step_size=step_size,
        beta_indices=np.arange(num_features, dtype=np.int64),
        optimize_for_subset=False,
    )
    
    # SE Method 2: Subset matrices
    if rank == 0:
        print("\n[Method 2] Subset SE computation...")
    bc.standard_errors.clear_cache()
    se_subset = bc.standard_errors.compute(
        theta_hat=theta_hat, num_simulations=num_se_simulations,
        step_size=step_size,
        beta_indices=non_fe_indices, optimize_for_subset=True,
    )
    
    # Compare
    if rank == 0:
        print("\n" + "=" * 70)
        print("GREEDY COMPARISON")
        print("=" * 70)
        
        if se_full and se_subset:
            se_from_full = se_full.se_all[non_fe_indices]
            se_from_sub = se_subset.se
            print(f"\n{'Param':<12} {'θ_hat':<10} {'SE(Full)':<12} {'SE(Sub)':<12} {'Ratio':<8}")
            print("-" * 54)
            for i, idx in enumerate(non_fe_indices):
                name = "Agent" if idx == 0 else "Quadratic"
                ratio = se_from_sub[i] / se_from_full[i] if se_from_full[i] > 0 else np.nan
                print(f"{name:<12} {theta_hat[idx]:>8.4f}  {se_from_full[i]:>10.6f}  {se_from_sub[i]:>10.6f}  {ratio:>6.4f}")
            return se_from_full, se_from_sub, theta_hat, non_fe_indices
        elif se_subset:
            print("  Full SE failed, subset SE succeeded")
            for i, idx in enumerate(non_fe_indices):
                name = "Agent" if idx == 0 else "Quadratic"
                print(f"  {name}: θ={theta_hat[idx]:.4f}, SE={se_subset.se[i]:.6f}")
            return None, se_subset.se, theta_hat, non_fe_indices
    return None, None, None, None


def run_knapsack_experiment():
    """Run knapsack experiment with fixed effects and compare SE methods."""
    num_agent_features = 1
    num_items = 50
    num_agents = 100
    num_features = num_agent_features + num_items
    sigma = 2
    num_se_simulations = 100
    step_size = 1e-2
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("\n" + "=" * 70)
        print("KNAPSACK EXPERIMENT WITH FIXED EFFECTS")
        print("=" * 70)
        print(f"  {num_agents} agents, {num_items} items, {num_features} features")
        print(f"  SE simulations: {num_se_simulations}, step_size: {step_size}")
    
    non_fe_indices = np.arange(num_agent_features, dtype=np.int64)
    
    # Retry loop for good item distribution
    max_retries = 10
    for attempt in range(max_retries):
        seed = 43 + attempt
        np.random.seed(seed)
        theta_0 = np.concatenate([np.array([2.0]), np.random.uniform(0, 2, num_items)])
        
        if rank == 0:
            np.random.seed(seed)
            modular_agent = np.abs(np.random.randn(num_agents, num_items, num_agent_features))
            modular_item = np.eye(num_items)
            weights = np.random.randint(1, 4, num_items).astype(float)  # Lighter weights
            capacity = np.full(num_agents, 0.4 * weights.sum())  # Tighter capacity
            errors = sigma * np.random.randn(num_agents, num_items)
            input_data = {
                "agent_data": {"modular": modular_agent, "capacity": capacity},
                "item_data": {"modular": modular_item, "weights": weights},
                "errors": errors,
            }
        else:
            input_data = None
        
        config = {
            "dimensions": {"num_agents": num_agents, "num_items": num_items,
                          "num_features": num_features, "num_simulations": 1},
            "subproblem": {"name": "LinearKnapsack", "settings": {"TimeLimit": 5}},
            "row_generation": {
                "max_iters": 300, "tolerance_optimality": 0.001,
                "theta_lbs": [0] * num_agent_features + [-1e10] * num_items,
                "theta_ubs": 1e10,
            },
            "standard_errors": {"num_simulations": num_se_simulations, "step_size": step_size, "seed": 2024},
        }
        
        bc_gen = BundleChoice()
        bc_gen.load_config(config)
        bc_gen.data.load_and_scatter(input_data)
        bc_gen.features.build_from_data()
        bc_gen.subproblems.load()
        obs_bundles = bc_gen.subproblems.init_and_solve(theta_0)
        
        item_demands = obs_bundles.sum(axis=0) if rank == 0 else None
        item_demands = comm.bcast(item_demands, root=0)
        never_chosen = np.where(item_demands == 0)[0]
        always_chosen = np.where(item_demands == num_agents)[0]
        
        if rank == 0:
            print(f"  [Attempt {attempt+1}] Item demands: min={item_demands.min()}, max={item_demands.max()}")
        
        if len(never_chosen) == 0 and len(always_chosen) == 0:
            if rank == 0:
                print(f"  ✓ All items chosen by some (not all) agents")
            break
        else:
            if rank == 0:
                if len(never_chosen) > 0:
                    print(f"    ⚠ {len(never_chosen)} never chosen")
                if len(always_chosen) > 0:
                    print(f"    ⚠ {len(always_chosen)} always chosen")
    
    if rank == 0:
        np.random.seed(2024)
        est_data = {
            "agent_data": {"modular": modular_agent, "capacity": capacity},
            "item_data": {"modular": modular_item, "weights": weights},
            "errors": sigma * np.random.randn(num_agents, num_items),
            "obs_bundle": obs_bundles,
        }
    else:
        est_data = None
    
    bc = BundleChoice()
    bc.load_config(config)
    bc.data.load_and_scatter(est_data)
    bc.features.build_from_data()
    bc.subproblems.load()
    
    if rank == 0:
        print("\nRunning estimation...")
    result = bc.row_generation.solve()
    theta_hat = result.theta_hat
    
    if rank == 0:
        print(f"θ_hat agent: {theta_hat[0]:.4f}, True: {theta_0[0]:.4f}")
        fe_params = theta_hat[num_agent_features:]
        at_bounds = np.sum(np.abs(fe_params) > 1e9)
        if at_bounds > 0:
            print(f"  ⚠ {at_bounds} FE params at bounds!")
    
    if rank == 0:
        print("\n[Method 1] Full SE computation...")
    bc.standard_errors.clear_cache()
    se_full = bc.standard_errors.compute(
        theta_hat=theta_hat, num_simulations=num_se_simulations,
        step_size=step_size,
        beta_indices=np.arange(num_features, dtype=np.int64),
        optimize_for_subset=False,
    )
    
    if rank == 0:
        print("\n[Method 2] Subset SE computation...")
    bc.standard_errors.clear_cache()
    se_subset = bc.standard_errors.compute(
        theta_hat=theta_hat, num_simulations=num_se_simulations,
        step_size=step_size,
        beta_indices=non_fe_indices, optimize_for_subset=True,
    )
    
    if rank == 0:
        print("\n" + "=" * 70)
        print("KNAPSACK COMPARISON")
        print("=" * 70)
        
        if se_full and se_subset:
            se_from_full = se_full.se_all[non_fe_indices]
            se_from_sub = se_subset.se
            print(f"\n{'Param':<12} {'θ_hat':<10} {'SE(Full)':<12} {'SE(Sub)':<12} {'Ratio':<8}")
            print("-" * 54)
            for i, idx in enumerate(non_fe_indices):
                ratio = se_from_sub[i] / se_from_full[i] if se_from_full[i] > 0 else np.nan
                print(f"Agent_{idx:<6} {theta_hat[idx]:>8.4f}  {se_from_full[i]:>10.6f}  {se_from_sub[i]:>10.6f}  {ratio:>6.4f}")
            return se_from_full, se_from_sub, theta_hat, non_fe_indices
        elif se_subset:
            print("  Full SE failed, subset SE succeeded")
            for i, idx in enumerate(non_fe_indices):
                print(f"  Agent_{idx}: θ={theta_hat[idx]:.4f}, SE={se_subset.se[i]:.6f}")
            return None, se_subset.se, theta_hat, non_fe_indices
    return None, None, None, None


def _greedy_oracle(agent_idx, bundles, data):
    """Feature oracle for greedy with FE."""
    modular_agent = data["agent_data"]["modular"][agent_idx]
    modular_item = data["item_data"]["modular"]
    
    if bundles.ndim == 1:
        agent_feat = modular_agent.T @ bundles
        item_feat = modular_item.T @ bundles
        quad = -bundles.sum() ** 2
        return np.concatenate([agent_feat, item_feat, [quad]])
    else:
        agent_feat = modular_agent.T @ bundles
        item_feat = modular_item.T @ bundles
        quad = -np.sum(bundles, axis=0, keepdims=True) ** 2
        return np.vstack([agent_feat, item_feat, quad])


def _install_greedy_find_best(solver):
    """Install find_best_item for greedy with FE."""
    def find_best_item(local_id, base_bundle, items_left, theta, error_j):
        modular_agent = solver.local_data["agent_data"]["modular"][local_id]
        modular_item = solver.local_data["item_data"]["modular"]
        num_agent_f = modular_agent.shape[1]
        num_items = modular_item.shape[0]
        
        theta_agent = theta[:num_agent_f]
        theta_item = theta[num_agent_f:num_agent_f + num_items]
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
        print("\n" + "#" * 70)
        print("  SE COMPARISON: Full vs Subset (Heuristic)")
        print("#" * 70)
    
    se_g_full, se_g_sub, theta_g, idx_g = run_greedy_experiment()
    comm.Barrier()
    se_k_full, se_k_sub, theta_k, idx_k = run_knapsack_experiment()
    
    if rank == 0:
        print("\n" + "#" * 70)
        print("FINAL SUMMARY")
        print("#" * 70)
        
        if se_g_full is not None and se_g_sub is not None:
            ratios_g = se_g_sub / se_g_full
            print(f"\nGreedy SE ratios (Subset/Full): {ratios_g}")
            print(f"  Mean ratio: {np.mean(ratios_g):.4f}")
        
        if se_k_full is not None and se_k_sub is not None:
            ratios_k = se_k_sub / se_k_full
            print(f"\nKnapsack SE ratios (Subset/Full): {ratios_k}")
            print(f"  Mean ratio: {np.mean(ratios_k):.4f}")
        
        print("\nInterpretation:")
        print("  Ratio ≈ 1.0 → subset heuristic matches full computation")
        print("  Ratio < 1.0 → subset underestimates SE (anti-conservative)")
        print("  Ratio > 1.0 → subset overestimates SE (conservative)")


if __name__ == "__main__":
    main()
