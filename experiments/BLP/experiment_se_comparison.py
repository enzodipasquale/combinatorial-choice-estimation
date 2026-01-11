"""
Experiment: Compare SE computation methods (Full, Subset, B-inverse).

Compares:
1. Full sandwich: A^{-1} B A^{-1} for all parameters
2. Subset sandwich: A^{-1} B A^{-1} for non-FE only  
3. B-inverse: B^{-1} only (no finite differences)
"""
import numpy as np
from mpi4py import MPI
from bundlechoice.core import BundleChoice


def run_greedy_experiment():
    """Run greedy experiment with fixed effects and compare SE methods."""
    num_agent_features = 1
    num_items = 10
    num_agents = 100
    num_features = num_agent_features + num_items + 1
    sigma = 2
    num_se_simulations = 500
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
    
    np.random.seed(42)
    theta_0 = np.concatenate([
        np.array([2.0]),
        np.random.uniform(0, 2, num_items),
        np.array([0.1])
    ])
    
    if rank == 0:
        np.random.seed(42)
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
            "max_iters": 200, "tolerance_optimality": 0.001,
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
    if rank == 0:
        print(f"  Item demands: min={item_demands.min()}, max={item_demands.max()}")
    
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
    
    # Method 1: Full sandwich (all params)
    if rank == 0:
        print("\n" + "#" * 70)
        print("METHOD 1: Full Sandwich (all parameters)")
        print("#" * 70)
    bc.standard_errors.clear_cache()
    se_full = bc.standard_errors.compute(
        theta_hat=theta_hat, num_simulations=num_se_simulations,
        step_size=step_size,
        beta_indices=np.arange(num_features, dtype=np.int64),
        optimize_for_subset=False,
    )
    
    # Method 2: Subset sandwich (non-FE only)
    if rank == 0:
        print("\n" + "#" * 70)
        print("METHOD 2: Subset Sandwich (non-FE only)")
        print("#" * 70)
    bc.standard_errors.clear_cache()
    se_subset = bc.standard_errors.compute(
        theta_hat=theta_hat, num_simulations=num_se_simulations,
        step_size=step_size,
        beta_indices=non_fe_indices, optimize_for_subset=True,
    )
    
    # Method 3: B-inverse (non-FE only, no finite differences)
    if rank == 0:
        print("\n" + "#" * 70)
        print("METHOD 3: B-Inverse (non-FE only, no finite diff)")
        print("#" * 70)
    bc.standard_errors.clear_cache()
    se_binv = bc.standard_errors.compute_B_inverse(
        theta_hat=theta_hat, num_simulations=num_se_simulations,
        beta_indices=non_fe_indices,
    )
    
    # Compare
    if rank == 0:
        print("\n" + "=" * 70)
        print("GREEDY COMPARISON: ALL THREE METHODS")
        print("=" * 70)
        
        if se_full and se_subset and se_binv:
            se_from_full = se_full.se_all[non_fe_indices]
            se_from_sub = se_subset.se
            se_from_binv = se_binv.se
            
            print(f"\n{'Param':<12} {'θ_hat':<10} {'SE(Full)':<12} {'SE(Sub)':<12} {'SE(B-inv)':<12} {'Sub/Full':<10} {'Binv/Full':<10}")
            print("-" * 88)
            for i, idx in enumerate(non_fe_indices):
                name = "Agent" if idx == 0 else "Quadratic"
                r_sub = se_from_sub[i] / se_from_full[i] if se_from_full[i] > 0 else np.nan
                r_binv = se_from_binv[i] / se_from_full[i] if se_from_full[i] > 0 else np.nan
                print(f"{name:<12} {theta_hat[idx]:>8.4f}  {se_from_full[i]:>10.6f}  {se_from_sub[i]:>10.6f}  {se_from_binv[i]:>10.6f}  {r_sub:>8.4f}  {r_binv:>8.4f}")
            
            return {
                "theta": theta_hat, "indices": non_fe_indices,
                "se_full": se_from_full, "se_sub": se_from_sub, "se_binv": se_from_binv
            }
    return None


def run_knapsack_experiment():
    """Run knapsack experiment with fixed effects and compare SE methods."""
    num_agent_features = 1
    num_items = 10
    num_agents = 100
    num_features = num_agent_features + num_items
    sigma = 2
    num_se_simulations = 500
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
    
    np.random.seed(43)
    theta_0 = np.concatenate([np.array([2.0]), np.random.uniform(0.5, 2.5, num_items)])
    
    if rank == 0:
        np.random.seed(43)
        modular_agent = np.abs(np.random.randn(num_agents, num_items, num_agent_features))
        modular_item = np.eye(num_items)
        weights = np.random.randint(1, 4, num_items).astype(float)
        capacity = np.full(num_agents, 0.5 * weights.sum())
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
            "max_iters": 200, "tolerance_optimality": 0.001,
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
    if rank == 0:
        print(f"  Item demands: min={item_demands.min()}, max={item_demands.max()}")
    
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
    
    # Method 1: Full sandwich
    if rank == 0:
        print("\n" + "#" * 70)
        print("METHOD 1: Full Sandwich (all parameters)")
        print("#" * 70)
    bc.standard_errors.clear_cache()
    se_full = bc.standard_errors.compute(
        theta_hat=theta_hat, num_simulations=num_se_simulations,
        step_size=step_size,
        beta_indices=np.arange(num_features, dtype=np.int64),
        optimize_for_subset=False,
    )
    
    # Method 2: Subset sandwich
    if rank == 0:
        print("\n" + "#" * 70)
        print("METHOD 2: Subset Sandwich (non-FE only)")
        print("#" * 70)
    bc.standard_errors.clear_cache()
    se_subset = bc.standard_errors.compute(
        theta_hat=theta_hat, num_simulations=num_se_simulations,
        step_size=step_size,
        beta_indices=non_fe_indices, optimize_for_subset=True,
    )
    
    # Method 3: B-inverse
    if rank == 0:
        print("\n" + "#" * 70)
        print("METHOD 3: B-Inverse (non-FE only, no finite diff)")
        print("#" * 70)
    bc.standard_errors.clear_cache()
    se_binv = bc.standard_errors.compute_B_inverse(
        theta_hat=theta_hat, num_simulations=num_se_simulations,
        beta_indices=non_fe_indices,
    )
    
    if rank == 0:
        print("\n" + "=" * 70)
        print("KNAPSACK COMPARISON: ALL THREE METHODS")
        print("=" * 70)
        
        if se_full and se_subset and se_binv:
            se_from_full = se_full.se_all[non_fe_indices]
            se_from_sub = se_subset.se
            se_from_binv = se_binv.se
            
            print(f"\n{'Param':<12} {'θ_hat':<10} {'SE(Full)':<12} {'SE(Sub)':<12} {'SE(B-inv)':<12} {'Sub/Full':<10} {'Binv/Full':<10}")
            print("-" * 88)
            for i, idx in enumerate(non_fe_indices):
                r_sub = se_from_sub[i] / se_from_full[i] if se_from_full[i] > 0 else np.nan
                r_binv = se_from_binv[i] / se_from_full[i] if se_from_full[i] > 0 else np.nan
                print(f"Agent_{idx:<6} {theta_hat[idx]:>8.4f}  {se_from_full[i]:>10.6f}  {se_from_sub[i]:>10.6f}  {se_from_binv[i]:>10.6f}  {r_sub:>8.4f}  {r_binv:>8.4f}")
            
            return {
                "theta": theta_hat, "indices": non_fe_indices,
                "se_full": se_from_full, "se_sub": se_from_sub, "se_binv": se_from_binv
            }
    return None


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
        print("  SE COMPARISON: Full vs Subset vs B-Inverse")
        print("#" * 70)
    
    res_g = run_greedy_experiment()
    comm.Barrier()
    res_k = run_knapsack_experiment()
    
    if rank == 0:
        print("\n" + "#" * 70)
        print("FINAL SUMMARY")
        print("#" * 70)
        
        if res_g:
            print("\nGreedy:")
            print(f"  Sub/Full ratios: {res_g['se_sub'] / res_g['se_full']}")
            print(f"  Binv/Full ratios: {res_g['se_binv'] / res_g['se_full']}")
        
        if res_k:
            print("\nKnapsack:")
            print(f"  Sub/Full ratios: {res_k['se_sub'] / res_k['se_full']}")
            print(f"  Binv/Full ratios: {res_k['se_binv'] / res_k['se_full']}")
        
        print("\nInterpretation:")
        print("  Ratio ≈ 1.0 → method matches full sandwich")
        print("  Ratio < 1.0 → underestimates SE (anti-conservative)")
        print("  Ratio > 1.0 → overestimates SE (conservative)")


if __name__ == "__main__":
    main()
