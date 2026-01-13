"""
Experiment: Compare SE computation methods WITHOUT fixed effects.
Compares: Full sandwich, Bootstrap, Subsampling.
"""
import argparse
import numpy as np
from mpi4py import MPI
from bundlechoice.core import BundleChoice


def run_greedy_experiment(num_se_simulations=300, num_bootstrap=500, step_size=1e-2, num_agents=200, debug=False):
    """Run greedy experiment WITHOUT fixed effects."""
    num_agent_features = 1
    num_item_features = 2
    num_items = 10
    num_features = num_agent_features + num_item_features + 1
    sigma = 2
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("=" * 70)
        print("GREEDY EXPERIMENT (NO FE)")
        print("=" * 70)
        print(f"  {num_agents} agents, {num_items} items")
        print(f"  Bootstrap: {num_bootstrap}, SE sims: {num_se_simulations}")
    
    all_indices = np.arange(num_features, dtype=np.int64)
    subset_indices = np.array([0, num_features - 1], dtype=np.int64)  # agent + quadratic
    
    theta_0 = np.array([2.0, 1.5, 0.8, 0.1])
    
    # Generate data on rank 0, broadcast
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
        "standard_errors": {"num_simulations": num_se_simulations, "step_size": step_size, "seed": 2024},
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
        print(f"θ_hat: {theta_hat}, True: {theta_0}")
    
    # Method 1: Full sandwich SE
    if rank == 0:
        print("\n[1] Full Sandwich SE...")
    bc.standard_errors.clear_cache()
    se_full = bc.standard_errors.compute(
        theta_hat=theta_hat, num_simulations=num_se_simulations, step_size=step_size,
        beta_indices=all_indices, optimize_for_subset=False,
    )
    
    # Method 2: Subset (partial) sandwich SE
    if rank == 0:
        print("\n[2] Subset (Partial) Sandwich SE...")
    bc.standard_errors.clear_cache()
    se_subset = bc.standard_errors.compute(
        theta_hat=theta_hat, num_simulations=num_se_simulations, step_size=step_size,
        beta_indices=subset_indices, optimize_for_subset=True,
    )
    
    # Method 3: B-inverse SE
    if rank == 0:
        print("\n[3] B-inverse SE...")
    bc.standard_errors.clear_cache()
    try:
        se_binv = bc.standard_errors.compute_B_inverse(
            theta_hat=theta_hat, num_simulations=num_se_simulations,
            beta_indices=subset_indices,
        )
    except AttributeError:
        se_binv = None
    
    # Method 4: Bootstrap
    if rank == 0:
        print(f"\n[4] Bootstrap ({num_bootstrap} resamples)...")
    
    np.random.seed(999 + rank)
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
        cfg_b = dict(config)
        bc_b.load_config(cfg_b)
        bc_b.data.load_and_scatter(boot_data if rank == 0 else None)
        bc_b.oracles.set_features_oracle(_greedy_oracle)
        bc_b.subproblems.load()
        _install_greedy_find_best(bc_b.subproblems.subproblem_instance)
        
        res_b = bc_b.row_generation.solve()
        if rank == 0:
            theta_boots.append(res_b.theta_hat)
            if (b + 1) % 10 == 0:
                print(f"    Bootstrap {b+1}/{num_bootstrap} done")
    
    # Method 5: Subsampling
    subsamp_size = int(num_agents ** 0.7)
    if rank == 0:
        print(f"\n[5] Subsampling ({num_bootstrap} subsamples, size={subsamp_size})...")
    
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
            if (s + 1) % 10 == 0:
                print(f"    Subsample {s+1}/{num_bootstrap} done")
    
    # Compute and report SEs
    if rank == 0 and se_full:
        se_from_full = se_full.se_all[subset_indices]
        se_from_subset = se_subset.se if se_subset else np.full(len(subset_indices), np.nan)
        se_from_binv = se_binv.se if se_binv else np.full(len(subset_indices), np.nan)
        
        theta_boots_arr = np.array(theta_boots) if len(theta_boots) >= 10 else None
        theta_subs_arr = np.array(theta_subs) if len(theta_subs) >= 10 else None
        
        theta_boot_mean = np.mean(theta_boots_arr, axis=0)[subset_indices] if theta_boots_arr is not None else theta_hat[subset_indices]
        se_boot = np.std(theta_boots_arr, axis=0, ddof=1)[subset_indices] if theta_boots_arr is not None else np.full(len(subset_indices), np.nan)
        
        theta_subs_mean = np.mean(theta_subs_arr, axis=0)[subset_indices] if theta_subs_arr is not None else theta_hat[subset_indices]
        se_subsamp = np.sqrt(subsamp_size / num_agents) * np.std(theta_subs_arr, axis=0, ddof=1)[subset_indices] if theta_subs_arr is not None else np.full(len(subset_indices), np.nan)
        
        print("\n" + "=" * 120)
        print("GREEDY RESULTS")
        print("=" * 120)
        print(f"\n{'Param':<12} {'θ_hat':<10} {'SE(Full)':<12} {'SE(Subset)':<12} {'SE(B-inv)':<12} {'θ_boot':<10} {'SE(Boot)':<12} {'θ_subs':<10} {'SE(Subs)':<12}")
        print("-" * 120)
        names = ["Agent", "Quadratic"]
        for i, idx in enumerate(subset_indices):
            print(f"{names[i]:<12} {theta_hat[idx]:>8.4f}  {se_from_full[i]:>10.6f}  {se_from_subset[i]:>10.6f}  {se_from_binv[i]:>10.6f}  {theta_boot_mean[i]:>8.4f}  {se_boot[i]:>10.6f}  {theta_subs_mean[i]:>8.4f}  {se_subsamp[i]:>10.6f}")
        
        return {
            "theta_hat": theta_hat[subset_indices],
            "se_full": se_from_full,
            "se_subset": se_from_subset,
            "se_binv": se_from_binv,
            "theta_boot": theta_boot_mean,
            "se_boot": se_boot,
            "theta_subs": theta_subs_mean,
            "se_subs": se_subsamp,
        }
    return None


def run_knapsack_experiment():
    """Run knapsack experiment WITHOUT fixed effects."""
    num_agent_features = 1
    num_item_features = 2
    num_items = 10
    num_agents = 100
    num_features = num_agent_features + num_item_features
    sigma = 2
    num_se_simulations = 300
    num_bootstrap = 500
    step_size = 1e-2
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("\n" + "=" * 70)
        print("KNAPSACK EXPERIMENT (NO FE)")
        print("=" * 70)
        print(f"  {num_agents} agents, {num_items} items")
        print(f"  Bootstrap: {num_bootstrap}, SE sims: {num_se_simulations}")
    
    all_indices = np.arange(num_features, dtype=np.int64)
    subset_indices = np.arange(num_agent_features, dtype=np.int64)  # Just agent
    
    theta_0 = np.array([2.0, 1.5, 0.8])
    
    # Generate data on rank 0, broadcast
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
    bc_gen.oracles.build_from_data()
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
    bc.oracles.build_from_data()
    bc.subproblems.load()
    
    if rank == 0:
        print("\nMain estimation...")
    result = bc.row_generation.solve()
    theta_hat = result.theta_hat
    
    if rank == 0:
        print(f"θ_hat: {theta_hat}, True: {theta_0}")
    
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
    
    np.random.seed(999 + rank)
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
        cfg_b = dict(config)
        bc_b.load_config(cfg_b)
        bc_b.data.load_and_scatter(boot_data if rank == 0 else None)
        bc_b.oracles.build_from_data()
        bc_b.subproblems.load()
        
        res_b = bc_b.row_generation.solve()
        if rank == 0:
            theta_boots.append(res_b.theta_hat)
            if (b + 1) % 10 == 0:
                print(f"    Bootstrap {b+1}/{num_bootstrap} done")
    
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
            if (s + 1) % 10 == 0:
                print(f"    Subsample {s+1}/{num_bootstrap} done")
    
    # Compute SEs
    if rank == 0 and se_full:
        se_from_full = se_full.se_all[subset_indices]
        
        se_boot = np.std(np.array(theta_boots), axis=0, ddof=1) if len(theta_boots) >= 10 else np.full(num_features, np.nan)
        se_subsamp = np.sqrt(subsamp_size / num_agents) * np.std(np.array(theta_subs), axis=0, ddof=1) if len(theta_subs) >= 10 else np.full(num_features, np.nan)
        
        print("\n" + "=" * 70)
        print("KNAPSACK RESULTS")
        print("=" * 70)
        print(f"\n{'Param':<10} {'θ_hat':<8} {'SE(Full)':<10} {'SE(Boot)':<10} {'SE(Subs)':<10} {'Boot/Full':<10} {'Subs/Full':<10}")
        print("-" * 80)
        for i, idx in enumerate(subset_indices):
            r_boot = se_boot[idx] / se_from_full[i] if se_from_full[i] > 0 else np.nan
            r_subs = se_subsamp[idx] / se_from_full[i] if se_from_full[i] > 0 else np.nan
            print(f"Agent_{idx:<4} {theta_hat[idx]:>6.4f}  {se_from_full[i]:>8.5f}  {se_boot[idx]:>8.5f}  {se_subsamp[idx]:>8.5f}  {r_boot:>8.3f}  {r_subs:>8.3f}")
        
        return {"se_full": se_from_full, "se_boot": se_boot[subset_indices], "se_subs": se_subsamp[subset_indices]}
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
    
    # Parse arguments (only on rank 0, then broadcast)
    if rank == 0:
        parser = argparse.ArgumentParser(description="Compare SE methods: Full Sandwich vs Bootstrap vs Subsampling")
        parser.add_argument("--experiment", type=str, choices=["greedy", "knapsack", "both"], 
                          default="both", help="Which experiment(s) to run (default: both)")
        parser.add_argument("--num-se-sims", type=int, default=300, help="Number of simulations for full sandwich SE")
        parser.add_argument("--num-bootstrap", type=int, default=500, help="Number of bootstrap/subsampling resamples")
        parser.add_argument("--step-size", type=float, default=1e-2, help="Step size for finite differences")
        parser.add_argument("--debug", action="store_true", help="Test multiple hyperparameter settings")
        args = parser.parse_args()
        experiment = args.experiment
        num_se_sims = args.num_se_sims
        num_bootstrap = args.num_bootstrap
        step_size = args.step_size
        debug = args.debug
    else:
        experiment = num_se_sims = num_bootstrap = step_size = debug = None
    experiment = comm.bcast(experiment, root=0)
    num_se_sims = comm.bcast(num_se_sims, root=0)
    num_bootstrap = comm.bcast(num_bootstrap, root=0)
    step_size = comm.bcast(step_size, root=0)
    debug = comm.bcast(debug, root=0)
    
    if rank == 0:
        print("\n" + "#" * 70)
        print("  SE COMPARISON: Full Sandwich vs Bootstrap vs Subsampling")
        print(f"  Experiment(s): {experiment}")
        print(f"  Hyperparameters: SE sims={num_se_sims}, Bootstrap={num_bootstrap}, Step size={step_size}")
        if debug:
            print("  DEBUG MODE: Testing multiple hyperparameter settings")
        print("#" * 70)
    
    results = {}
    
    if debug:
        # Test multiple step sizes and simulation counts
        step_sizes = [1e-4, 1e-3, 1e-2, 0.05]
        num_sims_list = [100, 200, 300, 500]
        
        if rank == 0:
            print("\n" + "=" * 70)
            print("DEBUG MODE: Testing multiple hyperparameter settings")
            print("=" * 70)
        
        for ss in step_sizes:
            for ns in num_sims_list:
                if rank == 0:
                    print(f"\n{'='*70}")
                    print(f"Testing: step_size={ss:.0e}, num_sims={ns}")
                    print(f"{'='*70}")
                
                if experiment in ["greedy", "both"]:
                    res_g = run_greedy_experiment(num_se_simulations=ns, num_bootstrap=200, step_size=ss, num_agents=200)
                    if rank == 0 and res_g:
                        print(f"Greedy (step={ss:.0e}, sims={ns}): Results computed")
                comm.Barrier()
    else:
        # Run experiments with specified hyperparameters
        if experiment in ["greedy", "both"]:
            res_g = run_greedy_experiment(num_se_simulations=num_se_sims, num_bootstrap=num_bootstrap, step_size=step_size, num_agents=200)
            if rank == 0 and res_g:
                results["greedy"] = res_g
            comm.Barrier()
        
        if experiment in ["knapsack", "both"]:
            res_k = run_knapsack_experiment()
            if rank == 0 and res_k:
                results["knapsack"] = res_k
        
        # Summary - results already printed in experiment functions


if __name__ == "__main__":
    main()
