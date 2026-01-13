"""
Bayesian Bootstrap for Standard Errors - NO FIXED EFFECTS.

Instead of resampling agents, we reweight them with Dirichlet weights.
This avoids s_j = 0 problem and keeps all items represented.

Algorithm:
1. Estimate θ_hat with uniform weights
2. For b = 1,...,B:
   a. Draw w_i ~ Exp(1), normalize so mean(w) = 1
   b. Solve weighted estimation: weighted moments, weighted objective
   c. Store θ_b
3. SE = std(θ_1,...,θ_B)
"""
import numpy as np
from mpi4py import MPI
import gurobipy as gp
from gurobipy import GRB
from bundlechoice.core import BundleChoice




def run_experiment(num_agents=200, num_se_simulations=100, num_bootstrap=100):
    """Run greedy experiment with bayesian bootstrap."""
    num_agent_features = 1
    num_item_features = 2
    num_items = 10
    num_features = num_agent_features + num_item_features + 1  # +1 for quadratic
    sigma = 2
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("=" * 70)
        print("BAYESIAN BOOTSTRAP EXPERIMENT (NO FE)")
        print("=" * 70)
        print(f"  {num_agents} agents, {num_items} items, {num_features} features")
        print(f"  Bayesian bootstrap: {num_bootstrap} resamples")
    
    theta_0 = np.array([2.0, 1.5, 0.8, 0.1])
    
    # Generate data on rank 0
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
        "standard_errors": {"num_simulations": num_se_simulations, "step_size": 1e-2, "seed": 2024},
    }
    
    # Oracle
    def _greedy_oracle(agent_idx, bundles, data):
        modular_agent = data["agent_data"]["modular"][agent_idx]
        modular_item = data["item_data"]["modular"]
        if bundles.ndim == 1:
            return np.concatenate([modular_agent.T @ bundles, modular_item.T @ bundles, [-bundles.sum() ** 2]])
        else:
            return np.column_stack([
                (modular_agent[:, :, None] * bundles[:, None, :]).sum(axis=0).T,
                bundles @ modular_item,
                -bundles.sum(axis=1) ** 2
            ])
    
    # Generate observed bundles
    gen_data = {"agent_data": {"modular": modular_agent}, "item_data": {"modular": modular_item}, "errors": gen_errors}
    bc_gen = BundleChoice()
    bc_gen.load_config(config)
    bc_gen.data.load_and_scatter(gen_data if rank == 0 else None)
    bc_gen.oracles.set_features_oracle(_greedy_oracle)
    bc_gen.subproblems.load()
    obs_bundles = bc_gen.subproblems.init_and_solve(theta_0)
    obs_bundles = comm.bcast(obs_bundles, root=0)
    
    # Main estimation (uniform weights)
    if rank == 0:
        print("\n[1] Main estimation (uniform weights)...")
    est_data = {"agent_data": {"modular": modular_agent}, "item_data": {"modular": modular_item}, 
                "errors": est_errors, "obs_bundle": obs_bundles}
    bc = BundleChoice()
    bc.load_config(config)
    bc.data.load_and_scatter(est_data if rank == 0 else None)
    bc.oracles.set_features_oracle(_greedy_oracle)
    bc.subproblems.load()
    
    result = bc.row_generation.solve()
    theta_hat = result.theta_hat
    
    if rank == 0:
        print(f"θ_hat: {theta_hat}")
        print(f"True:  {theta_0}")
    
    # Standard bootstrap for comparison
    if rank == 0:
        print(f"\n[2] Standard Bootstrap ({num_bootstrap} resamples)...")
    theta_boots_std = []
    np.random.seed(999)
    boot_indices_list = [np.random.choice(num_agents, num_agents, replace=True) for _ in range(num_bootstrap)]
    boot_indices_list = comm.bcast(boot_indices_list, root=0)
    
    for b_idx, boot_indices in enumerate(boot_indices_list):
        if rank == 0 and (b_idx + 1) % 25 == 0:
            print(f"    Bootstrap {b_idx + 1}/{num_bootstrap}")
        
        boot_data = {
            "agent_data": {"modular": modular_agent[boot_indices]}, 
            "item_data": {"modular": modular_item}, 
            "errors": est_errors[boot_indices], 
            "obs_bundle": obs_bundles[boot_indices]
        }
        bc_b = BundleChoice()
        bc_b.load_config(config)
        bc_b.data.load_and_scatter(boot_data if rank == 0 else None)
        bc_b.oracles.set_features_oracle(_greedy_oracle)
        bc_b.subproblems.load()
        
        try:
            res_b = bc_b.row_generation.solve()
            if rank == 0:
                theta_boots_std.append(res_b.theta_hat)
        except Exception:
            pass
    
    theta_boots_std = np.array(theta_boots_std) if rank == 0 and len(theta_boots_std) > 0 else None
    theta_boots_std = comm.bcast(theta_boots_std, root=0)
    
    # Bayesian Bootstrap (using proper weighted estimation)
    if rank == 0:
        print(f"\n[3] Bayesian Bootstrap ({num_bootstrap} resamples)...")
    theta_boots_bayes = []
    np.random.seed(777)
    
    # Create a single bc instance for bayesian bootstrap (same data, different weights each time)
    bc_bayes = BundleChoice()
    bc_bayes.load_config(config)
    bc_bayes.data.load_and_scatter(est_data if rank == 0 else None)
    bc_bayes.oracles.set_features_oracle(_greedy_oracle)
    bc_bayes.subproblems.load()
    
    for b_idx in range(num_bootstrap):
        if rank == 0 and (b_idx + 1) % 25 == 0:
            print(f"    Bayesian Bootstrap {b_idx + 1}/{num_bootstrap}")
        
        # Generate Dirichlet weights (Exp(1) normalized to mean=1)
        if rank == 0:
            weights = np.random.exponential(1.0, num_agents)
            weights = weights / weights.mean()  # Normalize so mean = 1
        else:
            weights = None
        weights = comm.bcast(weights, root=0)
        
        try:
            res_b = bc_bayes.row_generation.solve(agent_weights=weights)
            if rank == 0:
                theta_boots_bayes.append(res_b.theta_hat)
        except Exception as e:
            if rank == 0:
                print(f"    Bayesian bootstrap {b_idx} failed: {e}")
    
    theta_boots_bayes = np.array(theta_boots_bayes) if rank == 0 and len(theta_boots_bayes) > 0 else None
    theta_boots_bayes = comm.bcast(theta_boots_bayes, root=0)
    
    # Sandwich SE for comparison
    if rank == 0:
        print(f"\n[4] Sandwich SE...")
    bc.standard_errors.clear_cache()
    se_full = bc.standard_errors.compute(
        theta_hat=theta_hat, num_simulations=num_se_simulations, step_size=1e-2,
        beta_indices=np.arange(num_features, dtype=np.int64), optimize_for_subset=False,
    )
    
    # Results
    if rank == 0:
        se_sandwich = se_full.se_all if se_full else np.full(num_features, np.nan)
        
        if theta_boots_std is not None and len(theta_boots_std) > 0:
            theta_std_mean = np.mean(theta_boots_std, axis=0)
            se_std = np.std(theta_boots_std, axis=0, ddof=1)
        else:
            theta_std_mean = theta_hat
            se_std = np.full(num_features, np.nan)
        
        if theta_boots_bayes is not None and len(theta_boots_bayes) > 0:
            theta_bayes_mean = np.mean(theta_boots_bayes, axis=0)
            se_bayes = np.std(theta_boots_bayes, axis=0, ddof=1)
        else:
            theta_bayes_mean = theta_hat
            se_bayes = np.full(num_features, np.nan)
        
        print("\n" + "=" * 100)
        print("RESULTS")
        print("=" * 100)
        
        print(f"\n{'Param':<12} {'θ_true':<10} {'θ_hat':<10} {'SE(Sand)':<12} {'SE(Std)':<12} {'SE(Bayes)':<12} {'Sand/Std':<10} {'Bayes/Std':<10}")
        print("-" * 100)
        names = ["Agent", "Item1", "Item2", "Quadratic"]
        for i in range(num_features):
            r_sand = se_sandwich[i] / se_std[i] if se_std[i] > 0 else np.nan
            r_bayes = se_bayes[i] / se_std[i] if se_std[i] > 0 else np.nan
            print(f"{names[i]:<12} {theta_0[i]:>8.4f}  {theta_hat[i]:>8.4f}  {se_sandwich[i]:>10.6f}  "
                  f"{se_std[i]:>10.6f}  {se_bayes[i]:>10.6f}  {r_sand:>8.3f}  {r_bayes:>8.3f}")
        
        print("\n" + "=" * 100)
        print("SUMMARY")
        print("=" * 100)
        print(f"Standard Bootstrap mean: {theta_std_mean}")
        print(f"Bayesian Bootstrap mean: {theta_bayes_mean}")
        print(f"Successful standard boots: {len(theta_boots_std)}/{num_bootstrap}")
        print(f"Successful bayesian boots: {len(theta_boots_bayes)}/{num_bootstrap}")


if __name__ == "__main__":
    run_experiment(num_agents=200, num_se_simulations=100, num_bootstrap=100)
