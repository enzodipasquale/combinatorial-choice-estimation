"""
Test Bayesian Bootstrap vs Standard Bootstrap.
Compare on Greedy and Knapsack, with 2 sample sizes each.
"""
import numpy as np
from mpi4py import MPI
from bundlechoice.core import BundleChoice


def run_experiment(subproblem_name, num_agents, num_bootstrap=50):
    """Run bootstrap comparison for a given subproblem and sample size."""
    num_items = 10
    num_agent_features = 1
    num_item_features = 2
    num_features = num_agent_features + num_item_features + 1
    sigma = 2
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    theta_0 = np.array([2.0, 1.5, 0.8, 0.1])
    
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
        "subproblem": {"name": subproblem_name, "settings": {"TimeLimit": 5}},
        "row_generation": {"max_iters": 200, "tolerance_optimality": 0.001,
                          "theta_lbs": [0] * num_features, "theta_ubs": 100},
    }
    
    # Oracle
    def _oracle(agent_idx, bundles, data):
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
    bc_gen.oracles.set_features_oracle(_oracle)
    bc_gen.subproblems.load()
    obs_bundles = bc_gen.subproblems.init_and_solve(theta_0)
    obs_bundles = comm.bcast(obs_bundles, root=0)
    
    # Main estimation
    est_data = {"agent_data": {"modular": modular_agent}, "item_data": {"modular": modular_item}, 
                "errors": est_errors, "obs_bundle": obs_bundles}
    bc = BundleChoice()
    bc.load_config(config)
    bc.data.load_and_scatter(est_data if rank == 0 else None)
    bc.oracles.set_features_oracle(_oracle)
    bc.subproblems.load()
    
    result = bc.row_generation.solve()
    theta_hat = result.theta_hat
    
    # Standard Bootstrap
    theta_boots_std = []
    np.random.seed(999)
    boot_indices_list = [np.random.choice(num_agents, num_agents, replace=True) for _ in range(num_bootstrap)]
    boot_indices_list = comm.bcast(boot_indices_list, root=0)
    
    for boot_indices in boot_indices_list:
        boot_data = {
            "agent_data": {"modular": modular_agent[boot_indices]}, 
            "item_data": {"modular": modular_item}, 
            "errors": est_errors[boot_indices], 
            "obs_bundle": obs_bundles[boot_indices]
        }
        bc_b = BundleChoice()
        bc_b.load_config(config)
        bc_b.data.load_and_scatter(boot_data if rank == 0 else None)
        bc_b.oracles.set_features_oracle(_oracle)
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
    theta_boots_bayes = []
    np.random.seed(777)
    
    bc_bayes = BundleChoice()
    bc_bayes.load_config(config)
    bc_bayes.data.load_and_scatter(est_data if rank == 0 else None)
    bc_bayes.oracles.set_features_oracle(_oracle)
    bc_bayes.subproblems.load()
    
    for _ in range(num_bootstrap):
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
    
    # Compute SEs
    if rank == 0:
        se_std = np.std(theta_boots_std, axis=0, ddof=1) if theta_boots_std is not None else np.full(num_features, np.nan)
        se_bayes = np.std(theta_boots_bayes, axis=0, ddof=1) if theta_boots_bayes is not None else np.full(num_features, np.nan)
        n_std = len(theta_boots_std) if theta_boots_std is not None else 0
        n_bayes = len(theta_boots_bayes) if theta_boots_bayes is not None else 0
        
        return {
            "theta_hat": theta_hat,
            "se_std": se_std,
            "se_bayes": se_bayes,
            "n_std": n_std,
            "n_bayes": n_bayes,
        }
    return None


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    configs = [
        ("Greedy", 100),
        ("Greedy", 200),
    ]
    
    results = {}
    
    for subproblem, num_agents in configs:
        if rank == 0:
            print(f"\n{'='*70}")
            print(f"{subproblem}, N={num_agents}")
            print(f"{'='*70}")
        
        res = run_experiment(subproblem, num_agents, num_bootstrap=50)
        
        if rank == 0 and res:
            results[(subproblem, num_agents)] = res
    
    # Summary table
    if rank == 0:
        print("\n" + "=" * 100)
        print("SUMMARY: Standard Bootstrap vs Bayesian Bootstrap")
        print("=" * 100)
        print(f"\n{'Config':<25} {'Param':<12} {'θ_hat':<10} {'SE(Std)':<12} {'SE(Bayes)':<12} {'Bayes/Std':<10}")
        print("-" * 100)
        
        param_names = ["Agent", "Quadratic"]
        param_idx = [0, 3]  # Agent and Quadratic
        
        for (subproblem, num_agents), res in results.items():
            config_name = f"{subproblem[:6]}, N={num_agents}"
            for i, idx in enumerate(param_idx):
                ratio = res["se_bayes"][idx] / res["se_std"][idx] if res["se_std"][idx] > 0 else np.nan
                print(f"{config_name:<25} {param_names[i]:<12} {res['theta_hat'][idx]:>8.4f}  "
                      f"{res['se_std'][idx]:>10.6f}  {res['se_bayes'][idx]:>10.6f}  {ratio:>8.3f}")
            print()
        
        print("-" * 100)
        print("Success rates:")
        for (subproblem, num_agents), res in results.items():
            print(f"  {subproblem}, N={num_agents}: Std={res['n_std']}/50, Bayes={res['n_bayes']}/50")


if __name__ == "__main__":
    main()
