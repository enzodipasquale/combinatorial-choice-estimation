"""
Test the refactored SE module: Greedy without FE (all methods) and with FE (Bayesian only).
"""
import numpy as np
from mpi4py import MPI
from bundlechoice.core import BundleChoice


def run_greedy_no_fe(num_agents=150, num_bootstrap=30):
    """Test all SE methods on Greedy without FE."""
    num_items = 10
    num_features = 2  # Agent linear + Quadratic
    sigma = 1.5
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("\n" + "=" * 80)
        print("GREEDY WITHOUT FE - ALL SE METHODS")
        print("=" * 80)
    
    # True parameters
    theta_0 = np.array([3.0, 0.3])  # Agent, Quadratic
    
    # Generate data
    if rank == 0:
        np.random.seed(42)
        modular_agent = np.abs(np.random.randn(num_agents, num_items, 1))
        gen_errors = sigma * np.random.randn(num_agents, num_items)
        np.random.seed(2024)
        est_errors = sigma * np.random.randn(num_agents, num_items)
    else:
        modular_agent = gen_errors = est_errors = None
    
    modular_agent = comm.bcast(modular_agent, root=0)
    gen_errors = comm.bcast(gen_errors, root=0)
    est_errors = comm.bcast(est_errors, root=0)
    
    config = {
        "dimensions": {"num_agents": num_agents, "num_items": num_items, 
                      "num_features": num_features, "num_simulations": 1},
        "subproblem": {"name": "Greedy"},
        "row_generation": {"max_iters": 200, "tolerance_optimality": 0.001},
        "standard_errors": {"num_simulations": 10, "step_size": 1e-4, "error_sigma": sigma},
    }
    
    def _oracle(agent_idx, bundles, data):
        modular = data["agent_data"]["modular"][agent_idx]
        if bundles.ndim == 1:
            bundle_size = bundles.sum()
            return np.array([(modular[:, 0] * bundles).sum(), -bundle_size ** 2])
        else:
            sizes = bundles.sum(axis=1)
            return np.column_stack([(modular[:, 0] * bundles).sum(axis=1), -sizes ** 2])
    
    # Generate observed bundles
    gen_data = {"agent_data": {"modular": modular_agent}, "item_data": {}, "errors": gen_errors}
    bc_gen = BundleChoice()
    bc_gen.load_config(config)
    bc_gen.data.load_and_scatter(gen_data if rank == 0 else None)
    bc_gen.features.set_oracle(_oracle)
    bc_gen.subproblems.load()
    obs_bundles = bc_gen.subproblems.init_and_solve(theta_0)
    obs_bundles = comm.bcast(obs_bundles, root=0)
    
    if rank == 0:
        sizes = obs_bundles.sum(axis=1)
        print(f"  Bundles: min={sizes.min()}, max={sizes.max()}, mean={sizes.mean():.1f}")
    
    # Main estimation
    est_data = {"agent_data": {"modular": modular_agent}, "item_data": {}, 
                "errors": est_errors, "obs_bundle": obs_bundles}
    bc = BundleChoice()
    bc.load_config(config)
    bc.data.load_and_scatter(est_data if rank == 0 else None)
    bc.features.set_oracle(_oracle)
    bc.subproblems.load()
    
    result = bc.row_generation.solve()
    theta_hat = result.theta_hat
    
    if rank == 0:
        print(f"\n  θ_hat: {theta_hat}")
        print(f"  True:  {theta_0}")
    
    results = {"theta_0": theta_0, "theta_hat": theta_hat}
    
    # 1. B-inverse SE
    if rank == 0:
        print("\n--- Testing B-inverse SE ---")
    se_binv = bc.standard_errors.compute_B_inverse(theta_hat, error_sigma=sigma)
    if rank == 0 and se_binv:
        results["se_binv"] = se_binv.se
    
    # 2. Standard Bootstrap SE
    if rank == 0:
        print("\n--- Testing Standard Bootstrap SE ---")
    
    def solve_fn(data_dict):
        bc_b = BundleChoice()
        bc_b.load_config(config)
        bc_b.data.load_and_scatter(data_dict if rank == 0 else None)
        bc_b.features.set_oracle(_oracle)
        bc_b.subproblems.load()
        try:
            res = bc_b.row_generation.solve()
            return res.theta_hat if rank == 0 else None
        except:
            return None
    
    se_boot = bc.standard_errors.compute_bootstrap(theta_hat, solve_fn, num_bootstrap=num_bootstrap, seed=999)
    if rank == 0 and se_boot:
        results["se_boot"] = se_boot.se
    
    # 3. Bayesian Bootstrap SE
    if rank == 0:
        print("\n--- Testing Bayesian Bootstrap SE ---")
    se_bayes = bc.standard_errors.compute_bayesian_bootstrap(theta_hat, bc.row_generation, num_bootstrap=num_bootstrap, seed=777)
    if rank == 0 and se_bayes:
        results["se_bayes"] = se_bayes.se
    
    return results if rank == 0 else None


def run_greedy_with_fe(num_agents=150, num_bootstrap=30):
    """Test Bayesian bootstrap on Greedy with FE."""
    num_items = 8
    num_fe = num_items
    num_features = 2 + num_fe  # Agent + Quadratic + FE
    sigma = 1.5
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("\n" + "=" * 80)
        print("GREEDY WITH FE - BAYESIAN BOOTSTRAP ONLY")
        print("=" * 80)
    
    # True parameters
    np.random.seed(123)
    theta_0 = np.zeros(num_features)
    theta_0[0] = 3.0   # Agent
    theta_0[1] = 0.2   # Quadratic
    theta_0[2:] = np.abs(np.random.randn(num_fe)) * 1.0 + 0.5  # Positive FE
    
    beta_indices = np.array([0, 1], dtype=np.int64)  # Only report non-FE params
    
    # Generate data
    if rank == 0:
        np.random.seed(42)
        modular_agent = np.abs(np.random.randn(num_agents, num_items, 1))
        gen_errors = sigma * np.random.randn(num_agents, num_items)
        np.random.seed(2024)
        est_errors = sigma * np.random.randn(num_agents, num_items)
    else:
        modular_agent = gen_errors = est_errors = None
    
    modular_agent = comm.bcast(modular_agent, root=0)
    gen_errors = comm.bcast(gen_errors, root=0)
    est_errors = comm.bcast(est_errors, root=0)
    theta_0 = comm.bcast(theta_0, root=0)
    
    theta_lbs = [0.0, 0.0] + [0.0] * num_fe
    theta_ubs = [20.0, 5.0] + [10.0] * num_fe
    
    config = {
        "dimensions": {"num_agents": num_agents, "num_items": num_items, 
                      "num_features": num_features, "num_simulations": 1},
        "subproblem": {"name": "Greedy"},
        "row_generation": {"max_iters": 200, "tolerance_optimality": 0.001,
                          "theta_lbs": theta_lbs, "theta_ubs": theta_ubs},
        "standard_errors": {"num_simulations": 10, "step_size": 1e-4, "error_sigma": sigma},
    }
    
    def _oracle_fe(agent_idx, bundles, data):
        modular = data["agent_data"]["modular"][agent_idx]
        if bundles.ndim == 1:
            bundle_size = bundles.sum()
            agent_part = (modular[:, 0] * bundles).sum()
            quad_part = -bundle_size ** 2
            fe_part = bundles.astype(np.float64)
            return np.concatenate([[agent_part, quad_part], fe_part])
        else:
            sizes = bundles.sum(axis=1)
            agent_parts = (modular[:, 0] * bundles).sum(axis=1)
            quad_parts = -sizes ** 2
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
    
    if rank == 0:
        sizes = obs_bundles.sum(axis=1)
        items_chosen = obs_bundles.sum(axis=0)
        print(f"  Bundles: min={sizes.min()}, max={sizes.max()}, mean={sizes.mean():.1f}")
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
        print(f"\n  θ_hat (non-FE): {theta_hat[beta_indices]}")
        print(f"  True (non-FE):  {theta_0[beta_indices]}")
    
    results = {"theta_0": theta_0[beta_indices], "theta_hat": theta_hat[beta_indices]}
    
    # Bayesian Bootstrap SE
    if rank == 0:
        print("\n--- Testing Bayesian Bootstrap SE (FE case) ---")
    se_bayes = bc.standard_errors.compute_bayesian_bootstrap(
        theta_hat, bc.row_generation, num_bootstrap=num_bootstrap, 
        beta_indices=beta_indices, seed=777
    )
    if rank == 0 and se_bayes:
        results["se_bayes"] = se_bayes.se
    
    return results if rank == 0 else None


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Test 1: Greedy without FE
    res_no_fe = run_greedy_no_fe(num_agents=150, num_bootstrap=30)
    comm.Barrier()
    
    # Test 2: Greedy with FE
    res_fe = run_greedy_with_fe(num_agents=150, num_bootstrap=30)
    comm.Barrier()
    
    # Summary
    if rank == 0:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        
        if res_no_fe:
            print("\n[GREEDY WITHOUT FE]")
            print(f"{'Param':<12} {'True':<10} {'θ_hat':<10} {'B-inv':<12} {'Boot':<12} {'Bayes':<12}")
            print("-" * 80)
            names = ["Agent", "Quadratic"]
            for i in range(2):
                se_bi = res_no_fe.get("se_binv", [np.nan, np.nan])[i]
                se_bt = res_no_fe.get("se_boot", [np.nan, np.nan])[i]
                se_by = res_no_fe.get("se_bayes", [np.nan, np.nan])[i]
                print(f"{names[i]:<12} {res_no_fe['theta_0'][i]:>8.4f}  {res_no_fe['theta_hat'][i]:>8.4f}  "
                      f"{se_bi:>10.6f}  {se_bt:>10.6f}  {se_by:>10.6f}")
        
        if res_fe:
            print("\n[GREEDY WITH FE]")
            print(f"{'Param':<12} {'True':<10} {'θ_hat':<10} {'Bayes SE':<12}")
            print("-" * 50)
            names = ["Agent", "Quadratic"]
            for i in range(2):
                se_by = res_fe.get("se_bayes", [np.nan, np.nan])[i]
                print(f"{names[i]:<12} {res_fe['theta_0'][i]:>8.4f}  {res_fe['theta_hat'][i]:>8.4f}  {se_by:>10.6f}")


if __name__ == "__main__":
    main()
