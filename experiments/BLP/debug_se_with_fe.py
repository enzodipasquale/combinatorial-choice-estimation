"""
Debug SE with fixed effects - Greedy only.
Key: Use bounded FE with tight bounds to prevent bootstrap divergence.
"""
import numpy as np
from mpi4py import MPI
from bundlechoice.core import BundleChoice


def main():
    num_agents = 500
    num_items = 20
    num_agent_features = 1
    num_fe = num_items
    num_features = num_agent_features + 1 + num_fe  # agent + quad + FE
    sigma = 1.5
    num_se_simulations = 100
    num_bootstrap = 50
    step_size = 1e-2
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("=" * 80)
        print("DEBUG SE WITH FIXED EFFECTS - GREEDY")
        print("=" * 80)
        print(f"Agents: {num_agents}, Items: {num_items}")
        print(f"Features: {num_agent_features} agent + 1 quad + {num_fe} FE = {num_features}")
    
    subset_indices = np.array([0, 1], dtype=np.int64)  # Agent, Quadratic
    
    # True parameters - moderate values, FE centered around 0
    np.random.seed(123)
    theta_0 = np.zeros(num_features)
    theta_0[0] = 3.0   # Agent coefficient
    theta_0[1] = 0.2   # Quadratic (small but positive)
    theta_0[2:] = np.random.randn(num_fe) * 1.0  # FE ~ N(0, 1)
    
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
    
    # Bounds: non-FE reasonable, FE bounded to prevent extreme values in bootstrap
    theta_lbs = [0.0, 0.0] + [-5.0] * num_fe  # FE bounded [-5, 5]
    theta_ubs = [15.0, 3.0] + [5.0] * num_fe
    
    config = {
        "dimensions": {"num_agents": num_agents, "num_items": num_items, 
                      "num_features": num_features, "num_simulations": 1},
        "subproblem": {"name": "Greedy"},
        "row_generation": {"max_iters": 200, "tolerance_optimality": 0.001,
                          "theta_lbs": theta_lbs, "theta_ubs": theta_ubs},
        "standard_errors": {"num_simulations": num_se_simulations, "step_size": step_size, 
                           "seed": 2024, "error_sigma": sigma},
    }
    
    # Feature oracle for greedy with FE
    def _greedy_oracle_fe(agent_idx, bundles, data):
        """Features = [sum(agent_modular), -bundle_size^2, FE for each item]"""
        modular = data["agent_data"]["modular"][agent_idx]  # (num_items, 1)
        
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
    bc_gen.oracles.set_features_oracle(_greedy_oracle_fe)
    bc_gen.subproblems.load()
    obs_bundles = bc_gen.subproblems.init_and_solve(theta_0)
    obs_bundles = comm.bcast(obs_bundles, root=0)
    
    # Bundle validation
    if rank == 0:
        bundle_sizes = obs_bundles.sum(axis=1)
        items_chosen = obs_bundles.sum(axis=0)
        
        print(f"\n--- BUNDLE VALIDATION ---")
        print(f"  Bundle sizes: min={bundle_sizes.min()}, max={bundle_sizes.max()}, mean={bundle_sizes.mean():.2f}")
        print(f"  Items chosen: min={items_chosen.min()}, max={items_chosen.max()}")
        
        issues = []
        if bundle_sizes.min() == 0:
            issues.append("Some agents choose no items")
        if bundle_sizes.max() == num_items:
            issues.append("Some agents choose all items")
        if items_chosen.min() < 5:
            issues.append(f"{(items_chosen < 5).sum()} items chosen by <5 agents")
        
        if issues:
            print(f"  ⚠️ Issues: {'; '.join(issues)}")
        else:
            print(f"  ✓ Distribution looks good")
        print("-------------------------\n")
    
    # Main estimation
    if rank == 0:
        print("[1] Main estimation...")
    est_data = {"agent_data": {"modular": modular_agent}, "item_data": {}, 
                "errors": est_errors, "obs_bundle": obs_bundles}
    bc = BundleChoice()
    bc.load_config(config)
    bc.data.load_and_scatter(est_data if rank == 0 else None)
    bc.oracles.set_features_oracle(_greedy_oracle_fe)
    bc.subproblems.load()
    
    result = bc.row_generation.solve()
    theta_hat = result.theta_hat
    
    if rank == 0:
        print(f"\nθ_hat (non-FE): {theta_hat[subset_indices]}")
        print(f"True (non-FE):  {theta_0[subset_indices]}")
    
    # [2] Partial (subset) sandwich SE
    if rank == 0:
        print("\n[2] Partial Sandwich SE (subset)...")
    bc.standard_errors.clear_cache()
    se_partial = bc.standard_errors.compute(
        theta_hat=theta_hat, num_simulations=num_se_simulations, step_size=step_size,
        beta_indices=subset_indices, optimize_for_subset=True, error_sigma=sigma,
    )
    
    # [3] B-inverse SE
    if rank == 0:
        print("\n[3] B-inverse SE...")
    bc.standard_errors.clear_cache()
    try:
        se_binv = bc.standard_errors.compute_B_inverse(
            theta_hat=theta_hat, num_simulations=num_se_simulations,
            beta_indices=subset_indices, error_sigma=sigma,
        )
    except Exception as e:
        if rank == 0:
            print(f"  B-inverse failed: {e}")
        se_binv = None
    
    # [4] Bootstrap
    if rank == 0:
        print(f"\n[4] Bootstrap ({num_bootstrap} resamples)...")
    theta_boots = []
    np.random.seed(999)
    boot_indices_list = [np.random.choice(num_agents, num_agents, replace=True) for _ in range(num_bootstrap)]
    boot_indices_list = comm.bcast(boot_indices_list, root=0)
    
    for b_idx, boot_indices in enumerate(boot_indices_list):
        if rank == 0 and (b_idx + 1) % 20 == 0:
            print(f"    Bootstrap {b_idx + 1}/{num_bootstrap}")
        
        boot_modular = modular_agent[boot_indices]
        boot_obs = obs_bundles[boot_indices]
        boot_errors = est_errors[boot_indices]
        
        boot_data = {"agent_data": {"modular": boot_modular}, "item_data": {}, 
                     "errors": boot_errors, "obs_bundle": boot_obs}
        bc_b = BundleChoice()
        bc_b.load_config(config)
        bc_b.data.load_and_scatter(boot_data if rank == 0 else None)
        bc_b.oracles.set_features_oracle(_greedy_oracle_fe)
        bc_b.subproblems.load()
        
        try:
            res_b = bc_b.row_generation.solve()
            if rank == 0:
                theta_boots.append(res_b.theta_hat)
        except Exception:
            pass
    
    theta_boots_arr = np.array(theta_boots) if rank == 0 and len(theta_boots) > 0 else None
    theta_boots_arr = comm.bcast(theta_boots_arr, root=0)
    
    # [5] Subsampling
    if rank == 0:
        print(f"\n[5] Subsampling ({num_bootstrap} subsamples)...")
    subsamp_size = num_agents // 4
    theta_subs = []
    np.random.seed(888)
    subs_indices_list = [np.random.choice(num_agents, subsamp_size, replace=False) for _ in range(num_bootstrap)]
    subs_indices_list = comm.bcast(subs_indices_list, root=0)
    
    for s_idx, subs_indices in enumerate(subs_indices_list):
        if rank == 0 and (s_idx + 1) % 20 == 0:
            print(f"    Subsample {s_idx + 1}/{num_bootstrap}")
        
        subs_modular = modular_agent[subs_indices]
        subs_obs = obs_bundles[subs_indices]
        subs_errors = est_errors[subs_indices]
        
        subs_config = config.copy()
        subs_config["dimensions"] = dict(config["dimensions"])
        subs_config["dimensions"]["num_agents"] = subsamp_size
        
        subs_data = {"agent_data": {"modular": subs_modular}, "item_data": {}, 
                     "errors": subs_errors, "obs_bundle": subs_obs}
        bc_s = BundleChoice()
        bc_s.load_config(subs_config)
        bc_s.data.load_and_scatter(subs_data if rank == 0 else None)
        bc_s.oracles.set_features_oracle(_greedy_oracle_fe)
        bc_s.subproblems.load()
        
        try:
            res_s = bc_s.row_generation.solve()
            if rank == 0:
                theta_subs.append(res_s.theta_hat)
        except Exception:
            pass
    
    theta_subs_arr = np.array(theta_subs) if rank == 0 and len(theta_subs) > 0 else None
    theta_subs_arr = comm.bcast(theta_subs_arr, root=0)
    
    # Results
    if rank == 0:
        # Extract SEs
        se_part = se_partial.se if se_partial else np.full(2, np.nan)
        se_binv_vals = se_binv.se if se_binv else np.full(2, np.nan)
        
        if theta_boots_arr is not None and len(theta_boots_arr) > 0:
            theta_boot_mean = np.mean(theta_boots_arr[:, subset_indices], axis=0)
            se_boot = np.std(theta_boots_arr[:, subset_indices], axis=0, ddof=1)
        else:
            theta_boot_mean = theta_hat[subset_indices]
            se_boot = np.full(2, np.nan)
        
        if theta_subs_arr is not None and len(theta_subs_arr) > 0:
            theta_subs_mean = np.mean(theta_subs_arr[:, subset_indices], axis=0)
            se_subs_raw = np.std(theta_subs_arr[:, subset_indices], axis=0, ddof=1)
            se_subs = np.sqrt(subsamp_size / num_agents) * se_subs_raw
        else:
            theta_subs_mean = theta_hat[subset_indices]
            se_subs = np.full(2, np.nan)
        
        print("\n" + "=" * 100)
        print("RESULTS (Fixed Effects)")
        print("=" * 100)
        
        print(f"\n{'Param':<12} {'θ_hat':<10} {'SE(Part)':<12} {'SE(B-inv)':<12} {'θ_boot':<10} {'SE(Boot)':<12} {'θ_subs':<10} {'SE(Subs)':<12}")
        print("-" * 100)
        names = ["Agent", "Quadratic"]
        for i, idx in enumerate(subset_indices):
            print(f"{names[i]:<12} {theta_hat[idx]:>8.4f}  {se_part[i]:>10.6f}  {se_binv_vals[i]:>10.6f}  "
                  f"{theta_boot_mean[i]:>8.4f}  {se_boot[i]:>10.6f}  {theta_subs_mean[i]:>8.4f}  {se_subs[i]:>10.6f}")
        
        print("\n" + "=" * 100)
        print("RATIOS (to Bootstrap)")
        print("=" * 100)
        for i in range(2):
            r_part = se_part[i] / se_boot[i] if se_boot[i] > 0 else np.nan
            r_binv = se_binv_vals[i] / se_boot[i] if se_boot[i] > 0 else np.nan
            r_subs = se_subs[i] / se_boot[i] if se_boot[i] > 0 else np.nan
            print(f"{names[i]:<12} Part/Boot: {r_part:.3f}, B-inv/Boot: {r_binv:.3f}, Subs/Boot: {r_subs:.3f}")


if __name__ == "__main__":
    main()
