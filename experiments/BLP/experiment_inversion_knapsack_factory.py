"""
Row Generation Linear Knapsack Experiment with Endogeneity (Factory-based)

This script runs an experiment using the RowGenerationSolver with linear knapsack subproblem manager.
Uses the factory pattern with endogeneity support for BLP inversion.
"""
import numpy as np
import time
from mpi4py import MPI
from bundlechoice.scenarios import ScenarioLibrary
from bundlechoice.core import BundleChoice

def run_row_generation_knapsack_experiment():
    """Run the row generation linear knapsack experiment with endogeneity."""
    # Experiment parameters
    num_agent_features = 1
    num_item_features = 1
    num_agents = 1000
    num_items = 80
    num_features = num_agent_features + num_item_features
    num_simulations = 1
    sigma = 3
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    ########## Generate data using factory with retry if needed ##########
    theta_0 = np.ones(num_features) * 2
    
    # Parameters that we'll adjust if needed
    weight_low = 1
    weight_high = 5
    capacity_fraction = 0.7
    max_retries = 5
    
    for attempt in range(max_retries):
        scenario = (
            ScenarioLibrary.linear_knapsack()
            .with_dimensions(num_agents=num_agents, num_items=num_items)
            .with_feature_counts(
                num_agent_features=num_agent_features,
                num_item_features=num_item_features,
            )
            .with_sigma(sigma)
            .with_num_simulations(num_simulations)
            .with_weight_config(
                distribution="uniform",
                low=weight_low,
                high=weight_high,
            )
            .with_capacity_fraction(
                fraction=capacity_fraction,
            )
            .with_endogeneity(
                endogenous_feature_indices=[0],
                num_instruments=1,
                pi_matrix=np.array([[1.0]]),
                lambda_matrix=np.array([[1.0]]),
                xi_cov=1.0,
            )
            .build()
        )
        
        prepared = scenario.prepare(comm=comm, seed=None, theta=theta_0)
        
        # Check item demands on all ranks (but only root prints)
        obs_bundles = prepared.estimation_data["obs_bundle"]
        item_demands = obs_bundles.sum(axis=0) if rank == 0 else None
        item_demands = comm.bcast(item_demands, root=0)
        
        never_chosen = np.where(item_demands == 0)[0]
        always_chosen = np.where(item_demands == num_agents)[0]
        
        has_issue = len(never_chosen) > 0 or len(always_chosen) > 0
        
        if rank == 0:
            if len(never_chosen) > 0:
                print(f"\n⚠️  FLAG (attempt {attempt+1}): {len(never_chosen)} items are NEVER chosen by any agent!")
                print(f"   Never-chosen items: {never_chosen.tolist()}")
                # Adjust: increase capacity and reduce weight range
                capacity_fraction = min(0.9, capacity_fraction + 0.1)
                weight_high = max(weight_low + 1, weight_high - 1)
            if len(always_chosen) > 0:
                print(f"\n⚠️  FLAG (attempt {attempt+1}): {len(always_chosen)} items are chosen by ALL agents!")
                print(f"   Always-chosen items: {always_chosen.tolist()}")
                # Adjust: decrease capacity and increase weight range
                capacity_fraction = max(0.3, capacity_fraction - 0.1)
                weight_high = min(10, weight_high + 2)
            
            if has_issue:
                if attempt < max_retries - 1:
                    print(f"   Item demands: min={item_demands.min()}, max={item_demands.max()}, mean={item_demands.mean():.2f}")
                    print(f"   Adjusting parameters: weight_high={weight_high}, capacity_fraction={capacity_fraction:.2f}")
                else:
                    print(f"   Item demands: min={item_demands.min()}, max={item_demands.max()}, mean={item_demands.mean():.2f}")
                    raise ValueError(f"Item choice issue persists after {max_retries} attempts: {len(never_chosen)} never chosen, {len(always_chosen)} always chosen.")
            else:
                print(f"\n✓ All {num_items} items are chosen by at least one agent (and not by all).")
                print(f"   Item demands: min={item_demands.min()}, max={item_demands.max()}, mean={item_demands.mean():.2f}")
        
        # Broadcast updated parameters to all ranks
        capacity_fraction = comm.bcast(capacity_fraction, root=0)
        weight_high = comm.bcast(weight_high, root=0)
        
        if not has_issue:
            break
    
    if rank == 0:
        # Extract data for BLP inversion (after successful generation)
        gen_data = prepared.generation_data
        instruments = gen_data["instruments"]
        original_modular_item = gen_data["original_modular_item"]
        modular_item = gen_data["item_data"]["modular"]  # Endogenous features
    
    ########## Naive estimation (without fixed effects, using endogenous features) ##########
    if rank == 0:
        print(f"[Rank {rank}] Setting up for NAIVE estimation (no fixed effects, endogenous features)...")
    
    bc_naive = BundleChoice()
    num_simulations_actual = prepared.metadata.get("num_simulations", 1)
    # Ensure config has correct num_simulations for estimation stage
    prepared.config["dimensions"]["num_simulations"] = num_simulations_actual
    # Update row generation config
    prepared.config["row_generation"]["max_iters"] = 500
    prepared.config["row_generation"]["tolerance_optimality"] = 0.001
    prepared.config["row_generation"]["max_slack_counter"] = 5
    prepared.config["row_generation"]["min_iters"] = 1
    
    # Use estimation data which contains endogenous item features (no fixed effects)
    # This gives us the biased naive estimate
    prepared.apply(bc_naive, comm=comm, stage="estimation")
    
    # Run row generation method (naive estimation with endogenous features, no IV correction)
    result_naive = bc_naive.row_generation.solve()
    theta_hat_naive = result_naive.theta_hat
    
    if rank == 0:
        print("\n" + "=" * 100)
        print("NAIVE ESTIMATION (no fixed effects, endogenous features)")
        print("=" * 100)
        print(f"True theta: {theta_0}")
        print(f"Naive estimate: {theta_hat_naive}")
        print(f"Difference: {theta_hat_naive - theta_0}")
        print("=" * 100 + "\n")
    
    ########## BLP inversion (with fixed effects) ##########
    if rank == 0:
        print(f"[Rank 0] Starting BLP inversion (with fixed effects)...")
        print("#" * 100)
        
        # Prepare data for BLP inversion: replace item features with identity matrix (fixed effects)
        est_data_blp = prepared.estimation_data.copy()
        est_data_blp["item_data"] = est_data_blp["item_data"].copy()
        est_data_blp["item_data"]["modular"] = np.eye(num_items)  # Identity matrix for fixed effects
        # Keep weights and other data
        # obs_bundle and errors are already in est_data from prepared.estimation_data
        
        # Update config for BLP inversion (modify in place)
        prepared.config["dimensions"]["num_features"] = num_agent_features + num_items
        prepared.config["dimensions"]["num_simulations"] = num_simulations_actual
        prepared.config["row_generation"]["theta_lbs"] = [0] * num_agent_features + [-1e8] * num_items  # Very generous bounds
        prepared.config["row_generation"]["theta_ubs"] = 1e8  # Very generous bounds
        prepared.config["row_generation"]["max_iters"] = 500
        prepared.config["row_generation"]["tolerance_optimality"] = 0.0001  # Tighter tolerance
        prepared.config["row_generation"]["max_slack_counter"] = 4  # More slack iterations
        prepared.config["row_generation"]["min_iters"] = 1
        parameters_to_log = [i for i in range(num_agent_features)]
        prepared.config["row_generation"]["parameters_to_log"] = parameters_to_log
    else:
        est_data_blp = None
        # Update config on non-root ranks too (same modifications)
        prepared.config["dimensions"]["num_features"] = num_agent_features + num_items
        prepared.config["dimensions"]["num_simulations"] = num_simulations_actual
        prepared.config["row_generation"]["theta_lbs"] = [0] * num_agent_features + [-1e8] * num_items  # Very generous bounds
        prepared.config["row_generation"]["theta_ubs"] = 1e8  # Very generous bounds
        prepared.config["row_generation"]["max_iters"] = 500
        prepared.config["row_generation"]["tolerance_optimality"] = 0.0001  # Tighter tolerance
        prepared.config["row_generation"]["max_slack_counter"] = 4  # More slack iterations
        prepared.config["row_generation"]["min_iters"] = 1
        parameters_to_log = [i for i in range(num_agent_features)]
        prepared.config["row_generation"]["parameters_to_log"] = parameters_to_log
    
    # Setup for BLP inversion (new BundleChoice instance for fixed effects estimation)
    bc_blp = BundleChoice()
    bc_blp.load_config(prepared.config)
    bc_blp.data.load_and_scatter(est_data_blp if rank == 0 else None)
    # Linear knapsack uses FeatureSpec.build() which auto-generates from data
    # So we don't need to set an oracle - it will build from the identity matrix
    bc_blp.features.build_from_data()
    bc_blp.subproblems.load()
    
    start_time = time.time()
    result_blp = bc_blp.row_generation.solve()
    theta_hat_blp = result_blp.theta_hat
    
    if rank == 0:
        elapsed_time = time.time() - start_time
        print(f"\nBLP inversion completed in {elapsed_time:.2f}s")
        # Extract IV coefficient from BLP inversion
        delta_hat = theta_hat_blp[num_agent_features:]
        import statsmodels.api as sm
        from statsmodels.sandbox.regression.gmm import IV2SLS
        
        # Extract endogenous feature (the one used in data generation)
        endogenous_feature = modular_item[:, 0]  # First column is the endogenous feature
        
        # All items are chosen (verified above), so use all items
        delta_hat_filtered = delta_hat
        endogenous_feature_filtered = endogenous_feature
        instruments_filtered = instruments
        
        # Compute true deltas for comparison (only for chosen items)
        # True delta_j = true_item_feature_coefficient * endogenous_feature_j
        # This is what the deltas would be if we knew the true item feature coefficient
        true_item_feature_coefficient = theta_0[1]  # Should be 2.0
        true_deltas_filtered = true_item_feature_coefficient * endogenous_feature_filtered
        
        # Run IV regression on true deltas - should recover true_item_feature_coefficient exactly
        instrument_filtered = instruments_filtered[:, 0]
        iv_model_true = IV2SLS(true_deltas_filtered, endogenous_feature_filtered, instrument_filtered)
        iv_results_true = iv_model_true.fit()
        iv_coefficient_true = iv_results_true.params[0]
        
        print(f"\nDEBUG: IV regression on TRUE deltas:")
        print(f"  True item feature coefficient: {true_item_feature_coefficient:.4f}")
        print(f"  IV coefficient from true deltas: {iv_coefficient_true:.4f}")
        print(f"  Error: {abs(iv_coefficient_true - true_item_feature_coefficient):.6f}")
        
        # Debug: Compare estimated deltas to true deltas
        true_deltas_all = true_item_feature_coefficient * endogenous_feature
        print(f"\nDEBUG: Delta comparison:")
        print(f"  True deltas: min={true_deltas_all.min():.2f}, max={true_deltas_all.max():.2f}, mean={true_deltas_all.mean():.2f}, std={true_deltas_all.std():.2f}")
        print(f"  Estimated deltas: min={delta_hat.min():.2f}, max={delta_hat.max():.2f}, mean={delta_hat.mean():.2f}, std={delta_hat.std():.2f}")
        print(f"  Correlation between true and estimated deltas: {np.corrcoef(true_deltas_all, delta_hat)[0,1]:.4f}")
        print(f"  Mean absolute error: {np.mean(np.abs(delta_hat - true_deltas_all)):.2f}")
        print(f"  RMSE: {np.sqrt(np.mean((delta_hat - true_deltas_all)**2)):.2f}")
        
        # IV regression (unbiased) - regress on endogenous feature using instruments
        # Only use items that were chosen by at least one agent
        iv_model = IV2SLS(delta_hat_filtered, endogenous_feature_filtered, instrument_filtered)
        iv_results = iv_model.fit()
        iv_coefficient = iv_results.params[0]
        iv_se = iv_results.bse[0]
        
        print(f"\nDEBUG: IV regression on estimated deltas:")
        print(f"  IV coefficient: {iv_coefficient:.4f} ± {iv_se:.4f}")
        print(f"  True coefficient: {true_item_feature_coefficient:.4f}")
        print(f"  Error: {abs(iv_coefficient - true_item_feature_coefficient):.4f} ({100*abs(iv_coefficient - true_item_feature_coefficient)/true_item_feature_coefficient:.1f}%)")
        
        # Construct IV version: [agent_feature, IV_item_coefficient]
        theta_iv = np.array([
            theta_hat_naive[0],  # Agent feature from naive estimate
            iv_coefficient,      # Item feature from IV regression
        ])
        
        print("\n" + "=" * 100)
        print("RESULTS")
        print("=" * 100)
        print(f"Experiment size:   {num_agents} agents, {num_items} items, {num_features} features")
        print(f"\nCoefficient breakdown:")
        print(f"  {'Parameter':<30} {'True':<12} {'Naive':<12} {'IV':<12} {'Error (Naive)':<15} {'Error (IV)':<15}")
        print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*12} {'-'*15} {'-'*15}")
        for i in range(num_features):
            param_name = f"Feature {i}"
            if i < num_agent_features:
                param_name = f"Agent feature {i}"
            elif i < num_agent_features + num_item_features:
                param_name = f"Item feature {i - num_agent_features} (endogenous)"
            true_val = theta_0[i]
            naive_val = theta_hat_naive[i]
            iv_val = theta_iv[i]
            naive_error = abs(naive_val - true_val)
            iv_error = abs(iv_val - true_val)
            naive_error_pct = 100 * naive_error / abs(true_val) if true_val != 0 else 0
            iv_error_pct = 100 * iv_error / abs(true_val) if true_val != 0 else 0
            print(f"  {param_name:<30} {true_val:>11.4f} {naive_val:>11.4f} {iv_val:>11.4f} {naive_error:>8.4f} ({naive_error_pct:>5.1f}%) {iv_error:>8.4f} ({iv_error_pct:>5.1f}%)")
        print(f"\nSummary arrays:")
        print(f"True theta:        {theta_0}")
        print(f"Naive (no fixed effects, endogenous features): {theta_hat_naive}")
        print(f"IV (with fixed effects + IV regression):        {theta_iv}")
        print("=" * 100)


if __name__ == "__main__":
    run_row_generation_knapsack_experiment()

