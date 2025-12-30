"""
Row Generation Greedy Experiment with Endogeneity (Factory-based)

This script runs an experiment using the RowGenerationSolver with greedy subproblem manager.
Uses the factory pattern with endogeneity support for BLP inversion.
"""
import numpy as np
import time
from mpi4py import MPI
from bundlechoice.factory import ScenarioLibrary
from bundlechoice.core import BundleChoice

def run_row_generation_greedy_experiment():
    """Run the row generation greedy experiment with endogeneity."""
    # Experiment parameters
    num_agent_features = 1
    num_item_features = 1
    num_agents = 1000  # Increased from 500
    num_items = 60  # Increased from 50
    num_features = num_agent_features + num_item_features + 1
    num_simuls = 1
    sigma = 6
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    ########## Generate data using factory with retry if needed ##########
    theta_0 = np.ones(num_features) * 2
    theta_0[-1] = 0.1
    
    # Parameters that we'll adjust if needed
    item_multiplier = 1.5
    item_std = 1.5
    max_retries = 5
    
    for attempt in range(max_retries):
        scenario = (
            ScenarioLibrary.greedy()
            .with_dimensions(num_agents=num_agents, num_items=num_items)
            .with_num_features(num_features)
            .with_sigma(sigma)
            .with_num_simuls(num_simuls)
            .with_item_config(apply_abs=True, multiplier=item_multiplier, mean=0.0, std=item_std)
            .with_endogeneity(
                endogenous_feature_indices=[0],
                num_instruments=1,
                pi_matrix=np.array([[1.0]]),
                lambda_matrix=np.array([[1.0]]),
                xi_cov=1.0,
            )
            .build()
        )
        
        prepared = scenario.prepare(comm=comm, seed=42, theta=theta_0)
        
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
                # Adjust: increase variation to make items more attractive
                item_multiplier = min(3.0, item_multiplier + 0.3)
                item_std = min(3.0, item_std + 0.3)
            if len(always_chosen) > 0:
                print(f"\n⚠️  FLAG (attempt {attempt+1}): {len(always_chosen)} items are chosen by ALL agents!")
                print(f"   Always-chosen items: {always_chosen.tolist()}")
                # Adjust: decrease variation to make items less universally attractive
                item_multiplier = max(0.5, item_multiplier - 0.3)
                item_std = max(0.5, item_std - 0.3)
            
            if has_issue:
                if attempt < max_retries - 1:
                    print(f"   Item demands: min={item_demands.min()}, max={item_demands.max()}, mean={item_demands.mean():.2f}")
                    print(f"   Adjusting parameters: multiplier={item_multiplier:.2f}, std={item_std:.2f}")
                else:
                    print(f"   Item demands: min={item_demands.min()}, max={item_demands.max()}, mean={item_demands.mean():.2f}")
                    raise ValueError(f"Item choice issue persists after {max_retries} attempts: {len(never_chosen)} never chosen, {len(always_chosen)} always chosen.")
            else:
                print(f"\n✓ All {num_items} items are chosen by at least one agent (and not by all).")
                print(f"   Item demands: min={item_demands.min()}, max={item_demands.max()}, mean={item_demands.mean():.2f}")
        
        # Broadcast updated parameters to all ranks
        item_multiplier = comm.bcast(item_multiplier, root=0)
        item_std = comm.bcast(item_std, root=0)
        
        if not has_issue:
            break
    
    if rank == 0:
        # Extract data for BLP inversion (after successful generation)
        gen_data = prepared.generation_data
        instruments = gen_data["instruments"]
        original_modular_item = gen_data["original_modular_item"]
        modular_item = gen_data["item_data"]["modular"]
    
    ########## Naive estimation (without fixed effects, using endogenous features) ##########
    if rank == 0:
        print(f"[Rank {rank}] Setting up for NAIVE estimation (no fixed effects, endogenous features)...")
    
    bc_naive = BundleChoice()
    num_simuls_actual = prepared.metadata.get("num_simuls", 1)
    # Ensure config has correct num_simuls for estimation stage
    prepared.config["dimensions"]["num_simuls"] = num_simuls_actual
    # Update row generation config to match original
    prepared.config["row_generation"]["max_iters"] = 500
    prepared.config["row_generation"]["tolerance_optimality"] = 0.001
    prepared.config["row_generation"]["max_slack_counter"] = 5
    
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
        # obs_bundle and errors are already in est_data from prepared.estimation_data
        
        # Update config for BLP inversion (modify in place like old version)
        prepared.config["dimensions"]["num_features"] = num_agent_features + num_items + 1
        prepared.config["dimensions"]["num_simuls"] = num_simuls_actual
        prepared.config["row_generation"]["theta_lbs"] = [0] * num_agent_features + [-500] * num_items + [0]
        prepared.config["row_generation"]["theta_ubs"] = 500
        parameters_to_log = [i for i in range(num_agent_features)] + [-1]
        prepared.config["row_generation"]["parameters_to_log"] = parameters_to_log
    else:
        est_data_blp = None
        # Update config on non-root ranks too (same modifications)
        prepared.config["dimensions"]["num_features"] = num_agent_features + num_items + 1
        prepared.config["dimensions"]["num_simuls"] = num_simuls_actual
        prepared.config["row_generation"]["theta_lbs"] = [0] * num_agent_features + [-500] * num_items + [0]
        prepared.config["row_generation"]["theta_ubs"] = 500
        parameters_to_log = [i for i in range(num_agent_features)] + [-1]
        prepared.config["row_generation"]["parameters_to_log"] = parameters_to_log
    
    # Setup for BLP inversion (new BundleChoice instance for fixed effects estimation)
    bc_blp = BundleChoice()
    bc_blp.load_config(prepared.config)
    bc_blp.data.load_and_scatter(est_data_blp if rank == 0 else None)
    # For BLP inversion, we use identity matrix for modular, so we need the oracle
    bc_blp.features.set_oracle(_greedy_features_oracle)
    bc_blp.subproblems.load()
    
    # Reinstall find_best_item after subproblems are reloaded
    from bundlechoice.subproblems.registry.greedy import GreedySubproblem
    from bundlechoice.factory.greedy import _install_find_best_item
    if isinstance(bc_blp.subproblems.subproblem_instance, GreedySubproblem):
        _install_find_best_item(bc_blp.subproblems.subproblem_instance)
    
    start_time = time.time()
    result_blp = bc_blp.row_generation.solve()
    theta_hat_blp = result_blp.theta_hat
    
    if rank == 0:
        # Extract IV coefficient from BLP inversion
        delta_hat = theta_hat_blp[num_agent_features:-1]
        import statsmodels.api as sm
        from statsmodels.sandbox.regression.gmm import IV2SLS
        
        # Extract endogenous feature (the one used in data generation)
        endogenous_feature = modular_item[:, 0]  # First column is the endogenous feature
        
        # IV regression (unbiased) - regress on endogenous feature using instruments
        instrument = instruments[:, 0]  # Extract single instrument
        iv_model = IV2SLS(delta_hat, endogenous_feature, instrument)
        iv_results = iv_model.fit()
        iv_coefficient = iv_results.params[0]
        
        # Construct IV version: [agent_feature, IV_item_coefficient, quadratic]
        theta_iv = np.array([
            theta_hat_naive[0],  # Agent feature from naive estimate
            iv_coefficient,      # Item feature from IV regression
            theta_hat_naive[2]  # Quadratic from naive estimate
        ])
        
        print("\n" + "=" * 100)
        print("RESULTS")
        print("=" * 100)
        print(f"Experiment size:   {num_agents} agents, {num_items} items, {num_features} features")
        print(f"True theta:        {theta_0}")
        print(f"Naive (no fixed effects, endogenous features): {theta_hat_naive}")
        print(f"IV (with fixed effects + IV regression):        {theta_iv}")
        print("=" * 100)


def _greedy_features_oracle(agent_idx: int, bundles: np.ndarray, data: dict) -> np.ndarray:
    """
    Compute features for a given agent and bundle(s).
    Includes both agent and item modular features (needed when endogeneity is enabled).
    Supports both single (1D) and multiple (2D) bundles.
    Returns array of shape (num_features,) for a single bundle,
    or (num_features, m) for m bundles.
    """
    modular_agent = data["agent_data"]["modular"][agent_idx]
    
    # Check if item features exist (when endogeneity is enabled)
    has_item_features = (
        "item_data" in data 
        and data["item_data"] is not None 
        and "modular" in data["item_data"]
    )
    
    if has_item_features:
        modular_item = data["item_data"]["modular"]
        modular_item = np.atleast_2d(modular_item)
        modular_agent = np.atleast_2d(modular_agent)
        modular = np.concatenate([modular_agent, modular_item], axis=1)
    else:
        # No item features - use only agent features
        modular_agent = np.atleast_2d(modular_agent)
        modular = modular_agent
    
    single_bundle = False
    if bundles.ndim == 1:
        bundles = bundles[:, None]
        single_bundle = True
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        modular_features = modular.T @ bundles
    neg_sq = -np.sum(bundles, axis=0, keepdims=True) ** 2
    
    features = np.vstack((modular_features, neg_sq))
    if single_bundle:
        return features[:, 0]  # Return as 1D array for a single bundle
    return features


if __name__ == "__main__":
    run_row_generation_greedy_experiment()
