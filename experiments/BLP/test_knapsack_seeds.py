#!/usr/bin/env python3
"""Test knapsack experiment with different seeds."""
import numpy as np
from mpi4py import MPI
from bundlechoice.scenarios import ScenarioLibrary
from bundlechoice.core import BundleChoice
from statsmodels.sandbox.regression.gmm import IV2SLS

def run_with_seed(seed: int):
    """Run knapsack experiment with given seed."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Experiment parameters
    num_agent_features = 1
    num_item_features = 1
    num_agents = 1000
    num_items = 80
    num_features = num_agent_features + num_item_features
    num_simuls = 1
    sigma = 3
    
    # Generate data using factory
    scenario = (
        ScenarioLibrary.linear_knapsack()
        .with_dimensions(num_agents=num_agents, num_items=num_items)
        .with_feature_counts(
            num_agent_features=num_agent_features,
            num_item_features=num_item_features,
        )
        .with_sigma(sigma)
        .with_num_simuls(num_simuls)
        .with_weight_config(
            distribution="uniform",
            low=1,
            high=3,  # Narrower range to ensure all items can be chosen
        )
        .with_capacity_fraction(
            fraction=0.6,  # Fixed capacity = 60% of total weight sum
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
    
    theta_0 = np.ones(num_features) * 2
    prepared = scenario.prepare(comm=comm, seed=seed, theta=theta_0)
    
    if rank == 0:
        gen_data = prepared.generation_data
        instruments = gen_data["instruments"]
        modular_item = gen_data["item_data"]["modular"]
        endogenous_feature = modular_item[:, 0]
        obs_bundles = prepared.estimation_data["obs_bundle"]
        
        # Check that all items are chosen by at least one agent
        item_demands = obs_bundles.sum(axis=0)
        never_chosen = np.where(item_demands == 0)[0]
        if len(never_chosen) > 0:
            print(f"Seed {seed}: {len(never_chosen)} items never chosen: {never_chosen.tolist()}")
    
    # Naive estimation
    bc_naive = BundleChoice()
    num_simuls_actual = prepared.metadata.get("num_simuls", 1)
    prepared.config["dimensions"]["num_simuls"] = num_simuls_actual
    prepared.config["row_generation"]["max_iters"] = 500
    prepared.config["row_generation"]["tolerance_optimality"] = 0.001
    prepared.config["row_generation"]["max_slack_counter"] = 5
    prepared.config["row_generation"]["min_iters"] = 1
    prepared.apply(bc_naive, comm=comm, stage="estimation")
    result_naive = bc_naive.row_generation.solve()
    theta_hat_naive = result_naive.theta_hat
    
    # BLP inversion
    if rank == 0:
        est_data_blp = prepared.estimation_data.copy()
        est_data_blp["item_data"] = est_data_blp["item_data"].copy()
        est_data_blp["item_data"]["modular"] = np.eye(num_items)
        prepared.config["dimensions"]["num_features"] = num_agent_features + num_items
        prepared.config["dimensions"]["num_simuls"] = num_simuls_actual
        prepared.config["row_generation"]["theta_lbs"] = [0] * num_agent_features + [-1e8] * num_items  # Very generous bounds
        prepared.config["row_generation"]["theta_ubs"] = 1e8  # Very generous bounds
        prepared.config["row_generation"]["max_iters"] = 500
        prepared.config["row_generation"]["tolerance_optimality"] = 0.001
        prepared.config["row_generation"]["max_slack_counter"] = 5
        prepared.config["row_generation"]["min_iters"] = 1
        parameters_to_log = [i for i in range(num_agent_features)]
        prepared.config["row_generation"]["parameters_to_log"] = parameters_to_log
    else:
        est_data_blp = None
        prepared.config["dimensions"]["num_features"] = num_agent_features + num_items
        prepared.config["dimensions"]["num_simuls"] = num_simuls_actual
        prepared.config["row_generation"]["theta_lbs"] = [0] * num_agent_features + [-1e8] * num_items  # Very generous bounds
        prepared.config["row_generation"]["theta_ubs"] = 1e8  # Very generous bounds
        prepared.config["row_generation"]["max_iters"] = 500
        prepared.config["row_generation"]["tolerance_optimality"] = 0.001
        prepared.config["row_generation"]["max_slack_counter"] = 5
        prepared.config["row_generation"]["min_iters"] = 1
        parameters_to_log = [i for i in range(num_agent_features)]
        prepared.config["row_generation"]["parameters_to_log"] = parameters_to_log
    
    bc_blp = BundleChoice()
    bc_blp.load_config(prepared.config)
    bc_blp.data.load_and_scatter(est_data_blp if rank == 0 else None)
    bc_blp.features.build_from_data()
    bc_blp.subproblems.load()
    result_blp = bc_blp.row_generation.solve()
    theta_hat_blp = result_blp.theta_hat
    
    if rank == 0:
        delta_hat = theta_hat_blp[num_agent_features:]
        
        # All items should be chosen (verified by weight/capacity config)
        item_demands = obs_bundles.sum(axis=0)
        never_chosen = np.where(item_demands == 0)[0]
        if len(never_chosen) > 0:
            raise ValueError(f"Seed {seed}: {len(never_chosen)} items never chosen: {never_chosen.tolist()}")
        
        # Use all items
        delta_hat_filtered = delta_hat
        endogenous_feature_filtered = endogenous_feature
        instruments_filtered = instruments
        
        instrument = instruments_filtered[:, 0]
        iv_model = IV2SLS(delta_hat_filtered, endogenous_feature_filtered, instrument)
        iv_results = iv_model.fit()
        iv_coefficient = iv_results.params[0]
        iv_se = iv_results.bse[0]
        
        return {
            "seed": seed,
            "naive_item": theta_hat_naive[1],
            "iv_item": iv_coefficient,
            "iv_se": iv_se,
            "true_item": theta_0[1],
            "never_chosen": len(never_chosen),
        }
    return None

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Single seed from command line
        seed = int(sys.argv[1])
        result = run_with_seed(seed)
        if result:
            print(f"Seed {result['seed']:3d}: Naive={result['naive_item']:7.4f}, IV={result['iv_item']:7.4f}±{result['iv_se']:.4f}, True={result['true_item']:.1f}")
    else:
        # Run with multiple seeds (smaller subset for faster testing)
        seeds = [42, 123, 456, 789, 1, 10, 25, 50, 100]
        print("Running knapsack experiment with multiple seeds...")
        print("=" * 80)
        results = []
        for seed in seeds:
            result = run_with_seed(seed)
            if result:
                results.append(result)
                print(f"Seed {result['seed']:4d}: Naive={result['naive_item']:7.4f}, IV={result['iv_item']:10.2f}±{result['iv_se']:.2f}, True={result['true_item']:.1f}")
        
        if results:
            print("=" * 80)
            naive_vals = [r['naive_item'] for r in results]
            iv_vals = [r['iv_item'] for r in results]
            print(f"Summary across {len(results)} seeds:")
            print(f"  Naive: mean={np.mean(naive_vals):.4f}, std={np.std(naive_vals):.4f}, min={np.min(naive_vals):.4f}, max={np.max(naive_vals):.4f}")
            print(f"  IV:    mean={np.mean(iv_vals):.2f}, std={np.std(iv_vals):.2f}, min={np.min(iv_vals):.2f}, max={np.max(iv_vals):.2f}")
            print(f"  True:  {results[0]['true_item']:.1f}")

