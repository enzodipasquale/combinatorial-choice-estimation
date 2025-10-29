"""
Quadratic Supermodular Experiment

This script runs an experiment using the RowGenerationSolver with quadratic supermodular subproblem manager.
Based on the test but adapted for standalone execution and experimentation.
"""

import numpy as np
import time
from mpi4py import MPI
from bundlechoice.core import BundleChoice



def run_quad_supermod_experiment():
    """Run the quadratic supermodular experiment."""
    # Experiment parameters
    num_agents = 100
    num_items = 150
    num_modular_agent_features = 1
    num_modular_item_features = 1
    num_quadratic_item_features = 2
    num_features = num_modular_agent_features + num_modular_item_features + num_quadratic_item_features
    num_simuls = 10
    sigma = 3
    
    # Configuration
    cfg = {
        "dimensions": {
            "num_agents": num_agents,
            "num_items": num_items,
            "num_features": num_features,        },
        "subproblem": {
            "name": "QuadSupermodularNetwork",
            "settings": {}
        },
        "row_generation": {
            "max_iters": 1000,
            "tolerance_optimality": 0.0001,
            "max_slack_counter": 10,
            "min_iters": 10,
            "gurobi_settings": {
                "OutputFlag": 0
            }
        }
    }
    cfg["row_generation"]["theta_ubs"] = 10
    cfg["row_generation"]["theta_lbs"] = 0
    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print(f"[Rank {rank}] Starting Quadratic Supermodular Experiment")
        print(f"[Rank {rank}] Parameters: {num_agents} agents, {num_items} items, {num_features} features")
        print(f"[Rank {rank}] Features: {num_modular_agent_features} agent modular, {num_modular_item_features} item modular, {num_quadratic_item_features} item quadratic")
    
    # Generate data on rank 0
    if rank == 0:
        print("[Rank 0] Generating synthetic data...")
        
        # Generate agent modular features
        agent_modular = - np.abs(np.random.normal(2, 1, (num_agents, num_items, num_modular_agent_features)))
        
        # Generate modular item features with endogenous errors (same as greedy)
        modular_item =  -  np.abs(np.random.normal(0, 1, (num_items, num_modular_item_features)))
        endogenous_errors = np.random.normal(0, 1, size=(num_items,)) 
        instrument = np.random.normal(0, 1, size=(num_items,)) 
        modular_item[:,0] = instrument + endogenous_errors + np.random.normal(0, .5, size=(num_items,))
        # Generate quadratic features
        item_quadratic =  np.random.choice([0, 1], size=(num_items, num_items, num_quadratic_item_features), p=[.8, .2]) * .5
        item_quadratic *= (1.0 - np.eye(num_items))[:,:,None]

        # Generate errors with endogenous component
        errors = sigma * np.random.normal(0, 1, size=(num_agents, num_items)) + endogenous_errors[None,:]
        estimation_errors = sigma * np.random.normal(0, 1, size=(num_simuls, num_agents, num_items))

        input_data = {
            "item_data": {
                "modular":  modular_item,
                "quadratic": item_quadratic
            },
            "agent_data": {
                "modular": agent_modular,
            },
            "errors": errors,
        }
    else:
        input_data = None

    # Initialize BundleChoice
    if rank == 0:
        print(f"[Rank {rank}] Initializing BundleChoice...")
    quad_demo = BundleChoice()
    quad_demo.load_config(cfg)
    quad_demo.data.load_and_scatter(input_data)
    quad_demo.features.build_from_data()

    # Simulate theta_0 and generate obs_bundles
    if rank == 0:
        print(f"[Rank {rank}] Generating observed bundles...")
    theta_0 = np.ones(num_features) 
    theta_0[-num_quadratic_item_features:] = .1
    start_time = time.time()
    obs_bundles = quad_demo.subproblems.init_and_solve(theta_0)
    bundle_time = time.time() - start_time
    
    if rank == 0:
        print(f"[Rank 0] Bundle generation completed in {bundle_time:.2f} seconds")
        if obs_bundles is not None:
            total_demand = obs_bundles.sum(1)
            print(f"[Rank 0] Demand range: {total_demand.min():.2f} to {total_demand.max():.2f}")
            print(f"[Rank 0] Total aggregate: {obs_bundles.sum():.2f}")
            print(total_demand)
        else:
            print("[Rank 0] No bundles generated")
        
        input_data["obs_bundle"] = obs_bundles
        input_data["errors"] = estimation_errors
        cfg["dimensions"]["num_simuls"] = num_simuls
        # print(obs_bundles[:10]*1)
    else:
        input_data = None

    # Reinitialize for estimation
    if rank == 0:
        print(f"[Rank {rank}] Setting up for parameter estimation...")
    quad_demo.load_config(cfg)
    quad_demo.data.load_and_scatter(input_data)
    quad_demo.features.build_from_data()
    quad_demo.subproblems.load()
    
    # Run row generation method
    if rank == 0:
        print(f"[Rank {rank}] Starting row generation optimization...")
    start_time = time.time()
    theta_hat = quad_demo.row_generation.solve()
    optimization_time = time.time() - start_time
    
    # Compute objective values on all ranks
    try:
        obj_at_star = quad_demo.row_generation.objective(theta_0)
        obj_at_hat = quad_demo.row_generation.objective(theta_hat)
    except AttributeError:
        obj_at_star = None
        obj_at_hat = None
    
    if rank == 0:
        print(f"[Rank 0] Optimization completed in {optimization_time:.2f} seconds")
        print(f"[Rank 0] Estimated parameters (theta_hat): {theta_hat}")
        print(f"[Rank 0] True parameters (theta_0): {theta_0}")
        print(f"[Rank 0] Parameter difference: {np.round(np.abs(theta_hat - theta_0), 2)}")
        
        # Print objective values if available
        if obj_at_star is not None and obj_at_hat is not None:
            print(f"[Rank 0] Objective at true parameters: {obj_at_star:.4f}")
            print(f"[Rank 0] Objective at estimated parameters: {obj_at_hat:.4f}")
            print(f"[Rank 0] Objective improvement: {obj_at_star - obj_at_hat:.4f}")
        else:
            print("[Rank 0] Objective function not available for row generation solver")
    
    ########## Modular BLP inversion ##########
    if rank == 0:
        print(f"[Rank 0] theta_hat: {theta_hat}")
        input_data["item_data"]["modular"] = np.eye(num_items)
        input_data["obs_bundle"] = obs_bundles
        input_data["errors"] = estimation_errors 
    else:
        input_data = None
    cfg["dimensions"]["num_simuls"] = num_simuls
    cfg["dimensions"]["num_features"] = num_modular_agent_features + num_items + num_quadratic_item_features
    cfg["row_generation"]["theta_lbs"] = [0] * num_modular_agent_features + [-500] * num_items + [0] * num_quadratic_item_features
    cfg["row_generation"]["theta_ubs"] = 500
    parameters_to_log = [i for i in range(num_modular_agent_features)] + [-i-1 for i in range(num_quadratic_item_features)] 
    cfg["row_generation"]["parameters_to_log"] = parameters_to_log
    quad_demo.load_config(cfg)
    quad_demo.data.load_and_scatter(input_data)
    quad_demo.features.build_from_data()
    quad_demo.subproblems.load()
    start_time = time.time()
    theta_hat = quad_demo.row_generation.solve()
    if rank == 0:
        delta_hat = theta_hat[num_modular_agent_features:-num_quadratic_item_features]
        import statsmodels.api as sm
        from statsmodels.sandbox.regression.gmm import IV2SLS
        ols_model = sm.OLS(delta_hat, modular_item)
        ols_results = ols_model.fit()
        print(f"Coefficients: {ols_results.params}")
        print(f"Standard errors: {ols_results.bse}")
        # print(f"theta_hat: {theta_hat}")    
        iv_model = IV2SLS(delta_hat, modular_item, instrument)
        iv_results = iv_model.fit()
        print(f"Coefficients: {iv_results.params}")
        print(f"Standard errors: {iv_results.bse}")
        print(f"theta_hat: {theta_hat[parameters_to_log]}")   


if __name__ == "__main__":
    run_quad_supermod_experiment() 