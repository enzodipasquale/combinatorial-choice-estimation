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
    num_agents = 200
    num_items = 100
    num_modular_item_features = num_items
    num_quadratic_item_features = 2
    num_features = num_modular_item_features + num_quadratic_item_features
    num_simuls = 1
    sigma = 3
    
    # Configuration
    cfg = {
        "dimensions": {
            "num_agents": num_agents,
            "num_items": num_items,
            "num_features": num_features,
            "num_simuls": num_simuls
        },
        "subproblem": {
            "name": "QuadSupermodularNetwork",
            "settings": {}
        },
        "row_generation": {
            "max_iters": 100,
            "tolerance_optimality": 0.0001,
            "max_slack_counter": 10,
            "min_iters": 10,
            "master_settings": {
                "OutputFlag": 0
            }
        }
    }
    
    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print(f"[Rank {rank}] Starting Quadratic Supermodular Experiment")
        print(f"[Rank {rank}] Parameters: {num_agents} agents, {num_items} items, {num_features} features")
        print(f"[Rank {rank}] Features: {num_modular_item_features} item modular, {num_quadratic_item_features} item quadratic")
    
    # Generate data on rank 0
    if rank == 0:
        print("[Rank 0] Generating synthetic data...")
        item_modular = - 3 * np.eye(num_items)
        # item_quadratic = np.exp(-np.abs(np.random.normal(0, sigma, (num_items, num_items, num_quadratic_item_features))))
        # item_quadratic *= (np.random.rand(num_items, num_items, num_quadratic_item_features) < .3)
        item_quadratic = np.random.choice([0, 1], size=(num_items, num_items, num_quadratic_item_features), p=[.8, .2]) * 1.0
        item_quadratic *= (1.0 - np.eye(num_items))[:,:,None]

        errors = sigma * np.random.normal(0, 1, (num_agents, num_items))
        estimation_errors = sigma * np.random.normal(0, 1, (num_simuls, num_agents, num_items))

      
        input_data = {
            "item_data": {
                "modular": item_modular,
                "quadratic": item_quadratic
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
    # theta_0 = np.ones(num_features) 
    theta_0 = np.random.randint(1,3,num_features).astype(np.float64)
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
        
        # Summary
        print(f"\n[Rank 0] Experiment Summary:")
        print(f"  - Bundle generation time: {bundle_time:.2f}s")
        print(f"  - Optimization time: {optimization_time:.2f}s")
        print(f"  - Total time: {bundle_time + optimization_time:.2f}s")
        print(f"  - Parameter estimation error: {np.linalg.norm(theta_hat - theta_0):.4f}")
        

if __name__ == "__main__":
    run_quad_supermod_experiment() 