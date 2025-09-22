#!/usr/bin/env python3
"""
Minimal debug script for 1slack formulation - just run 2 iterations
"""

import numpy as np
from mpi4py import MPI
from bundlechoice import BundleChoice
from bundlechoice.estimation import RowGeneration1SlackSolver

def features_oracle(i_id, B_j, data):
    """Compute features for a given agent and bundle(s)."""
    modular_agent = data["agent_data"]["modular"][i_id]
    modular_agent = np.atleast_2d(modular_agent)

    single_bundle = False
    if B_j.ndim == 1:
        B_j = B_j[:, None]
        single_bundle = True
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        agent_sum = modular_agent.T @ B_j
    neg_sq = -np.sum(B_j, axis=0, keepdims=True) ** 2

    features = np.vstack((agent_sum, neg_sq))
    if single_bundle:
        return features[:, 0]
    return features

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("🔬 MINIMAL DEBUG 1SLACK FORMULATION")
        print("=" * 50)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # True parameters
    theta_true = np.ones(6)
    if rank == 0:
        print(f"True parameters (θ₀): {theta_true}")
    
    # Generate data
    num_agents = 10  # Smaller for debugging
    num_items = 5
    num_simuls = 5
    
    # Generate modular preferences
    modular = np.random.uniform(0.1, 1.0, (num_agents, num_items))
    
    # Generate weights and capacity for linear_knapsack
    weights = np.random.uniform(0.1, 1.0, num_items)
    capacity = np.random.uniform(2.0, 5.0, num_agents)
    
    # Generate errors
    errors = np.random.normal(0, 0.1, (num_agents, num_items))
    
    # Create input data
    agent_data = {"modular": modular, "capacity": capacity}
    item_data = {"weights": weights}
    input_data = {"agent_data": agent_data, "item_data": item_data, "errors": errors}
    
    if rank == 0:
        print("Generating observed bundles...")
    
    # Configuration for generation
    config_gen = {
        "dimensions": {
            "num_agents": num_agents,
            "num_items": num_items,
            "num_features": 6,
            "num_simuls": 1,  # No simulations for generation
        },
        "subproblem": {
            "name": "linear_knapsack",
        },
        "row_generation": {
            "max_iters": 2,  # Just 2 iterations
            "tolerance_optimality": 1e-6,
        }
    }
    
    # Generate observed bundles
    bc_gen = BundleChoice()
    bc_gen.load_config(config_gen)
    bc_gen.data.load_and_scatter(input_data)
    bc_gen.features.set_oracle(features_oracle)
    bc_gen.subproblems.load()
    
    # Get observed bundles
    observed_bundles = []
    for i in range(num_agents):
        bundle = bc_gen.subproblems.solve(i, theta_true)
        observed_bundles.append(bundle)
    observed_bundles = np.array(observed_bundles)
    
    if rank == 0:
        print(f"Generated {len(observed_bundles)} observed bundles")
    
    # Configuration for estimation
    config_est = {
        "dimensions": {
            "num_agents": num_agents,
            "num_items": num_items,
            "num_features": 6,
            "num_simuls": num_simuls,
        },
        "subproblem": {
            "name": "linear_knapsack",
        },
        "row_generation": {
            "max_iters": 2,  # Just 2 iterations
            "tolerance_optimality": 1e-6,
        },
        "observed_bundles": observed_bundles,
    }
    
    if rank == 0:
        print("\n1SLACK FORMULATION")
        print("-" * 30)
    
    # 1slack formulation
    bc_est = BundleChoice()
    bc_est.load_config(config_est)
    bc_est.data.load_and_scatter(input_data)
    bc_est.features.set_oracle(features_oracle)
    bc_est.subproblems.load()
    
    # Initialize solver
    solver = RowGeneration1SlackSolver(
        comm_manager=bc_est.comm_manager,
        dimensions_cfg=bc_est.config.dimensions,
        row_generation_cfg=bc_est.config.row_generation,
        data_manager=bc_est.data_manager,
        feature_manager=bc_est.feature_manager,
        subproblem_manager=bc_est.subproblem_manager
    )
    
    if rank == 0:
        print("Starting 1slack estimation...")
    
    # Run estimation
    try:
        theta_est = solver.solve()
        if rank == 0:
            print(f"Estimated parameters: {theta_est}")
            print(f"True parameters:     {theta_true}")
            print(f"Difference:          {np.abs(theta_est - theta_true)}")
    except Exception as e:
        if rank == 0:
            print(f"Error in 1slack estimation: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
