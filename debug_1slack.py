#!/usr/bin/env python3

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
        print("🔬 DEBUG 1SLACK FORMULATION")
        print("=" * 50)
    
    # Problem setup
    num_agents = 50
    num_items = 10
    num_features = 6
    num_simuls = 25
    
    # True parameters
    theta_0 = np.ones(num_features)
    if rank == 0:
        print(f"True parameters (θ₀): {theta_0}")
    
    # Generate data
    if rank == 0:
        modular = np.random.normal(0, 1, (num_agents, num_items, num_features-1))
        errors = np.random.normal(0, 1, size=(1, num_agents, num_items))  # 1 simulation for generation
        
        # For linear_knapsack, we need weights and capacity
        weights = np.random.uniform(0.1, 1.0, num_items)  # Random weights for items
        capacity = np.random.uniform(2.0, 5.0, num_agents)  # Random capacity for agents
        
        agent_data = {
            "modular": modular,
            "capacity": capacity
        }
        item_data = {
            "weights": weights
        }
        input_data = {
            "agent_data": agent_data, 
            "item_data": item_data,
            "errors": errors
        }
    else:
        input_data = None
    
    # Configuration for bundle generation
    config_gen = {
        "dimensions": {
            "num_agents": num_agents,
            "num_items": num_items,
            "num_features": num_features,
            "num_simuls": 1,  # Use 1 for bundle generation
        },
        "subproblem": {
            "name": "LinearKnapsack",
        },
        "row_generation": {
            "max_iters": 10,  # Reduced for debugging
            "tolerance_optimality": 1e-10,
            "min_iters": 1,
            "gurobi_settings": {
                "OutputFlag": 1  # Enable Gurobi output for debugging
            }
        }
    }
    
    # Generate observed bundles
    if rank == 0:
        print("Generating observed bundles...")
    
    bc_gen = BundleChoice()
    bc_gen.load_config(config_gen)
    bc_gen.data.load_and_scatter(input_data)
    
    # Set up features oracle
    bc_gen.features.set_oracle(features_oracle)
    
    observed_bundles = bc_gen.subproblems.init_and_solve(theta_0)
    
    # Prepare data for estimation with full simulations
    if rank == 0:
        input_data["obs_bundle"] = observed_bundles
        input_data["errors"] = np.random.normal(0, 1, size=(num_simuls, num_agents, num_items))
    
    # Configuration for estimation
    config_est = {
        "dimensions": {
            "num_agents": num_agents,
            "num_items": num_items,
            "num_features": num_features,
            "num_simuls": num_simuls,
        },
        "subproblem": {
            "name": "LinearKnapsack",
        },
        "row_generation": {
            "max_iters": 10,  # Reduced for debugging
            "tolerance_optimality": 1e-10,
            "min_iters": 1,
            "gurobi_settings": {
                "OutputFlag": 1  # Enable Gurobi output for debugging
            }
        }
    }
    
    # Run 1slack estimation
    if rank == 0:
        print("\n1SLACK FORMULATION")
        print("-" * 30)
    
    bc_est = BundleChoice()
    bc_est.load_config(config_est)
    bc_est.data.load_and_scatter(input_data)
    bc_est.features.set_oracle(features_oracle)
    bc_est.subproblems.load()
    
    solver = RowGeneration1SlackSolver(
        comm_manager=bc_est.comm_manager,
        dimensions_cfg=bc_est.config.dimensions,
        row_generation_cfg=bc_est.config.row_generation,
        data_manager=bc_est.data_manager,
        feature_manager=bc_est.feature_manager,
        subproblem_manager=bc_est.subproblem_manager
    )
    
    try:
        theta_hat = solver.solve()
        
        if rank == 0:
            print(f"Estimated parameters: {theta_hat}")
            print(f"True parameters:     {theta_0}")
            print(f"L2 error: {np.linalg.norm(theta_hat - theta_0):.6f}")
            print(f"Max relative error: {np.max(np.abs(theta_hat - theta_0) / (np.abs(theta_0) + 1e-8)):.6f}")
            
    except Exception as e:
        if rank == 0:
            print(f"❌ 1slack formulation failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
