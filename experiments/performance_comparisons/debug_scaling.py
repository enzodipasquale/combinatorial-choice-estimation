#!/usr/bin/env python3
"""
Debug script to test linear knapsack 1slack formulation scaling.
"""

import time
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

def test_scaling(num_agents, num_items, num_features=5):
    """Test linear knapsack with given dimensions."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"TESTING: {num_agents} agents, {num_items} items, {num_features} features")
        print(f"{'='*60}")
    
    # Generate data
    np.random.seed(42)
    modular = np.random.normal(0, 1, (num_agents, num_items, num_features-1))
    errors = np.random.normal(0, 0.1, (1, num_agents, num_items))
    theta_0 = np.ones(num_features)
    
    # For linear_knapsack, we need weights and capacity
    weights = np.random.uniform(0.1, 1.0, num_items)
    capacity = np.random.uniform(2.0, 5.0, num_agents)
    agent_data = {"modular": modular, "capacity": capacity}
    item_data = {"weights": weights}
    input_data = {"agent_data": agent_data, "item_data": item_data, "errors": errors}
    
    # Configuration
    config = {
        "dimensions": {
            "num_agents": num_agents,
            "num_items": num_items,
            "num_features": num_features,
            "num_simuls": 1,
        },
        "subproblem": {
            "name": "LinearKnapsack",
        },
        "row_generation": {
            "max_iters": 100,
            "tolerance_optimality": 1e-10,
            "min_iters": 1,
            "gurobi_settings": {
                "OutputFlag": 0
            }
        }
    }
    
    # Generate data
    bc_gen = BundleChoice()
    bc_gen.load_config(config)
    bc_gen.data.load_and_scatter(input_data)
    bc_gen.features.set_oracle(features_oracle)
    bc_gen.subproblems.load()
    
    # Run generation
    observed_bundles = bc_gen.subproblems.init_and_solve(theta_0)
    
    if rank == 0:
        print(f"Generated bundles shape: {observed_bundles.shape}")
    
    # Prepare data for estimation
    if rank == 0:
        input_data["obs_bundle"] = observed_bundles
        input_data["errors"] = np.random.normal(0, 0.1, (1, num_agents, num_items))
    else:
        input_data = None
    
    # Standard formulation
    bc_est = BundleChoice()
    bc_est.load_config(config)
    bc_est.data.load_and_scatter(input_data)
    bc_est.features.set_oracle(features_oracle)
    bc_est.subproblems.load()
    
    start_time = time.time()
    theta_hat_standard = bc_est.row_generation.solve()
    standard_time = time.time() - start_time
    standard_obj = bc_est.row_generation.master_model.objVal if bc_est.row_generation.master_model else None
    standard_constraints = bc_est.row_generation.master_model.NumConstrs if bc_est.row_generation.master_model else 0
    
    # 1slack formulation
    bc_est_1slack = BundleChoice()
    bc_est_1slack.load_config(config)
    bc_est_1slack.data.load_and_scatter(input_data)
    bc_est_1slack.features.set_oracle(features_oracle)
    bc_est_1slack.subproblems.load()
    
    solver_1slack = RowGeneration1SlackSolver(
        comm_manager=bc_est_1slack.comm_manager,
        dimensions_cfg=bc_est_1slack.config.dimensions,
        row_generation_cfg=bc_est_1slack.config.row_generation,
        data_manager=bc_est_1slack.data_manager,
        feature_manager=bc_est_1slack.feature_manager,
        subproblem_manager=bc_est_1slack.subproblem_manager
    )
    
    start_time = time.time()
    theta_hat_1slack = solver_1slack.solve()
    slack_time = time.time() - start_time
    slack_obj = solver_1slack.master_model.objVal if solver_1slack.master_model else None
    slack_constraints = solver_1slack.master_model.NumConstrs if solver_1slack.master_model else 0
    
    if rank == 0:
        print(f"Standard: {standard_obj:.6f} ({standard_time:.3f}s, {standard_constraints} constraints)")
        print(f"1slack:   {slack_obj:.6f} ({slack_time:.3f}s, {slack_constraints} constraints)")
        
        if standard_obj is not None and slack_obj is not None:
            abs_diff = abs(slack_obj - standard_obj)
            rel_diff = (abs_diff / abs(standard_obj)) * 100 if standard_obj != 0 else 0
            print(f"Difference: {abs_diff:.6f} ({rel_diff:.4f}%)")
            
            if rel_diff < 0.01:
                print("✅ PERFECT EQUIVALENCE")
            elif rel_diff < 1.0:
                print("✅ GOOD EQUIVALENCE")
            elif rel_diff < 10.0:
                print("⚠️  POOR EQUIVALENCE")
            else:
                print("❌ FAILED EQUIVALENCE")
        else:
            print("❌ MISSING OBJECTIVE VALUES")
        print()

def main():
    """Test scaling from small to medium instances."""
    # Test small instances first
    test_scaling(5, 5)
    test_scaling(10, 5)
    test_scaling(20, 5)
    test_scaling(50, 5)
    test_scaling(100, 5)
    test_scaling(200, 5)

if __name__ == "__main__":
    main()

