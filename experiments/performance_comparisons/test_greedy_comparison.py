#!/usr/bin/env python3
"""
Test comparison between standard and 1slack formulations for Greedy subproblem.
Based on test_estimation_row_generation_1slack_greedy.py but compares both formulations.
"""

import numpy as np
import time
from mpi4py import MPI
from bundlechoice.core import BundleChoice
from bundlechoice.estimation import RowGeneration1SlackSolver, RowGenerationSolver

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

def test_greedy_comparison():
    """Test comparison between standard and 1slack formulations for Greedy subproblem."""
    num_agents = 250
    num_items = 50
    num_features = 6
    num_simuls = 1
    
    cfg = {
        "dimensions": {
            "num_agents": num_agents,
            "num_items": num_items,
            "num_features": num_features,
            "num_simuls": num_simuls,
        },
        "subproblem": {
            "name": "Greedy",
        },
        "row_generation": {
            "max_iters": 100,
            "tolerance_optimality": 0.001,
            "min_iters": 1,
            "gurobi_settings": {
                "OutputFlag": 0
            }
        }
    }
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print(f"\n{'='*80}")
        print(f"GREEDY COMPARISON: {num_agents} agents, {num_items} items")
        print(f"{'='*80}")
    
    # Generate data on rank 0
    if rank == 0:
        np.random.seed(42)  # Fixed seed for reproducibility
        modular = np.random.normal(0, 1, (num_agents, num_items, num_features-1))
        errors = np.random.normal(0, 1, size=(num_simuls, num_agents, num_items)) 
        agent_data = {"modular": modular}
        input_data = {"agent_data": agent_data, "errors": errors}
    else:
        input_data = None

    # Initialize BundleChoice
    greedy_demo = BundleChoice()
    greedy_demo.load_config(cfg)
    greedy_demo.data.load_and_scatter(input_data)
    greedy_demo.features.set_oracle(features_oracle)

    # Simulate theta_0 and generate observed bundles
    theta_0 = np.ones(num_features)
    observed_bundles = greedy_demo.subproblems.init_and_solve(theta_0)

    # Prepare data for estimation
    if rank == 0:
        print(f"Aggregate demands: {observed_bundles.sum(1).min()}, {observed_bundles.sum(1).max()}")
        print(f"Total aggregate: {observed_bundles.sum()}")
        input_data["obs_bundle"] = observed_bundles
        input_data["errors"] = np.random.normal(0, 1, size=(num_simuls, num_agents, num_items))
    else:
        input_data = None

    # Test Standard Formulation
    if rank == 0:
        print(f"\n{'─'*50}")
        print("TESTING STANDARD FORMULATION")
        print(f"{'─'*50}")

    bc_standard = BundleChoice()
    bc_standard.load_config(cfg)
    bc_standard.data.load_and_scatter(input_data)
    bc_standard.features.set_oracle(features_oracle)
    bc_standard.subproblems.load()
    
    tic = time.time()
    solver_standard = RowGenerationSolver(
        comm_manager=bc_standard.comm_manager,
        dimensions_cfg=bc_standard.config.dimensions,
        row_generation_cfg=bc_standard.config.row_generation,
        data_manager=bc_standard.data_manager,
        feature_manager=bc_standard.feature_manager,
        subproblem_manager=bc_standard.subproblem_manager
    )
    theta_hat_standard = solver_standard.solve()
    toc = time.time()
    
    if rank == 0:
        standard_time = toc - tic
        standard_obj = solver_standard.master_model.objVal if solver_standard.master_model else None
        standard_constraints = solver_standard.master_model.NumConstrs if solver_standard.master_model else 0
        standard_iterations = getattr(solver_standard, 'iteration', 'N/A')
        
        print(f"Standard - Estimated parameters: {theta_hat_standard}")
        print(f"Standard - Parameter error (L2): {np.linalg.norm(theta_hat_standard - theta_0):.6f}")
        print(f"Standard - Max relative error: {np.max(np.abs(theta_hat_standard - theta_0) / (np.abs(theta_0) + 1e-8)):.6f}")
        print(f"Standard - Solve time: {standard_time:.3f}s")
        print(f"Standard - Final objective: {standard_obj:.6f}")
        print(f"Standard - Iterations: {standard_iterations}")
        print(f"Standard - Final constraints: {standard_constraints}")

    # Test 1slack Formulation
    if rank == 0:
        print(f"\n{'─'*50}")
        print("TESTING 1SLACK FORMULATION")
        print(f"{'─'*50}")

    bc_1slack = BundleChoice()
    bc_1slack.load_config(cfg)
    bc_1slack.data.load_and_scatter(input_data)
    bc_1slack.features.set_oracle(features_oracle)
    bc_1slack.subproblems.load()
    
    tic = time.time()
    solver_1slack = RowGeneration1SlackSolver(
        comm_manager=bc_1slack.comm_manager,
        dimensions_cfg=bc_1slack.config.dimensions,
        row_generation_cfg=bc_1slack.config.row_generation,
        data_manager=bc_1slack.data_manager,
        feature_manager=bc_1slack.feature_manager,
        subproblem_manager=bc_1slack.subproblem_manager
    )
    theta_hat_1slack = solver_1slack.solve()
    toc = time.time()
    
    if rank == 0:
        slack_time = toc - tic
        slack_obj = solver_1slack.master_model.objVal if solver_1slack.master_model else None
        slack_constraints = solver_1slack.master_model.NumConstrs if solver_1slack.master_model else 0
        slack_iterations = getattr(solver_1slack, 'iteration', 'N/A')
        
        print(f"1slack - Estimated parameters: {theta_hat_1slack}")
        print(f"1slack - Parameter error (L2): {np.linalg.norm(theta_hat_1slack - theta_0):.6f}")
        print(f"1slack - Max relative error: {np.max(np.abs(theta_hat_1slack - theta_0) / (np.abs(theta_0) + 1e-8)):.6f}")
        print(f"1slack - Solve time: {slack_time:.3f}s")
        print(f"1slack - Final objective: {slack_obj:.6f}")
        print(f"1slack - Iterations: {slack_iterations}")
        print(f"1slack - Final constraints: {slack_constraints}")

    # Summary comparison
    if rank == 0:
        print(f"\n{'='*80}")
        print("GREEDY COMPARISON RESULTS")
        print(f"{'='*80}")
        
        if standard_obj is not None and slack_obj is not None:
            abs_diff = abs(slack_obj - standard_obj)
            rel_diff = (abs_diff / abs(standard_obj)) * 100 if standard_obj != 0 else 0
            
            print(f"Standard objective: {standard_obj:.6f}")
            print(f"1slack objective:   {slack_obj:.6f}")
            print(f"Absolute difference: {abs_diff:.8f}")
            print(f"Relative difference: {rel_diff:.4f}%")
            print(f"Time ratio (1slack/std): {slack_time/standard_time:.2f}x")
            print(f"Constraint ratio (1slack/std): {slack_constraints/standard_constraints:.4f}x")
            
            if rel_diff < 0.001:
                print("✅ MATHEMATICAL EQUIVALENCE ACHIEVED!")
            else:
                print("❌ Mathematical equivalence NOT achieved")
        else:
            print("❌ Missing objective values for comparison")
        
        print(f"{'='*80}")

if __name__ == "__main__":
    test_greedy_comparison()
