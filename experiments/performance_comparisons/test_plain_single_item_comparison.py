#!/usr/bin/env python3
"""
Test comparison between standard and 1slack formulations for PlainSingleItem subproblem.
Based on test_estimation_row_generation_1slack_plain_single_item.py but compares both formulations.
"""

import numpy as np
import time
from mpi4py import MPI
from bundlechoice.core import BundleChoice
from bundlechoice.estimation import RowGeneration1SlackSolver, RowGenerationSolver

def test_plain_single_item_comparison():
    """Test comparison between standard and 1slack formulations for PlainSingleItem subproblem."""
    num_agents = 500
    num_items = 2
    num_modular_agent_features = 4
    num_modular_item_features = 1
    num_features = num_modular_agent_features + num_modular_item_features
    num_simuls = 1
    sigma = 1
    
    cfg = {
        "dimensions": {
            "num_agents": num_agents,
            "num_items": num_items,
            "num_features": num_features,
            "num_simuls": num_simuls
        },
        "subproblem": {
            "name": "PlainSingleItem",
            "settings": {}
        },
        "row_generation": {
            "max_iters": 100,
            "tolerance_optimality": .0001,
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
        print(f"PLAIN SINGLE ITEM COMPARISON: {num_agents} agents, {num_items} items")
        print(f"{'='*80}")
    
    # Generate data on rank 0
    if rank == 0:
        np.random.seed(42)  # Fixed seed for reproducibility
        errors = sigma * np.random.normal(0, 1, (num_agents, num_items))
        estimation_errors = sigma * np.random.normal(0, 1, (num_simuls, num_agents, num_items))
        input_data = {
            "item_data": {"modular": np.random.normal(0, 1, (num_items, num_modular_item_features))},
            "agent_data": {"modular": np.random.normal(0, 1, (num_agents, num_items, num_modular_agent_features))},
            "errors": errors,
        }
    else:
        input_data = None
        
    # Generate observed bundles
    demo = BundleChoice()
    demo.load_config(cfg)
    demo.data.load_and_scatter(input_data)
    demo.features.build_from_data()
    
    theta_0 = np.ones(num_features)
    observed_bundles = demo.subproblems.init_and_solve(theta_0)
    
    # Check that observed_bundles is not None
    if rank == 0:
        assert observed_bundles is not None, "observed_bundles is None!"
        assert input_data["errors"] is not None, "input_data['errors'] is None!"
        assert observed_bundles.shape == (num_agents, num_items)
        assert np.all(observed_bundles.sum(axis=1) <= 1), "Each agent should select at most one item."
        
        modular_agent = input_data["agent_data"]["modular"]
        modular_item = input_data["item_data"]["modular"]
        errors = input_data["errors"]
        agent_util = np.einsum('aij,j->ai', modular_agent, theta_0[:num_modular_agent_features])
        item_util = np.dot(modular_item, theta_0[num_modular_agent_features:])
        total_util = agent_util + item_util + errors
        j_star = np.argmax(total_util, axis=1)
        for i in range(num_agents):
            if observed_bundles[i, :].sum() == 1:
                assert observed_bundles[i, j_star[i]] == 1, f"Agent {i} did not select the max utility item."
            else:
                assert np.all(total_util[i, :] <= 0), f"Agent {i} made no selection, but has positive utility: {total_util[i, :]}"
        
        input_data["obs_bundle"] = observed_bundles
        input_data["errors"] = estimation_errors
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
    bc_standard.features.build_from_data()
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
    bc_1slack.features.build_from_data()
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
        print("PLAIN SINGLE ITEM COMPARISON RESULTS")
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
    test_plain_single_item_comparison()
