#!/usr/bin/env python3
"""
Plain single item experiment only - 2 medium-scale configurations.
"""

import time
import numpy as np
from mpi4py import MPI
from bundlechoice import BundleChoice
from bundlechoice.estimation import RowGeneration1SlackSolver

def run_plain_single_item_experiment(num_agents, num_items):
    """Run plain single item experiment with exact test data generation."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print(f"\n{'='*80}")
        print(f"PLAIN SINGLE ITEM EXPERIMENT: {num_agents} agents, {num_items} items")
        print(f"{'='*80}")
    
    num_modular_agent_features = 4
    num_modular_item_features = 1
    num_features = num_modular_agent_features + num_modular_item_features
    num_simuls = 1
    sigma = 1
    
    # Configuration - matches test exactly
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
            "max_iters": 1000,  # Large number to avoid early termination
            "tolerance_optimality": 1e-10,
            "min_iters": 1,
            "gurobi_settings": {
                "OutputFlag": 0
            }
        }
    }
    
    # Generate data on rank 0 - matches test exactly
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
    bc_gen = BundleChoice()
    bc_gen.load_config(cfg)
    bc_gen.data.load_and_scatter(input_data)
    bc_gen.features.build_from_data()
    bc_gen.subproblems.load()
    
    theta_0 = np.ones(num_features)
    observed_bundles = bc_gen.subproblems.init_and_solve(theta_0)
    
    if rank == 0 and observed_bundles is not None:
        print(f"Generated bundles shape: {observed_bundles.shape}")
        assert observed_bundles.shape == (num_agents, num_items)
        assert np.all(observed_bundles.sum(axis=1) <= 1), "Each agent should select at most one item."
        input_data["obs_bundle"] = observed_bundles
        input_data["errors"] = estimation_errors
    else:
        input_data = None

    # Standard formulation
    bc_std = BundleChoice()
    bc_std.load_config(cfg)
    bc_std.data.load_and_scatter(input_data)
    bc_std.features.build_from_data()
    bc_std.subproblems.load()
    
    start_time = time.time()
    theta_hat_std = bc_std.row_generation.solve()
    std_time = time.time() - start_time
    std_obj = bc_std.row_generation.master_model.objVal if bc_std.row_generation.master_model else None
    std_constraints = bc_std.row_generation.master_model.NumConstrs if bc_std.row_generation.master_model else 0
    
    if rank == 0:
        print(f"Standard formulation completed in {std_time:.3f}s")
        print(f"Standard objective: {std_obj:.6f}")
        print(f"Standard constraints: {std_constraints}")

    # 1slack formulation
    bc_1slack = BundleChoice()
    bc_1slack.load_config(cfg)
    bc_1slack.data.load_and_scatter(input_data)
    bc_1slack.features.build_from_data()
    bc_1slack.subproblems.load()
    
    solver_1slack = RowGeneration1SlackSolver(
        comm_manager=bc_1slack.comm_manager,
        dimensions_cfg=bc_1slack.config.dimensions,
        row_generation_cfg=bc_1slack.config.row_generation,
        data_manager=bc_1slack.data_manager,
        feature_manager=bc_1slack.feature_manager,
        subproblem_manager=bc_1slack.subproblem_manager
    )
    
    start_time = time.time()
    theta_hat_1slack = solver_1slack.solve()
    slack_time = time.time() - start_time
    slack_obj = solver_1slack.master_model.objVal if solver_1slack.master_model else None
    slack_constraints = solver_1slack.master_model.NumConstrs if solver_1slack.master_model else None
    
    if rank == 0:
        print(f"1slack formulation completed in {slack_time:.3f}s")
        print(f"1slack objective: {slack_obj:.6f}")
        print(f"1slack constraints: {slack_constraints}")
        
        if std_obj is not None and slack_obj is not None:
            abs_diff = abs(std_obj - slack_obj)
            rel_diff = (abs_diff / abs(std_obj)) * 100 if std_obj != 0 else 0
            print(f"\n{'='*50}")
            print(f"PLAIN SINGLE ITEM COMPARISON RESULTS")
            print(f"{'='*50}")
            print(f"Standard objective: {std_obj:.6f}")
            print(f"1slack objective: {slack_obj:.6f}")
            print(f"Absolute difference: {abs_diff:.8f}")
            print(f"Relative difference: {rel_diff:.4f}%")
            print(f"Time ratio (1slack/std): {slack_time/std_time:.2f}x")
            if slack_constraints is not None:
                print(f"Constraint ratio (1slack/std): {slack_constraints/std_constraints:.4f}x")
            if rel_diff < 0.001:
                print("✅ MATHEMATICAL EQUIVALENCE ACHIEVED!")
            else:
                print("❌ Mathematical equivalence NOT achieved")
            print(f"{'='*50}")

def main():
    """Run plain single item experiments for both medium-scale configurations."""
    print("="*80)
    print("PLAIN SINGLE ITEM EXPERIMENTS: Standard vs 1slack Formulations")
    print("="*80)
    
    # Medium-scale experiment 1: 50 items, 800 agents
    run_plain_single_item_experiment(num_agents=800, num_items=50)
    
    # Medium-scale experiment 2: 200 items, 300 agents
    run_plain_single_item_experiment(num_agents=300, num_items=200)

if __name__ == "__main__":
    main()




