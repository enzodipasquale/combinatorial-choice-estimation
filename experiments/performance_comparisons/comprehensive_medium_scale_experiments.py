#!/usr/bin/env python3
"""
Comprehensive medium-scale experiments comparing standard and 1slack formulations.
Includes all 4 subproblem types with exact test data generation processes.
"""

import time
import numpy as np
from mpi4py import MPI
from bundlechoice import BundleChoice
from bundlechoice.estimation import RowGeneration1SlackSolver

def features_oracle(i_id, B_j, data):
    """Compute features for a given agent and bundle(s) - matches test files exactly."""
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

def run_greedy_experiment(num_agents, num_items):
    """Run greedy experiment with exact test data generation."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print(f"\n{'='*80}")
        print(f"GREEDY EXPERIMENT: {num_agents} agents, {num_items} items")
        print(f"{'='*80}")
    
    num_features = 6
    num_simuls = 1
    
    # Configuration - matches test exactly
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
        modular = np.random.normal(0, 1, (num_agents, num_items, num_features-1))
        errors = np.random.normal(0, 1, size=(num_simuls, num_agents, num_items)) 
        agent_data = {"modular": modular}
        input_data = {"agent_data": agent_data, "errors": errors}
    else:
        input_data = None

    # Generate observed bundles
    bc_gen = BundleChoice()
    bc_gen.load_config(cfg)
    bc_gen.data.load_and_scatter(input_data)
    bc_gen.features.set_oracle(features_oracle)
    bc_gen.subproblems.load()
    
    theta_0 = np.ones(num_features)
    observed_bundles = bc_gen.subproblems.init_and_solve(theta_0)
    
    if rank == 0:
        print(f"Generated bundles shape: {observed_bundles.shape}")
        print(f"Aggregate demands: {observed_bundles.sum(1).min()}, {observed_bundles.sum(1).max()}")
        input_data["obs_bundle"] = observed_bundles
        input_data["errors"] = np.random.normal(0, 1, size=(num_simuls, num_agents, num_items))
    else:
        input_data = None

    # Standard formulation
    bc_std = BundleChoice()
    bc_std.load_config(cfg)
    bc_std.data.load_and_scatter(input_data)
    bc_std.features.set_oracle(features_oracle)
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
    bc_1slack.features.set_oracle(features_oracle)
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
    slack_constraints = solver_1slack.master_model.NumConstrs if solver_1slack.master_model else 0
    
    if rank == 0:
        print(f"1slack formulation completed in {slack_time:.3f}s")
        print(f"1slack objective: {slack_obj:.6f}")
        print(f"1slack constraints: {slack_constraints}")
        
        if std_obj is not None and slack_obj is not None:
            abs_diff = abs(std_obj - slack_obj)
            rel_diff = (abs_diff / abs(std_obj)) * 100 if std_obj != 0 else 0
            print(f"\n{'='*50}")
            print(f"GREEDY COMPARISON RESULTS")
            print(f"{'='*50}")
            print(f"Standard objective: {std_obj:.6f}")
            print(f"1slack objective: {slack_obj:.6f}")
            print(f"Absolute difference: {abs_diff:.8f}")
            print(f"Relative difference: {rel_diff:.4f}%")
            print(f"Time ratio (1slack/std): {slack_time/std_time:.2f}x")
            print(f"Constraint ratio (1slack/std): {slack_constraints/std_constraints:.4f}x")
            if rel_diff < 0.001:
                print("✅ MATHEMATICAL EQUIVALENCE ACHIEVED!")
            else:
                print("❌ Mathematical equivalence NOT achieved")
            print(f"{'='*50}")

def run_linear_knapsack_experiment(num_agents, num_items):
    """Run linear knapsack experiment with exact test data generation."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print(f"\n{'='*80}")
        print(f"LINEAR KNAPSACK EXPERIMENT: {num_agents} agents, {num_items} items")
        print(f"{'='*80}")
    
    num_modular_agent_features = 2
    num_modular_item_features = 2
    num_features = num_modular_agent_features + num_modular_item_features
    num_simuls = 1
    
    # Configuration - matches test exactly
    cfg = {
        "dimensions": {
            "num_agents": num_agents,
            "num_items": num_items,
            "num_features": num_features,
            "num_simuls": num_simuls
        },
        "subproblem": {
            "name": "LinearKnapsack",
            "settings": {"TimeLimit": 10, "MIPGap_tol": 0.01}
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
        input_data = {
            "item_data": {
                "modular": np.abs(np.random.normal(0, 1, (num_items, num_modular_item_features))),
                "weights": np.random.randint(1, 10, size=num_items)
            },
            "agent_data": {
                "modular": np.abs(np.random.normal(0, 1, (num_agents, num_items, num_modular_agent_features))),
                "capacity": np.random.randint(1, 100, size=num_agents)
            },
            "errors": np.random.normal(0, 1, (num_simuls, num_agents, num_items)),
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
        print(f"Total demand: {observed_bundles.sum(1).min()}, {observed_bundles.sum(1).max()}")
        input_data["obs_bundle"] = observed_bundles
        input_data["errors"] = np.random.normal(0, 1, (num_simuls, num_agents, num_items))
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
            print(f"LINEAR KNAPSACK COMPARISON RESULTS")
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

def run_quad_supermodular_experiment(num_agents, num_items):
    """Run quadratic supermodular experiment with exact test data generation."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print(f"\n{'='*80}")
        print(f"QUAD SUPERMODULAR EXPERIMENT: {num_agents} agents, {num_items} items")
        print(f"{'='*80}")
    
    num_modular_agent_features = 2
    num_modular_item_features = 2
    num_quadratic_agent_features = 0
    num_quadratic_item_features = 2
    num_features = num_modular_agent_features + num_modular_item_features + num_quadratic_agent_features + num_quadratic_item_features
    num_simuls = 1
    
    # Configuration - matches test exactly
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
        agent_modular = -2 * np.abs(np.random.normal(2, 1, (num_agents, num_items, num_modular_agent_features)))
        item_modular = -2 * np.abs(np.random.normal(2, 1, (num_items, num_modular_item_features)))
        item_quadratic = 1 * np.exp(-np.abs(np.random.normal(0, 1, (num_items, num_items, num_quadratic_item_features))))
        
        for k in range(num_quadratic_item_features):
            np.fill_diagonal(item_quadratic[:, :, k], 0)
            # Multiply by binary matrix with density .1
            item_quadratic[:, :, k] *= (np.random.rand(num_items, num_items) < .3)

        input_data = {
            "item_data": {
                "modular": item_modular,
                "quadratic": item_quadratic
            },
            "agent_data": {
                "modular": agent_modular,
            },
            "errors": 5 * np.random.normal(0, 1, (num_simuls, num_agents, num_items)),
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
        total_demand = observed_bundles.sum(1)
        print(f"Demand range: {total_demand.min()}, {total_demand.max()}")
        input_data["obs_bundle"] = observed_bundles
        input_data["errors"] = 5 * np.random.normal(0, 1, (num_simuls, num_agents, num_items))
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
            print(f"QUAD SUPERMODULAR COMPARISON RESULTS")
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
    """Run comprehensive medium-scale experiments."""
    print("="*80)
    print("COMPREHENSIVE MEDIUM-SCALE EXPERIMENTS")
    print("Standard vs 1slack Formulations - All 4 Subproblem Types")
    print("="*80)
    
    # Medium-scale experiment 1: 50 items, 800 agents
    print("\n" + "="*80)
    print("MEDIUM-SCALE EXPERIMENT 1: 50 items, 800 agents")
    print("="*80)
    
    run_greedy_experiment(num_agents=800, num_items=50)
    run_linear_knapsack_experiment(num_agents=800, num_items=50)
    run_quad_supermodular_experiment(num_agents=800, num_items=50)
    run_plain_single_item_experiment(num_agents=800, num_items=50)
    
    # Medium-scale experiment 2: 200 items, 300 agents
    print("\n" + "="*80)
    print("MEDIUM-SCALE EXPERIMENT 2: 200 items, 300 agents")
    print("="*80)
    
    run_greedy_experiment(num_agents=300, num_items=200)
    run_linear_knapsack_experiment(num_agents=300, num_items=200)
    run_quad_supermodular_experiment(num_agents=300, num_items=200)
    run_plain_single_item_experiment(num_agents=300, num_items=200)

if __name__ == "__main__":
    main()
