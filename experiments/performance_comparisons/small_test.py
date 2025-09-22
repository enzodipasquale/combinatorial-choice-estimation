#!/usr/bin/env python3
"""
Small test to verify mathematical equivalence between standard and 1slack formulations
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
    
    features = np.vstack((agent_sum, neg_sq))  # (2, num_bundles)
    if single_bundle:
        return features[:, 0]
    return features

def run_small_test(subproblem_type, num_agents=5, num_items=5):
    """Run a small test for the given subproblem type."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Generate medium-scale synthetic data
    np.random.seed(42)  # Fixed seed for reproducibility
    num_features = 5  # 4-5 features as requested
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"MEDIUM TEST: {subproblem_type.upper()}")
        print(f"Agents: {num_agents}, Items: {num_items}, Features: {num_features}")
        print(f"{'='*60}")
    modular = np.random.normal(0, 1, (num_agents, num_items, num_features-1))  # num_features-1 features
    errors = np.random.normal(0, 0.1, (1, num_agents, num_items))  # 1 simulation
    theta_0 = np.ones(num_features)  # True parameters
    
    # For linear_knapsack, we need weights and capacity
    if subproblem_type == "linear_knapsack":
        weights = np.random.uniform(0.1, 1.0, num_items)
        capacity = np.random.uniform(2.0, 5.0, num_agents)
        agent_data = {"modular": modular, "capacity": capacity}
        item_data = {"weights": weights}
        input_data = {"agent_data": agent_data, "item_data": item_data, "errors": errors}
    else:
        agent_data = {"modular": modular}
        input_data = {"agent_data": agent_data, "errors": errors}
    
    # Configuration for generation
    config_gen = {
        "dimensions": {
            "num_agents": num_agents,
            "num_items": num_items,
            "num_features": num_features,
            "num_simuls": 1,
        },
        "subproblem": {
            "name": subproblem_type.title().replace('_', ''),
        },
        "row_generation": {
            "max_iters": 50,  # Reduced for small test
            "tolerance_optimality": 1e-10,
            "min_iters": 1,
            "gurobi_settings": {
                "OutputFlag": 0
            }
        }
    }
    
    # Configuration for estimation
    config_est = {
        "dimensions": {
            "num_agents": num_agents,
            "num_items": num_items,
            "num_features": num_features,
            "num_simuls": 1,
        },
        "subproblem": {
            "name": subproblem_type.title().replace('_', ''),
        },
        "row_generation": {
            "max_iters": 200 if subproblem_type == "linear_knapsack" else 50,  # More iterations for linear knapsack
            "tolerance_optimality": 1e-10,
            "min_iters": 1,
            "gurobi_settings": {
                "OutputFlag": 0
            }
        }
    }
    
    # Generate data
    bc_gen = BundleChoice()
    bc_gen.load_config(config_gen)
    bc_gen.data.load_and_scatter(input_data)
    bc_gen.features.set_oracle(features_oracle)
    bc_gen.subproblems.load()
    
    # Run generation
    start_time = time.time()
    observed_bundles = bc_gen.subproblems.init_and_solve(theta_0)
    gen_time = time.time() - start_time
    
    if rank == 0:
        print(f"Generation completed in {gen_time:.3f}s")
        print(f"Generated bundles shape: {observed_bundles.shape}")
    
    # Prepare data for estimation
    if rank == 0:
        input_data["obs_bundle"] = observed_bundles
        input_data["errors"] = np.random.normal(0, 0.1, (1, num_agents, num_items))  # 1 simulation for estimation
    else:
        input_data = None
    
    # Estimation with standard formulation
    bc_est = BundleChoice()
    bc_est.load_config(config_est)
    bc_est.data.load_and_scatter(input_data)
    bc_est.features.set_oracle(features_oracle)
    bc_est.subproblems.load()
    
    start_time = time.time()
    theta_hat_standard = bc_est.row_generation.solve()
    standard_time = time.time() - start_time
    standard_obj = bc_est.row_generation.master_model.objVal if bc_est.row_generation.master_model else None
    standard_constraints = bc_est.row_generation.master_model.NumConstrs if bc_est.row_generation.master_model else 0
    
    if rank == 0:
        print(f"Standard formulation completed in {standard_time:.3f}s")
        print(f"Standard objective: {standard_obj:.6f}")
        print(f"Standard constraints: {standard_constraints}")
    
    # Estimation with 1slack formulation
    bc_est_1slack = BundleChoice()
    bc_est_1slack.load_config(config_est)
    bc_est_1slack.data.load_and_scatter(input_data)
    bc_est_1slack.features.set_oracle(features_oracle)
    bc_est_1slack.subproblems.load()
    
    # Use 1slack solver with 1000 iterations for all subproblems
    config_1slack = config_est.copy()
    config_1slack["row_generation"]["max_iters"] = 1000
    bc_est_1slack.load_config(config_1slack)
    
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
        print(f"1slack formulation completed in {slack_time:.3f}s")
        print(f"1slack objective: {slack_obj:.6f}")
        print(f"1slack constraints: {slack_constraints}")
        
        # Calculate differences
        obj_diff = abs(standard_obj - slack_obj)
        obj_diff_pct = (obj_diff / abs(standard_obj)) * 100
        
        print(f"\n{'='*40}")
        print(f"COMPARISON RESULTS")
        print(f"{'='*40}")
        print(f"Objective difference: {obj_diff:.8f}")
        print(f"Objective difference %: {obj_diff_pct:.4f}%")
        print(f"Time ratio (1slack/std): {slack_time/standard_time:.2f}x")
        print(f"Constraint ratio (1slack/std): {slack_constraints/standard_constraints:.4f}x")
        
        if obj_diff < 1e-6:
            print("✅ MATHEMATICAL EQUIVALENCE ACHIEVED!")
        else:
            print("❌ Mathematical equivalence NOT achieved")
        print(f"{'='*40}")

def main():
    """Run comprehensive tests starting with small examples."""
    print("="*80)
    print("STEP 1: VERIFY SMALL EXAMPLES STILL WORK")
    print("="*80)
    run_small_test("greedy", num_agents=5, num_items=5)
    run_small_test("linear_knapsack", num_agents=5, num_items=5)
    run_small_test("quad_supermodular_network", num_agents=5, num_items=5)
    run_small_test("plain_single_item", num_agents=5, num_items=5)
    
    print("\n" + "="*80)
    print("STEP 2: MEDIUM EXAMPLES - LINEAR KNAPSACK FOCUS")
    print("="*80)
    run_small_test("linear_knapsack", num_agents=50, num_items=10)
    run_small_test("linear_knapsack", num_agents=100, num_items=20)

if __name__ == "__main__":
    main()
