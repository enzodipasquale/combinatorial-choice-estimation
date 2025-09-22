#!/usr/bin/env python3
"""
Clear comparison between standard and 1slack row generation formulations
Shows parameters and objective values side by side for easy comparison.
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

def run_clear_comparison(subproblem_type, num_agents=50, num_items=10, num_features=6, num_simuls=25):
    """Run clear comparison for a single subproblem type."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print(f"\n{'='*100}")
        print(f"COMPARISON: {subproblem_type.upper()} FORMULATION")
        print(f"{'='*100}")
        print(f"Problem size: {num_agents} agents, {num_items} items, {num_features} features, {num_simuls} simulations")
        print(f"True parameters (θ₀): {np.ones(num_features)}")
        print()
    
    # Configuration for bundle generation (use 1 simulation)
    config_gen = {
        "dimensions": {
            "num_agents": num_agents,
            "num_items": num_items,
            "num_features": num_features,
            "num_simuls": 1,  # Use 1 for bundle generation
        },
        "subproblem": {
            "name": subproblem_type.title().replace('_', ''),
        },
        "row_generation": {
            "max_iters": 500,  # Increased for better convergence
            "tolerance_optimality": 1e-10,
            "min_iters": 1,
            "gurobi_settings": {
                "OutputFlag": 0
            }
        }
    }
    
    # Configuration for estimation (use full simulations)
    config_est = {
        "dimensions": {
            "num_agents": num_agents,
            "num_items": num_items,
            "num_features": num_features,
            "num_simuls": num_simuls,
        },
        "subproblem": {
            "name": subproblem_type.title().replace('_', ''),
        },
        "row_generation": {
            "max_iters": 500,  # Increased for better convergence
            "tolerance_optimality": 1e-10,
            "min_iters": 1,
            "gurobi_settings": {
                "OutputFlag": 0
            }
        }
    }
    
    # True parameters
    theta_0 = np.ones(num_features)
    
    # Generate data
    if rank == 0:
        modular = np.random.normal(0, 1, (num_agents, num_items, num_features-1))
        errors = np.random.normal(0, 1, size=(1, num_agents, num_items))  # 1 simulation for generation
        
        # For linear_knapsack, we need weights and capacity
        if subproblem_type == "linear_knapsack":
            weights = np.random.uniform(0.1, 1.0, num_items)  # Random weights for items
            capacity = np.random.uniform(2.0, 5.0, num_agents)  # Random capacity for each agent
            agent_data = {"modular": modular, "capacity": capacity}
            item_data = {"weights": weights}
            input_data = {"agent_data": agent_data, "item_data": item_data, "errors": errors}
        else:
            agent_data = {"modular": modular}
            input_data = {"agent_data": agent_data, "errors": errors}
    else:
        input_data = None
    
    # Test Standard Formulation
    if rank == 0:
        print("STANDARD FORMULATION")
        print("-" * 50)
    
    try:
        # Generate observed bundles with 1 simulation
        bc_gen = BundleChoice()
        bc_gen.load_config(config_gen)
        bc_gen.data.load_and_scatter(input_data)
        bc_gen.features.set_oracle(features_oracle)
        
        observed_bundles = bc_gen.subproblems.init_and_solve(theta_0)
        
        # Prepare data for estimation with full simulations
        if rank == 0:
            input_data["obs_bundle"] = observed_bundles
            input_data["errors"] = np.random.normal(0, 1, size=(num_simuls, num_agents, num_items))
        else:
            input_data = None
        
        # Run standard estimation
        bc_standard = BundleChoice()
        bc_standard.load_config(config_est)
        bc_standard.data.load_and_scatter(input_data)
        bc_standard.features.set_oracle(features_oracle)
        bc_standard.subproblems.load()
        
        tic = time.time()
        theta_hat_standard = bc_standard.row_generation.solve()
        toc = time.time()
        
        # Calculate errors
        l2_error_standard = np.linalg.norm(theta_hat_standard - theta_0)
        max_rel_error_standard = np.max(np.abs(theta_hat_standard - theta_0) / (np.abs(theta_0) + 1e-8))
        
        standard_success = True
        standard_time = toc - tic
        standard_obj = bc_standard.row_generation.master_model.objVal if bc_standard.row_generation.master_model else None
        standard_iterations = getattr(bc_standard.row_generation, 'iteration', 'N/A')
        standard_constraints = bc_standard.row_generation.master_model.NumConstrs if bc_standard.row_generation.master_model else 0
        
    except Exception as e:
        if rank == 0:
            print(f"❌ Standard formulation failed: {e}")
        standard_success = False
        standard_time = float('inf')
        standard_obj = float('inf')
        standard_iterations = 0
        standard_constraints = 0
        l2_error_standard = float('inf')
        max_rel_error_standard = float('inf')
        theta_hat_standard = np.full(num_features, np.nan)
    
    # Test 1slack Formulation
    if rank == 0:
        print("\n1SLACK FORMULATION")
        print("-" * 50)
    
    try:
        # Use the same observed bundles from standard formulation
        # Prepare data for 1slack estimation with full simulations
        if rank == 0:
            input_data["obs_bundle"] = observed_bundles
            input_data["errors"] = np.random.normal(0, 1, size=(num_simuls, num_agents, num_items))
        else:
            input_data = None
        
        # Run 1slack estimation
        bc_1slack = BundleChoice()
        bc_1slack.load_config(config_est)
        bc_1slack.data.load_and_scatter(input_data)
        bc_1slack.features.set_oracle(features_oracle)
        bc_1slack.subproblems.load()
        
        # Create separate config for 1slack with 1000 iterations
        config_1slack = config_est.copy()
        config_1slack["row_generation"]["max_iters"] = 1000
        
        solver_1slack = RowGeneration1SlackSolver(
            comm_manager=bc_1slack.comm_manager,
            dimensions_cfg=bc_1slack.config.dimensions,
            row_generation_cfg=bc_1slack.config.row_generation,
            data_manager=bc_1slack.data_manager,
            feature_manager=bc_1slack.feature_manager,
            subproblem_manager=bc_1slack.subproblem_manager
        )
        
        # Override the max_iters for 1slack solver
        solver_1slack.row_generation_cfg.max_iters = 1000
        
        tic = time.time()
        theta_hat_1slack = solver_1slack.solve()
        toc = time.time()
        
        # Calculate errors
        l2_error_1slack = np.linalg.norm(theta_hat_1slack - theta_0)
        max_rel_error_1slack = np.max(np.abs(theta_hat_1slack - theta_0) / (np.abs(theta_0) + 1e-8))
        
        slack_success = True
        slack_time = toc - tic
        slack_obj = solver_1slack.master_model.objVal if solver_1slack.master_model else None
        slack_iterations = getattr(solver_1slack, 'iteration', 'N/A')
        slack_constraints = solver_1slack.master_model.NumConstrs if solver_1slack.master_model else 0
        
    except Exception as e:
        if rank == 0:
            print(f"❌ 1slack formulation failed: {e}")
        slack_success = False
        slack_time = float('inf')
        slack_obj = float('inf')
        slack_iterations = 0
        slack_constraints = 0
        l2_error_1slack = float('inf')
        max_rel_error_1slack = float('inf')
        theta_hat_1slack = np.full(num_features, np.nan)
    
    # Only rank 0 prints results
    if rank == 0:
        # Clear comparison table
        print(f"\n{'='*100}")
        print("RESULTS COMPARISON")
        print(f"{'='*100}")
        
        print(f"{'Parameter':<15} {'True (θ₀)':<12} {'Standard':<12} {'1slack':<12} {'Std Error':<12} {'1slack Error':<12}")
        print("-" * 100)
        
        for i in range(num_features):
            true_val = theta_0[i]
            std_val = theta_hat_standard[i] if standard_success else np.nan
            slack_val = theta_hat_1slack[i] if slack_success else np.nan
            std_error = abs(std_val - true_val) if standard_success else np.nan
            slack_error = abs(slack_val - true_val) if slack_success else np.nan
            
            print(f"θ{i+1:<14} {true_val:<12.3f} {std_val:<12.3f} {slack_val:<12.3f} {std_error:<12.3f} {slack_error:<12.3f}")
        
        print("\n" + "="*100)
        print("PERFORMANCE METRICS")
        print("="*100)
        
        print(f"{'Metric':<25} {'Standard':<15} {'1slack':<15} {'Difference':<15}")
        print("-" * 70)
        
        if standard_success and slack_success:
            # Parameter errors
            print(f"{'L2 Parameter Error':<25} {l2_error_standard:<15.6f} {l2_error_1slack:<15.6f} {l2_error_1slack/l2_error_standard:<15.2f}x")
            print(f"{'Max Relative Error':<25} {max_rel_error_standard:<15.3f} {max_rel_error_1slack:<15.3f} {max_rel_error_1slack/max_rel_error_standard:<15.2f}x")
            
            # Objective values
            print(f"{'Final Objective':<25} {standard_obj:<15.6f} {slack_obj:<15.6f} {abs(slack_obj - standard_obj):<15.6f}")
            
            # Performance
            print(f"{'Solve Time (s)':<25} {standard_time:<15.3f} {slack_time:<15.3f} {slack_time/standard_time:<15.2f}x")
            print(f"{'Iterations':<25} {standard_iterations:<15} {slack_iterations:<15} {slack_iterations/standard_iterations if standard_iterations != 'N/A' else 'N/A':<15}")
            print(f"{'Final Constraints':<25} {standard_constraints:<15} {slack_constraints:<15} {slack_constraints/standard_constraints:<15.2f}x")
            
            # Objective difference percentage
            obj_diff_pct = abs(slack_obj - standard_obj) / abs(standard_obj) * 100
            print(f"{'Objective Diff %':<25} {'0.00%':<15} {obj_diff_pct:<15.2f}% {'N/A':<15}")
            
        else:
            print("❌ Cannot compare - one or both formulations failed")
        
        print("\n" + "="*100)
        
        if standard_success and slack_success:
            print(f"\n✅ Successfully compared {subproblem_type} formulation")
        else:
            print(f"\n❌ Failed to compare {subproblem_type} formulation")
    
    return {
        'subproblem': subproblem_type,
        'theta_0': theta_0,
        'theta_standard': theta_hat_standard if standard_success else None,
        'theta_1slack': theta_hat_1slack if slack_success else None,
        'obj_standard': standard_obj if standard_success else None,
        'obj_1slack': slack_obj if slack_success else None,
        'standard_success': standard_success,
        'slack_success': slack_success
    }

def main():
    """Run clear comparison for a specific formulation."""
    import sys
    
    if len(sys.argv) > 1:
        formulation = sys.argv[1]
    else:
        formulation = "greedy"
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("🔬 CLEAR ROW GENERATION COMPARISON")
        print("Standard vs 1slack formulations with detailed parameter comparison")
        print("=" * 100)
    
    result = run_clear_comparison(formulation)
    
    if rank == 0:
        if result['standard_success'] and result['slack_success']:
            print(f"\n✅ Successfully compared {formulation} formulation")
        else:
            print(f"\n❌ Failed to compare {formulation} formulation")

if __name__ == "__main__":
    main()
