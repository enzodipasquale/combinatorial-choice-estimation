#!/usr/bin/env python3
"""
Simple performance comparison between standard and 1slack row generation formulations
for a single subproblem type at a time.
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

def run_single_comparison(subproblem_type, num_agents=50, num_items=10, num_features=6, num_simuls=25):
    """Run comparison for a single subproblem type."""
    print(f"\n{'='*60}")
    print(f"TESTING {subproblem_type.upper()} FORMULATION")
    print(f"{'='*60}")
    
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
            "max_iters": 30,
            "tolerance_optimality": 1e-6,
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
            "max_iters": 30,
            "tolerance_optimality": 1e-6,
            "min_iters": 1,
            "gurobi_settings": {
                "OutputFlag": 0
            }
        }
    }
    
    # True parameters
    theta_0 = np.ones(num_features)
    print(f"True parameters (theta_0): {theta_0}")
    print(f"Problem size: {num_agents} agents, {num_items} items, {num_features} features, {num_simuls} simulations")
    
    # Generate data
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        modular = np.random.normal(0, 1, (num_agents, num_items, num_features-1))
        errors = np.random.normal(0, 1, size=(1, num_agents, num_items))  # 1 simulation for generation
        agent_data = {"modular": modular}
        input_data = {"agent_data": agent_data, "errors": errors}
    else:
        input_data = None
    
    # Test Standard Formulation
    print(f"\n{'─'*30}")
    print("STANDARD FORMULATION")
    print(f"{'─'*30}")
    
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
        
        print(f"Estimated parameters: {theta_hat_standard}")
        print(f"Parameter error (L2): {l2_error_standard:.6f}")
        print(f"Max relative error: {max_rel_error_standard:.6f}")
        print(f"Solve time: {toc - tic:.3f}s")
        print(f"Final objective: {bc_standard.row_generation.master_model.objVal:.6f}")
        print(f"Iterations: {bc_standard.row_generation.iteration}")
        print(f"Final constraints: {bc_standard.row_generation.master_model.NumConstrs}")
        
        standard_success = True
        standard_time = toc - tic
        standard_obj = bc_standard.row_generation.master_model.objVal
        standard_iterations = bc_standard.row_generation.iteration
        standard_constraints = bc_standard.row_generation.master_model.NumConstrs
        
    except Exception as e:
        print(f"Standard formulation failed: {e}")
        standard_success = False
        standard_time = float('inf')
        standard_obj = float('inf')
        standard_iterations = 0
        standard_constraints = 0
        l2_error_standard = float('inf')
        max_rel_error_standard = float('inf')
    
    # Test 1slack Formulation
    print(f"\n{'─'*30}")
    print("1SLACK FORMULATION")
    print(f"{'─'*30}")
    
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
        
        solver_1slack = RowGeneration1SlackSolver(
            comm_manager=bc_1slack.comm_manager,
            dimensions_cfg=bc_1slack.config.dimensions,
            row_generation_cfg=bc_1slack.config.row_generation,
            data_manager=bc_1slack.data_manager,
            feature_manager=bc_1slack.feature_manager,
            subproblem_manager=bc_1slack.subproblem_manager
        )
        
        tic = time.time()
        theta_hat_1slack = solver_1slack.solve()
        toc = time.time()
        
        # Calculate errors
        l2_error_1slack = np.linalg.norm(theta_hat_1slack - theta_0)
        max_rel_error_1slack = np.max(np.abs(theta_hat_1slack - theta_0) / (np.abs(theta_0) + 1e-8))
        
        print(f"Estimated parameters: {theta_hat_1slack}")
        print(f"Parameter error (L2): {l2_error_1slack:.6f}")
        print(f"Max relative error: {max_rel_error_1slack:.6f}")
        print(f"Solve time: {toc - tic:.3f}s")
        print(f"Final objective: {solver_1slack.master_model.objVal:.6f}")
        print(f"Iterations: {solver_1slack.iteration}")
        print(f"Final constraints: {solver_1slack.master_model.NumConstrs}")
        
        slack_success = True
        slack_time = toc - tic
        slack_obj = solver_1slack.master_model.objVal
        slack_iterations = solver_1slack.iteration
        slack_constraints = solver_1slack.master_model.NumConstrs
        
    except Exception as e:
        print(f"1slack formulation failed: {e}")
        slack_success = False
        slack_time = float('inf')
        slack_obj = float('inf')
        slack_iterations = 0
        slack_constraints = 0
        l2_error_1slack = float('inf')
        max_rel_error_1slack = float('inf')
    
    # Summary comparison
    print(f"\n{'─'*30}")
    print("SUMMARY")
    print(f"{'─'*30}")
    
    if standard_success and slack_success:
        print(f"Parameter Recovery (L2 error):")
        print(f"  Standard: {l2_error_standard:.6f}")
        print(f"  1slack:   {l2_error_1slack:.6f}")
        print(f"  Ratio:    {l2_error_1slack/l2_error_standard:.2f}x")
        
        print(f"\nSolve Time:")
        print(f"  Standard: {standard_time:.3f}s")
        print(f"  1slack:   {slack_time:.3f}s")
        print(f"  Ratio:    {slack_time/standard_time:.2f}x")
        
        print(f"\nObjective Value:")
        print(f"  Standard: {standard_obj:.6f}")
        print(f"  1slack:   {slack_obj:.6f}")
        print(f"  Difference: {abs(slack_obj - standard_obj):.6f}")
        
        print(f"\nConstraints:")
        print(f"  Standard: {standard_constraints}")
        print(f"  1slack:   {slack_constraints}")
        print(f"  Ratio:    {slack_constraints/standard_constraints:.2f}x")
    
    return {
        'subproblem': subproblem_type,
        'standard_success': standard_success,
        'slack_success': slack_success,
        'standard_time': standard_time,
        'slack_time': slack_time,
        'standard_obj': standard_obj,
        'slack_obj': slack_obj,
        'l2_error_standard': l2_error_standard,
        'l2_error_1slack': l2_error_1slack
    }

def main():
    """Run comparison for a specific formulation."""
    import sys
    
    if len(sys.argv) > 1:
        formulation = sys.argv[1]
    else:
        formulation = "greedy"
    
    print(f"🚀 ROW GENERATION COMPARISON: {formulation.upper()}")
    print("Testing Standard vs 1slack formulations")
    print("=" * 60)
    
    result = run_single_comparison(formulation)
    
    if result['standard_success'] and result['slack_success']:
        print(f"\n✅ Successfully compared {formulation} formulation")
    else:
        print(f"\n❌ Failed to compare {formulation} formulation")

if __name__ == "__main__":
    main()
