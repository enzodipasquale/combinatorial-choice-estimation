#!/usr/bin/env python3
"""
Comprehensive performance comparison between standard and 1slack row generation formulations
across all 4 subproblem types: greedy, linear_knapsack, plain_single_item, quadratic_supermodular.
"""

import time
import numpy as np
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

def run_comparison(subproblem_type, num_agents=100, num_items=20, num_features=6, num_simuls=50):
    """Run comparison for a specific subproblem type."""
    print(f"\n{'='*80}")
    print(f"TESTING {subproblem_type.upper()} FORMULATION")
    print(f"{'='*80}")
    
    # Configuration for the specific subproblem
    config = {
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
            "max_iters": 50,
            "tolerance_optimality": 1e-6,
            "min_iters": 1,
            "gurobi_settings": {
                "OutputFlag": 0
            }
        }
    }
    
    # True parameters for comparison
    theta_0 = np.ones(num_features)
    
    print(f"True parameters (theta_0): {theta_0}")
    print(f"Problem size: {num_agents} agents, {num_items} items, {num_features} features, {num_simuls} simulations")
    
    # Generate data
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        modular = np.random.normal(0, 1, (num_agents, num_items, num_features-1))
        errors = np.random.normal(0, 1, size=(num_simuls, num_agents, num_items)) 
        agent_data = {"modular": modular}
        input_data = {"agent_data": agent_data, "errors": errors}
    else:
        input_data = None
    
    # Test Standard Formulation
    print(f"\n{'─'*40}")
    print("TESTING STANDARD FORMULATION")
    print(f"{'─'*40}")
    
    try:
        bc_standard = BundleChoice()
        bc_standard.load_config(config)
        bc_standard.data.load_and_scatter(input_data)
        bc_standard.features.set_oracle(features_oracle)
        
        # Generate observed bundles
        observed_bundles = bc_standard.subproblems.init_and_solve(theta_0)
        
        if rank == 0:
            input_data["obs_bundle"] = observed_bundles
            input_data["errors"] = np.random.normal(0, 1, size=(num_simuls, num_agents, num_items))
        else:
            input_data = None
        
        bc_standard.load_config(config)
        bc_standard.data.load_and_scatter(input_data)
        bc_standard.features.set_oracle(features_oracle)
        bc_standard.subproblems.load()
        
        tic = time.time()
        theta_hat_standard = bc_standard.row_generation.solve()
        toc = time.time()
        
        # Calculate errors
        l2_error_standard = np.linalg.norm(theta_hat_standard - theta_0)
        max_rel_error_standard = np.max(np.abs(theta_hat_standard - theta_0) / (np.abs(theta_0) + 1e-8))
        
        print(f"Standard - Estimated parameters: {theta_hat_standard}")
        print(f"Standard - Parameter error (L2): {l2_error_standard:.6f}")
        print(f"Standard - Max relative error: {max_rel_error_standard:.6f}")
        print(f"Standard - Solve time: {toc - tic:.3f}s")
        print(f"Standard - Final objective: {bc_standard.row_generation.master_model.objVal:.6f}")
        print(f"Standard - Iterations: {bc_standard.row_generation.iteration}")
        print(f"Standard - Final constraints: {bc_standard.row_generation.master_model.NumConstrs}")
        
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
    print(f"\n{'─'*40}")
    print("TESTING 1SLACK FORMULATION")
    print(f"{'─'*40}")
    
    try:
        bc_1slack = BundleChoice()
        bc_1slack.load_config(config)
        bc_1slack.data.load_and_scatter(input_data)
        bc_1slack.features.set_oracle(features_oracle)
        
        # Generate observed bundles (reuse same data)
        observed_bundles = bc_1slack.subproblems.init_and_solve(theta_0)
        
        if rank == 0:
            input_data["obs_bundle"] = observed_bundles
            input_data["errors"] = np.random.normal(0, 1, size=(num_simuls, num_agents, num_items))
        else:
            input_data = None
        
        bc_1slack.load_config(config)
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
        
        print(f"1slack - Estimated parameters: {theta_hat_1slack}")
        print(f"1slack - Parameter error (L2): {l2_error_1slack:.6f}")
        print(f"1slack - Max relative error: {max_rel_error_1slack:.6f}")
        print(f"1slack - Solve time: {toc - tic:.3f}s")
        print(f"1slack - Final objective: {solver_1slack.master_model.objVal:.6f}")
        print(f"1slack - Iterations: {solver_1slack.iteration}")
        print(f"1slack - Final constraints: {solver_1slack.master_model.NumConstrs}")
        
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
    print(f"\n{'─'*40}")
    print("SUMMARY COMPARISON")
    print(f"{'─'*40}")
    
    if standard_success and slack_success:
        print(f"Parameter Recovery (L2 error):")
        print(f"  Standard: {l2_error_standard:.6f}")
        print(f"  1slack:   {l2_error_1slack:.6f}")
        print(f"  Ratio:    {l2_error_1slack/l2_error_standard:.2f}x")
        
        print(f"\nParameter Recovery (Max relative error):")
        print(f"  Standard: {max_rel_error_standard:.6f}")
        print(f"  1slack:   {max_rel_error_1slack:.6f}")
        print(f"  Ratio:    {max_rel_error_1slack/max_rel_error_standard:.2f}x")
        
        print(f"\nSolve Time:")
        print(f"  Standard: {standard_time:.3f}s")
        print(f"  1slack:   {slack_time:.3f}s")
        print(f"  Ratio:    {slack_time/standard_time:.2f}x")
        
        print(f"\nObjective Value:")
        print(f"  Standard: {standard_obj:.6f}")
        print(f"  1slack:   {slack_obj:.6f}")
        print(f"  Difference: {abs(slack_obj - standard_obj):.6f}")
        
        print(f"\nIterations:")
        print(f"  Standard: {standard_iterations}")
        print(f"  1slack:   {slack_iterations}")
        
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
        'standard_iterations': standard_iterations,
        'slack_iterations': slack_iterations,
        'standard_constraints': standard_constraints,
        'slack_constraints': slack_constraints,
        'l2_error_standard': l2_error_standard,
        'l2_error_1slack': l2_error_1slack,
        'max_rel_error_standard': max_rel_error_standard,
        'max_rel_error_1slack': max_rel_error_1slack
    }

def main():
    """Run comprehensive comparison across all formulations."""
    print("🚀 COMPREHENSIVE ROW GENERATION COMPARISON")
    print("Testing Standard vs 1slack formulations across all subproblem types")
    print("=" * 80)
    
    # Test all 4 formulations
    formulations = ['greedy', 'linear_knapsack', 'plain_single_item', 'quadratic_supermodular']
    
    results = []
    for formulation in formulations:
        try:
            result = run_comparison(formulation)
            results.append(result)
        except Exception as e:
            print(f"❌ Failed to test {formulation}: {e}")
            results.append({
                'subproblem': formulation,
                'standard_success': False,
                'slack_success': False,
                'error': str(e)
            })
    
    # Overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    
    successful_tests = [r for r in results if r.get('standard_success', False) and r.get('slack_success', False)]
    
    if successful_tests:
        print(f"Successfully tested {len(successful_tests)}/{len(formulations)} formulations")
        
        # Average performance metrics
        avg_time_ratio = np.mean([r['slack_time']/r['standard_time'] for r in successful_tests])
        avg_obj_diff = np.mean([abs(r['slack_obj'] - r['standard_obj']) for r in successful_tests])
        avg_l2_ratio = np.mean([r['l2_error_1slack']/r['l2_error_standard'] for r in successful_tests])
        avg_constraint_ratio = np.mean([r['slack_constraints']/r['standard_constraints'] for r in successful_tests])
        
        print(f"\nAverage Performance Ratios (1slack/Standard):")
        print(f"  Time: {avg_time_ratio:.2f}x")
        print(f"  L2 Error: {avg_l2_ratio:.2f}x")
        print(f"  Constraints: {avg_constraint_ratio:.2f}x")
        print(f"  Average Objective Difference: {avg_obj_diff:.6f}")
        
        # Best and worst cases
        best_time = min(successful_tests, key=lambda x: x['slack_time']/x['standard_time'])
        worst_time = max(successful_tests, key=lambda x: x['slack_time']/x['standard_time'])
        
        print(f"\nBest Time Performance: {best_time['subproblem']} ({best_time['slack_time']/best_time['standard_time']:.2f}x)")
        print(f"Worst Time Performance: {worst_time['subproblem']} ({worst_time['slack_time']/worst_time['standard_time']:.2f}x)")
    
    else:
        print("❌ No successful tests completed")

if __name__ == "__main__":
    main()
