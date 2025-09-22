#!/usr/bin/env python3
"""
Large-scale speed test between standard and 1slack formulations
Tests with 100 items and 2000-5000 agents to see how they scale.
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

def run_large_scale_test(subproblem_type, num_agents, num_items, num_features, num_simuls, max_iters=100):
    """Run large-scale speed test for a single configuration."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print(f"🚀 LARGE-SCALE TEST: {subproblem_type.upper()} - {num_agents} agents, {num_items} items")
        print("=" * 80)
    
    # Generate synthetic data
    np.random.seed(42)
    modular = np.random.normal(0, 1, (num_agents, num_items))
    errors = np.random.normal(0, 0.1, (num_simuls, num_agents, num_items))
    
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
    
    # Configuration
    config = {
        "dimensions": {
            "num_agents": num_agents,
            "num_items": num_items,
            "num_features": num_features,
            "num_simuls": num_simuls,
        },
        "subproblem": {"name": subproblem_type},
        "row_generation": {
            "max_iters": max_iters,
            "tolerance_optimality": 1e-6,
        }
    }
    
    results = {}
    
    # Test Standard Formulation
    if rank == 0:
        print("Testing Standard Formulation...")
    
    try:
        bc_standard = BundleChoice()
        bc_standard.load_config(config)
        bc_standard.data.load_and_scatter(input_data)
        bc_standard.features.set_oracle(features_oracle)
        bc_standard.subproblems.load()
        
        start_time = time.time()
        solver_standard = bc_standard.estimation.get_solver()
        solver_standard.solve()
        standard_time = time.time() - start_time
        
        standard_obj = solver_standard.master_model.ObjVal
        standard_constraints = solver_standard.master_model.NumConstrs
        
        results['standard'] = {
            'time': standard_time,
            'objective': standard_obj,
            'constraints': standard_constraints,
            'success': True
        }
        
        if rank == 0:
            print(f"✅ Standard: {standard_time:.2f}s, {standard_constraints} constraints")
            
    except Exception as e:
        if rank == 0:
            print(f"❌ Standard failed: {e}")
        results['standard'] = {'success': False, 'error': str(e)}
    
    # Test 1slack Formulation
    if rank == 0:
        print("Testing 1slack Formulation...")
    
    try:
        bc_1slack = BundleChoice()
        bc_1slack.load_config(config)
        bc_1slack.data.load_and_scatter(input_data)
        bc_1slack.features.set_oracle(features_oracle)
        bc_1slack.subproblems.load()
        
        start_time = time.time()
        solver_1slack = RowGeneration1SlackSolver(
            comm_manager=bc_1slack.comm_manager,
            dimensions_cfg=bc_1slack.config.dimensions,
            row_generation_cfg=bc_1slack.config.row_generation,
            data_manager=bc_1slack.data_manager,
            feature_manager=bc_1slack.feature_manager,
            subproblem_manager=bc_1slack.subproblem_manager
        )
        solver_1slack.solve()
        slack_time = time.time() - start_time
        
        slack_obj = solver_1slack.master_model.ObjVal
        slack_constraints = solver_1slack.master_model.NumConstrs
        
        results['1slack'] = {
            'time': slack_time,
            'objective': slack_obj,
            'constraints': slack_constraints,
            'success': True
        }
        
        if rank == 0:
            print(f"✅ 1slack: {slack_time:.2f}s, {slack_constraints} constraints")
            
    except Exception as e:
        if rank == 0:
            print(f"❌ 1slack failed: {e}")
        results['1slack'] = {'success': False, 'error': str(e)}
    
    # Compare results
    if rank == 0 and results['standard']['success'] and results['1slack']['success']:
        std_time = results['standard']['time']
        slack_time = results['1slack']['time']
        speedup = std_time / slack_time if slack_time > 0 else float('inf')
        
        print(f"\n📊 RESULTS:")
        print(f"Standard Time:  {std_time:.2f}s")
        print(f"1slack Time:    {slack_time:.2f}s")
        print(f"Speedup:        {speedup:.2f}x {'(Standard faster)' if speedup > 1 else '(1slack faster)'}")
        print(f"Constraint Ratio: {results['1slack']['constraints'] / results['standard']['constraints']:.3f}x")
        
        obj_diff = abs(results['1slack']['objective'] - results['standard']['objective'])
        obj_diff_pct = obj_diff / abs(results['standard']['objective']) * 100
        print(f"Objective Diff: {obj_diff_pct:.2f}%")
        
        # Determine winner
        if speedup > 1.1:
            winner = "Standard"
        elif speedup < 0.9:
            winner = "1slack"
        else:
            winner = "Tie"
        print(f"🏆 Winner: {winner}")
        
        return {
            'subproblem': subproblem_type,
            'num_agents': num_agents,
            'num_items': num_items,
            'standard_time': std_time,
            'slack_time': slack_time,
            'speedup': speedup,
            'winner': winner,
            'constraint_ratio': results['1slack']['constraints'] / results['standard']['constraints'],
            'obj_diff_pct': obj_diff_pct
        }
    
    return None

def main():
    """Run large-scale speed tests."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("🚀 LARGE-SCALE SPEED COMPARISON: Standard vs 1slack Formulations")
        print("=" * 80)
    
    # Test configurations for large-scale problems
    test_configs = [
        # (subproblem, agents, items, features, simuls, max_iters)
        ("greedy", 2000, 100, 6, 50, 100),
        ("greedy", 5000, 100, 6, 50, 100),
        ("linear_knapsack", 2000, 100, 6, 50, 100),
        ("linear_knapsack", 5000, 100, 6, 50, 100),
        ("plain_single_item", 2000, 100, 6, 50, 100),
        ("plain_single_item", 5000, 100, 6, 50, 100),
    ]
    
    all_results = []
    
    for config in test_configs:
        result = run_large_scale_test(*config)
        if result:
            all_results.append(result)
    
    # Summary
    if rank == 0 and all_results:
        print(f"\n{'='*80}")
        print("📈 LARGE-SCALE SUMMARY")
        print(f"{'='*80}")
        
        print(f"{'Subproblem':<20} {'Agents':<8} {'Standard':<12} {'1slack':<12} {'Speedup':<10} {'Winner':<10}")
        print("-" * 80)
        
        for result in all_results:
            print(f"{result['subproblem']:<20} {result['num_agents']:<8} {result['standard_time']:<12.2f} {result['slack_time']:<12.2f} {result['speedup']:<10.2f} {result['winner']:<10}")
        
        # Overall statistics
        standard_wins = sum(1 for r in all_results if r['winner'] == 'Standard')
        slack_wins = sum(1 for r in all_results if r['winner'] == '1slack')
        ties = sum(1 for r in all_results if r['winner'] == 'Tie')
        
        avg_speedup = np.mean([r['speedup'] for r in all_results])
        avg_constraint_ratio = np.mean([r['constraint_ratio'] for r in all_results])
        avg_obj_diff = np.mean([r['obj_diff_pct'] for r in all_results])
        
        print(f"\n📊 STATISTICS:")
        print(f"Standard Wins: {standard_wins}")
        print(f"1slack Wins: {slack_wins}")
        print(f"Ties: {ties}")
        print(f"Average Speedup: {avg_speedup:.2f}x")
        print(f"Average Constraint Ratio: {avg_constraint_ratio:.3f}x")
        print(f"Average Objective Diff: {avg_obj_diff:.2f}%")
        
        if avg_speedup > 1.1:
            overall_winner = "Standard Formulation"
        elif avg_speedup < 0.9:
            overall_winner = "1slack Formulation"
        else:
            overall_winner = "Tie - Both are competitive"
        
        print(f"\n🏆 OVERALL WINNER: {overall_winner}")

if __name__ == "__main__":
    main()

