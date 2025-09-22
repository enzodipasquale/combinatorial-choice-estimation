"""
Row Generation Performance Comparison: 1slack vs Standard Formulation

This experiment compares the performance of the 1slack formulation against the standard
row generation formulation across different subproblem types and problem sizes.

Metrics compared:
- Total solve time
- Number of iterations to convergence
- Final objective value
- Parameter recovery accuracy
- Memory usage (constraint count)
"""
import numpy as np
import time
import json
from datetime import datetime
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

def run_single_experiment(subproblem_name, num_agents, num_items, num_features, 
                         num_simuls=1, max_iters=100, tolerance=0.001):
    """Run a single experiment comparing both formulations."""
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Generate true parameters
    theta_0 = np.ones(num_features)
    
    # Configuration
    cfg = {
        "dimensions": {
            "num_agents": num_agents,
            "num_items": num_items,
            "num_features": num_features,
            "num_simuls": num_simuls,
        },
        "subproblem": {
            "name": subproblem_name,
        },
        "row_generation": {
            "max_iters": max_iters,
            "tolerance_optimality": tolerance,
            "min_iters": 1,
            "gurobi_settings": {
                "OutputFlag": 0
            }
        }
    }
    
    results = {
        "subproblem": subproblem_name,
        "num_agents": num_agents,
        "num_items": num_items,
        "num_features": num_features,
        "theta_0": theta_0.tolist(),
        "standard": {},
        "1slack": {}
    }
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {subproblem_name}")
        print(f"Agents: {num_agents}, Items: {num_items}, Features: {num_features}")
        print(f"{'='*60}")
    
    # Generate data
    if rank == 0:
        if subproblem_name == "Greedy":
            modular = np.random.normal(0, 1, (num_agents, num_items, num_features-1))
            errors = np.random.normal(0, 1, size=(num_simuls, num_agents, num_items))
            agent_data = {"modular": modular}
            input_data = {"agent_data": agent_data, "errors": errors}
        elif subproblem_name == "LinearKnapsack":
            input_data = {
                "item_data": {
                    "modular": np.abs(np.random.normal(0, 1, (num_items, num_features//2))),
                    "weights": np.random.randint(1, 10, size=num_items)
                },
                "agent_data": {
                    "modular": np.abs(np.random.normal(0, 1, (num_agents, num_items, num_features//2))),
                    "capacity": np.random.randint(1, 100, size=num_agents)
                },
                "errors": np.random.normal(0, 1, (num_simuls, num_agents, num_items)),
            }
        elif subproblem_name == "PlainSingleItem":
            errors = np.random.normal(0, 1, (num_agents, num_items))
            estimation_errors = np.random.normal(0, 1, (num_simuls, num_agents, num_items))
            input_data = {
                "item_data": {"modular": np.random.normal(0, 1, (num_items, num_features//2))},
                "agent_data": {"modular": np.random.normal(0, 1, (num_agents, num_items, num_features//2))},
                "errors": errors,
            }
        else:
            raise ValueError(f"Unknown subproblem: {subproblem_name}")
    else:
        input_data = None
    
    # Test both formulations
    for formulation_name, solver_class in [("standard", RowGenerationSolver), ("1slack", RowGeneration1SlackSolver)]:
        if rank == 0:
            print(f"\n--- Testing {formulation_name.upper()} formulation ---")
        
        # Initialize BundleChoice
        demo = BundleChoice()
        demo.load_config(cfg)
        demo.data.load_and_scatter(input_data)
        
        if subproblem_name == "Greedy":
            demo.features.set_oracle(features_oracle)
        else:
            demo.features.build_from_data()
        
        # Generate observed bundles
        observed_bundles = demo.subproblems.init_and_solve(theta_0)
        
        if rank == 0 and observed_bundles is not None:
            input_data["obs_bundle"] = observed_bundles
            input_data["errors"] = np.random.normal(0, 1, size=(num_simuls, num_agents, num_items))
        else:
            input_data = None
        
        # Reload data with observed bundles
        demo.load_config(cfg)
        demo.data.load_and_scatter(input_data)
        
        if subproblem_name == "Greedy":
            demo.features.set_oracle(features_oracle)
        else:
            demo.features.build_from_data()
        demo.subproblems.load()
        
        # Run estimation
        start_time = time.time()
        solver = solver_class(
            comm_manager=demo.comm_manager,
            dimensions_cfg=demo.config.dimensions,
            row_generation_cfg=demo.config.row_generation,
            data_manager=demo.data_manager,
            feature_manager=demo.feature_manager,
            subproblem_manager=demo.subproblem_manager
        )
        theta_hat = solver.solve()
        end_time = time.time()
        
        if rank == 0:
            solve_time = end_time - start_time
            
            # Calculate metrics
            param_error = np.linalg.norm(theta_hat - theta_0)
            relative_errors = np.abs(theta_hat - theta_0) / (np.abs(theta_0) + 1e-8)
            max_relative_error = np.max(relative_errors)
            
            # Get solver-specific metrics
            if hasattr(solver, 'master_model') and solver.master_model is not None:
                final_obj = solver.master_model.ObjVal
                num_constraints = solver.master_model.NumConstrs
            else:
                final_obj = None
                num_constraints = None
            
            results[formulation_name] = {
                "solve_time": solve_time,
                "theta_hat": theta_hat.tolist(),
                "param_error_l2": param_error,
                "max_relative_error": max_relative_error,
                "final_objective": final_obj,
                "num_constraints": num_constraints,
                "converged": param_error < 1.0 and max_relative_error < 0.5
            }
            
            print(f"Solve time: {solve_time:.3f}s")
            print(f"Parameter error (L2): {param_error:.6f}")
            print(f"Max relative error: {max_relative_error:.6f}")
            print(f"Final objective: {final_obj}")
            print(f"Number of constraints: {num_constraints}")
            print(f"Converged: {results[formulation_name]['converged']}")
    
    return results

def run_performance_comparison():
    """Run comprehensive performance comparison across different scenarios."""
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("ROW GENERATION PERFORMANCE COMPARISON")
        print("1slack vs Standard Formulation")
        print(f"Started at: {datetime.now()}")
        print(f"MPI processes: {comm.Get_size()}")
    
    # Define experiment scenarios
    scenarios = [
        # (subproblem_name, num_agents, num_items, num_features)
        ("Greedy", 500, 50, 6),
        ("Greedy", 1000, 100, 6),
        ("Greedy", 2000, 200, 6),
        ("LinearKnapsack", 500, 20, 4),
        ("LinearKnapsack", 1000, 30, 4),
        ("PlainSingleItem", 1000, 5, 5),
        ("PlainSingleItem", 2000, 10, 5),
    ]
    
    all_results = []
    
    for i, (subproblem, agents, items, features) in enumerate(scenarios):
        if rank == 0:
            print(f"\n\nSCENARIO {i+1}/{len(scenarios)}")
        
        try:
            result = run_single_experiment(subproblem, agents, items, features)
            all_results.append(result)
        except Exception as e:
            if rank == 0:
                print(f"ERROR in scenario {i+1}: {e}")
            continue
    
    # Save results
    if rank == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"experiments/performance_comparisons/row_generation_comparison_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n\n{'='*60}")
        print("SUMMARY RESULTS")
        print(f"{'='*60}")
        print(f"Results saved to: {filename}")
        
        # Print summary table
        print(f"\n{'Scenario':<20} {'Formulation':<12} {'Time(s)':<10} {'L2 Error':<10} {'Rel Error':<10} {'Converged':<10}")
        print("-" * 80)
        
        for result in all_results:
            scenario = f"{result['subproblem']}_{result['num_agents']}"
            for formulation in ["standard", "1slack"]:
                if formulation in result:
                    data = result[formulation]
                    print(f"{scenario:<20} {formulation:<12} {data['solve_time']:<10.3f} "
                          f"{data['param_error_l2']:<10.3f} {data['max_relative_error']:<10.3f} "
                          f"{data['converged']:<10}")
        
        # Calculate speedup ratios
        print(f"\n{'Scenario':<20} {'Speedup':<10} {'Time Ratio':<12}")
        print("-" * 50)
        
        for result in all_results:
            if "standard" in result and "1slack" in result:
                scenario = f"{result['subproblem']}_{result['num_agents']}"
                std_time = result["standard"]["solve_time"]
                slack_time = result["1slack"]["solve_time"]
                speedup = std_time / slack_time if slack_time > 0 else float('inf')
                time_ratio = slack_time / std_time if std_time > 0 else float('inf')
                print(f"{scenario:<20} {speedup:<10.2f} {time_ratio:<12.2f}")

if __name__ == "__main__":
    run_performance_comparison()

