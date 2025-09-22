"""
Quick Row Generation Performance Comparison

A simplified version for quick testing of 1slack vs standard formulation.
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

def quick_comparison():
    """Run a quick comparison between formulations."""
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Test parameters
    num_agents = 1000
    num_items = 100
    num_features = 6
    num_simuls = 1
    
    theta_0 = np.ones(num_features)
    
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
            "max_iters": 50,
            "tolerance_optimality": 0.001,
            "min_iters": 1,
            "gurobi_settings": {
                "OutputFlag": 0
            }
        }
    }
    
    if rank == 0:
        print("QUICK ROW GENERATION COMPARISON")
        print("=" * 50)
        print(f"Agents: {num_agents}, Items: {num_items}, Features: {num_features}")
        print()
    
    # Generate data
    if rank == 0:
        modular = np.random.normal(0, 1, (num_agents, num_items, num_features-1))
        errors = np.random.normal(0, 1, size=(num_simuls, num_agents, num_items))
        agent_data = {"modular": modular}
        input_data = {"agent_data": agent_data, "errors": errors}
    else:
        input_data = None

    # Test both formulations
    for formulation_name, solver_class in [("STANDARD", RowGenerationSolver), ("1SLACK", RowGeneration1SlackSolver)]:
        if rank == 0:
            print(f"Testing {formulation_name} formulation...")
        
        # Setup
        demo = BundleChoice()
        demo.load_config(cfg)
        demo.data.load_and_scatter(input_data)
        demo.features.set_oracle(features_oracle)
        
        # Generate observed bundles
        observed_bundles = demo.subproblems.init_and_solve(theta_0)
        
        if rank == 0 and observed_bundles is not None:
            input_data["obs_bundle"] = observed_bundles
            input_data["errors"] = np.random.normal(0, 1, size=(num_simuls, num_agents, num_items))
        else:
            input_data = None
        
        # Reload and solve
        demo.load_config(cfg)
        demo.data.load_and_scatter(input_data)
        demo.features.set_oracle(features_oracle)
        demo.subproblems.load()
        
        # Time the solve
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
            param_error = np.linalg.norm(theta_hat - theta_0)
            relative_errors = np.abs(theta_hat - theta_0) / (np.abs(theta_0) + 1e-8)
            max_relative_error = np.max(relative_errors)
            
            print(f"  Solve time: {solve_time:.3f}s")
            print(f"  Parameter error (L2): {param_error:.6f}")
            print(f"  Max relative error: {max_relative_error:.6f}")
            print(f"  Converged: {param_error < 1.0 and max_relative_error < 0.5}")
            print()

if __name__ == "__main__":
    quick_comparison()

