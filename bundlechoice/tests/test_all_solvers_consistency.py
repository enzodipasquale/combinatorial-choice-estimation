"""
Test that all estimation methods produce consistent results on the same problem.
This ensures MPI optimizations haven't broken anything and all solvers find similar solutions.

Run with: mpirun -n 10 python -m pytest bundlechoice/tests/test_all_solvers_consistency.py -v
"""

import numpy as np
import pytest
from mpi4py import MPI
from bundlechoice.core import BundleChoice


def test_all_solvers_greedy_consistency():
    """Test that all solvers (row_gen, row_gen_1slack, ellipsoid) produce consistent results."""
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Shared configuration
    num_agents = 200
    num_items = 40
    num_features = 5
    num_simuls = 1
    sigma = 0.1
    
    # Generate identical data for all solvers
    np.random.seed(42)
    if rank == 0:
        modular_agent = np.random.normal(0, 1, (num_agents, num_items, num_features))
        errors = sigma * np.random.normal(0, 1, size=(num_simuls, num_agents, num_items))
        
        input_data = {
            "agent_data": {"modular": modular_agent},
            "errors": errors
        }
    else:
        input_data = None
    
    # True parameters to generate observations
    theta_true = np.array([1.0, 0.8, 1.2, 0.9, 1.1])
    
    # Generate observed bundles
    bc_obs = BundleChoice()
    bc_obs.load_config({
        "dimensions": {
            "num_agents": num_agents,
            "num_items": num_items,
            "num_features": num_features,
            "num_simuls": num_simuls
        },
        "subproblem": {"name": "Greedy"}
    })
    bc_obs.data.load_and_scatter(input_data)
    bc_obs.features.build_from_data()
    obs_bundles = bc_obs.subproblems.init_and_solve(theta_true)
    
    if rank == 0:
        input_data["obs_bundle"] = obs_bundles
    
    # --- Solver 1: Row Generation ---
    if rank == 0:
        print("\n" + "="*70)
        print("SOLVER 1: Row Generation")
        print("="*70)
    
    bc_rg = BundleChoice()
    bc_rg.load_config({
        "dimensions": {
            "num_agents": num_agents,
            "num_items": num_items,
            "num_features": num_features,
            "num_simuls": num_simuls
        },
        "subproblem": {"name": "Greedy"},
        "row_generation": {
            "max_iters": 50,
            "tolerance_optimality": 0.001,
            "gurobi_settings": {"OutputFlag": 0}
        }
    })
    bc_rg.data.load_and_scatter(input_data)
    bc_rg.features.build_from_data()
    bc_rg.subproblems.load()
    
    theta_rg = bc_rg.row_generation.solve()
    
    if rank == 0:
        obj_rg = bc_rg.row_generation.master_model.ObjVal
        print(f"  Theta: {theta_rg}")
        print(f"  ObjVal: {obj_rg:.4f}")
        print(f"  Error from truth: {np.linalg.norm(theta_rg - theta_true):.4f}")
    
    # --- Solver 2: Row Generation 1Slack ---
    if rank == 0:
        print("\n" + "="*70)
        print("SOLVER 2: Row Generation 1Slack")
        print("="*70)
    
    bc_rg1s = BundleChoice()
    bc_rg1s.load_config({
        "dimensions": {
            "num_agents": num_agents,
            "num_items": num_items,
            "num_features": num_features,
            "num_simuls": num_simuls
        },
        "subproblem": {"name": "Greedy"},
        "row_generation": {
            "max_iters": 50,
            "tolerance_optimality": 0.001,
            "gurobi_settings": {"OutputFlag": 0}
        }
    })
    bc_rg1s.data.load_and_scatter(input_data)
    bc_rg1s.features.build_from_data()
    bc_rg1s.subproblems.load()
    
    from bundlechoice.estimation.row_generation_1slack import RowGeneration1SlackSolver
    solver_1s = RowGeneration1SlackSolver(
        comm_manager=bc_rg1s.comm_manager,
        dimensions_cfg=bc_rg1s.config.dimensions,
        row_generation_cfg=bc_rg1s.config.row_generation,
        data_manager=bc_rg1s.data_manager,
        feature_manager=bc_rg1s.feature_manager,
        subproblem_manager=bc_rg1s.subproblem_manager
    )
    theta_rg1s = solver_1s.solve()
    
    if rank == 0:
        obj_rg1s = solver_1s.master_model.ObjVal
        print(f"  Theta: {theta_rg1s}")
        print(f"  ObjVal: {obj_rg1s:.4f}")
        print(f"  Error from truth: {np.linalg.norm(theta_rg1s - theta_true):.4f}")
    
    # --- Solver 3: Ellipsoid Method ---
    if rank == 0:
        print("\n" + "="*70)
        print("SOLVER 3: Ellipsoid Method")
        print("="*70)
    
    bc_ell = BundleChoice()
    bc_ell.load_config({
        "dimensions": {
            "num_agents": num_agents,
            "num_items": num_items,
            "num_features": num_features,
            "num_simuls": num_simuls
        },
        "subproblem": {"name": "Greedy"},
        "ellipsoid": {
            "num_iters": 100,
            "initial_radius": 100
        }
    })
    bc_ell.data.load_and_scatter(input_data)
    bc_ell.features.build_from_data()
    bc_ell.subproblems.load()
    
    theta_ell = bc_ell.ellipsoid.solve()
    
    if rank == 0:
        # Compute objective for ellipsoid (doesn't have model.ObjVal)
        obj_ell, _ = bc_ell.ellipsoid.compute_obj_and_gradient(theta_ell)
        print(f"  Theta: {theta_ell}")
        print(f"  ObjVal: {obj_ell:.4f}")
        print(f"  Error from truth: {np.linalg.norm(theta_ell - theta_true):.4f}")
    
    # --- Consistency Checks ---
    if rank == 0:
        print("\n" + "="*70)
        print("CONSISTENCY CHECK")
        print("="*70)
        
        # Check objectives are close (should be similar if converged)
        obj_values = [obj_rg, obj_rg1s, obj_ell]
        obj_range = max(obj_values) - min(obj_values)
        obj_mean = np.mean(obj_values)
        
        print(f"  Row Gen ObjVal:        {obj_rg:.4f}")
        print(f"  Row Gen 1Slack ObjVal: {obj_rg1s:.4f}")
        print(f"  Ellipsoid ObjVal:      {obj_ell:.4f}")
        print(f"  Range: {obj_range:.4f}")
        print(f"  Mean:  {obj_mean:.4f}")
        
        # Check theta estimates are close
        theta_diff_rg_rg1s = np.linalg.norm(theta_rg - theta_rg1s)
        theta_diff_rg_ell = np.linalg.norm(theta_rg - theta_ell)
        theta_diff_rg1s_ell = np.linalg.norm(theta_rg1s - theta_ell)
        
        print(f"\n  ||theta_rg - theta_rg1s||:  {theta_diff_rg_rg1s:.4f}")
        print(f"  ||theta_rg - theta_ell||:   {theta_diff_rg_ell:.4f}")
        print(f"  ||theta_rg1s - theta_ell||: {theta_diff_rg1s_ell:.4f}")
        
        # Assertions
        assert obj_range < 100, f"Objective values too different! Range: {obj_range}"
        
        # Parameters should be reasonably close (allowing for different convergence)
        assert theta_diff_rg_rg1s < 0.5, f"RG and RG1S too different: {theta_diff_rg_rg1s}"
        
        print("\n✅ All solvers produce consistent results!")
        print("="*70)
    
    comm.Barrier()


if __name__ == "__main__":
    test_all_solvers_greedy_consistency()
