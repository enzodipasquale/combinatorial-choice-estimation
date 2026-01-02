"""
Test that Row Generation and 1-Slack give identical objective values on the same problem.

1-Slack needs ~100+ iterations to converge to the same solution as Row Generation.
Both should reach identical objective values at convergence (they solve the same LP).

Run with: mpirun -n 10 python -m pytest bundlechoice/tests/test_solver_consistency.py -v
"""

import numpy as np
from mpi4py import MPI
from bundlechoice.core import BundleChoice
from bundlechoice.estimation import RowGeneration1SlackManager


def features_oracle(i_id, B_j, data):
    """Feature function for greedy test."""
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


def test_row_generation_vs_1slack_identical_objval():
    """Verify Row Generation and 1-Slack reach identical objective values."""
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Small problem for speed
    num_agents = 10
    num_items = 15
    num_features = 3
    num_simulations = 1
    
    # Generate data
    if rank == 0:
        modular = np.random.normal(0, 1, (num_agents, num_items, num_features-1))
        errors = np.random.normal(0, 1, size=(num_simulations, num_agents, num_items))
        input_data = {
            "agent_data": {"modular": modular},
            "errors": errors
        }
    else:
        input_data = None
    
    cfg_base = {
        "dimensions": {
            "num_agents": num_agents,
            "num_items": num_items,
            "num_features": num_features,
            "num_simulations": num_simulations,
        },
        "subproblem": {"name": "Greedy"},
    }
    
    # Generate observed bundles
    bc_gen = BundleChoice()
    bc_gen.load_config(cfg_base)
    bc_gen.data.load_and_scatter(input_data)
    bc_gen.features.set_oracle(features_oracle)
    observed_bundles = bc_gen.subproblems.init_and_solve(np.ones(num_features))
    
    if rank == 0:
        input_data["obs_bundle"] = observed_bundles
    
    # --- Method 1: Row Generation (converges fast, ~20 iterations) ---
    cfg_rg = cfg_base.copy()
    cfg_rg["row_generation"] = {
        "max_iters": 100,
        "tolerance_optimality": 1e-6,
        "gurobi_settings": {"OutputFlag": 0}
    }
    
    bc_rg = BundleChoice()
    bc_rg.load_config(cfg_rg)
    bc_rg.data.load_and_scatter(input_data)
    bc_rg.features.set_oracle(features_oracle)
    bc_rg.subproblems.load()
    result_rg = bc_rg.row_generation.solve()
    
    if rank == 0:
        theta_rg = result_rg.theta_hat
        objval_rg = bc_rg.row_generation.master_model.ObjVal
    
    # --- Method 2: 1-Slack (needs 1000 max iterations to ensure convergence) ---
    cfg_1s = cfg_base.copy()
    cfg_1s["row_generation"] = {
        "max_iters": 1000,
        "tolerance_optimality": 1e-6,
        "gurobi_settings": {"OutputFlag": 0}
    }
    
    bc_1s = BundleChoice()
    bc_1s.load_config(cfg_1s)
    bc_1s.data.load_and_scatter(input_data)
    bc_1s.features.set_oracle(features_oracle)
    bc_1s.subproblems.load()
    
    solver_1s = RowGeneration1SlackManager(
        comm_manager=bc_1s.comm_manager,
        dimensions_cfg=bc_1s.config.dimensions,
        row_generation_cfg=bc_1s.config.row_generation,
        data_manager=bc_1s.data_manager,
        feature_manager=bc_1s.feature_manager,
        subproblem_manager=bc_1s.subproblem_manager
    )
    result_1s = solver_1s.solve()
    
    if rank == 0:
        theta_1s = result_1s.theta_hat
        objval_1s = solver_1s.master_model.ObjVal
        
        objval_diff = abs(objval_rg - objval_1s)
        theta_diff = np.linalg.norm(theta_rg - theta_1s)
        
        print(f"\nRow Generation (100 max_iters):   objval={objval_rg:.10f}")
        print(f"1-Slack (1000 max_iters):         objval={objval_1s:.10f}")
        print(f"\nObjective difference: {objval_diff:.12f}")
        print(f"Theta difference:     {theta_diff:.6f} (may differ - multiple optimal solutions)")
        
        # Objective values must be identical (they solve the same LP)
        assert objval_diff < 1e-6, f"Row Gen and 1-Slack give different objvals: {objval_diff}"
        
        # Sanity checks
        assert not np.any(np.isnan(theta_rg)), "Row Gen theta has NaN"
        assert not np.any(np.isnan(theta_1s)), "1-Slack theta has NaN"
        
        print("\n✅ PASS: Row Generation and 1-Slack converge to identical objective value!")
        print(f"   Both methods solve the same LP and reach the same optimum.")


if __name__ == "__main__":
    test_row_generation_vs_1slack_identical_objval()
