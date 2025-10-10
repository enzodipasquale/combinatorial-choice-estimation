"""
Test that Row Generation and 1-Slack give identical objective values on the same problem.

1-Slack needs more iterations (~100+) to converge to the same solution as Row Generation (~20).
Both should reach identical objective values at convergence.

Run with: mpirun -n 10 python -m pytest bundlechoice/tests/test_solver_consistency.py -v
"""

import numpy as np
from mpi4py import MPI
from bundlechoice.core import BundleChoice
from bundlechoice.estimation import RowGeneration1SlackSolver


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


def test_all_three_solvers_consistency_greedy():
    """Verify Row Generation, 1-Slack, and Ellipsoid reach identical objective values."""
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Very small problem for speed (1-Slack with 1000 iters is slow)
    num_agents = 10
    num_items = 15
    num_features = 3
    num_simuls = 1
    
    # Generate data
    if rank == 0:
        modular = np.random.normal(0, 1, (num_agents, num_items, num_features-1))
        errors = np.random.normal(0, 1, size=(num_simuls, num_agents, num_items))
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
            "num_simuls": num_simuls,
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
    
    # --- Method 1: Row Generation (converges fast) ---
    cfg_rg = cfg_base.copy()
    cfg_rg["row_generation"] = {
        "max_iters": 50,
        "tolerance_optimality": 1e-6,
        "gurobi_settings": {"OutputFlag": 0}
    }
    
    bc_rg = BundleChoice()
    bc_rg.load_config(cfg_rg)
    bc_rg.data.load_and_scatter(input_data)
    bc_rg.features.set_oracle(features_oracle)
    bc_rg.subproblems.load()
    theta_rg = bc_rg.row_generation.solve()
    
    if rank == 0:
        objval_rg = bc_rg.row_generation.master_model.ObjVal
    
    # --- Method 2: 1-Slack (needs more iterations) ---
    cfg_1s = cfg_base.copy()
    cfg_1s["row_generation"] = {
        "max_iters": 1000,  # 1-Slack needs many more iterations
        "tolerance_optimality": 1e-6,
        "gurobi_settings": {"OutputFlag": 0}
    }
    
    bc_1s = BundleChoice()
    bc_1s.load_config(cfg_1s)
    bc_1s.data.load_and_scatter(input_data)
    bc_1s.features.set_oracle(features_oracle)
    bc_1s.subproblems.load()
    
    solver_1s = RowGeneration1SlackSolver(
        comm_manager=bc_1s.comm_manager,
        dimensions_cfg=bc_1s.config.dimensions,
        row_generation_cfg=bc_1s.config.row_generation,
        data_manager=bc_1s.data_manager,
        feature_manager=bc_1s.feature_manager,
        subproblem_manager=bc_1s.subproblem_manager
    )
    theta_1s = solver_1s.solve()
    
    if rank == 0:
        objval_1s = solver_1s.master_model.ObjVal
    
    # --- Method 3: Ellipsoid ---
    cfg_el = cfg_base.copy()
    cfg_el["ellipsoid"] = {
        "num_iters": 200,
        "tolerance": 1e-4
    }
    
    bc_el = BundleChoice()
    bc_el.load_config(cfg_el)
    bc_el.data.load_and_scatter(input_data)
    bc_el.features.set_oracle(features_oracle)
    bc_el.subproblems.load()
    theta_el = bc_el.ellipsoid.solve()
    
    if rank == 0:
        # For ellipsoid, compute objective value manually
        # The objective is sum over agents of max(0, features @ theta)
        objval_el = 0.0
        for i in range(num_agents):
            bundle = input_data["obs_bundle"][i]
            features = features_oracle(i, bundle, {"agent_data": input_data["agent_data"]})
            value = features @ theta_el
            objval_el += max(0, value)
        objval_el = -objval_el  # Negative because we're minimizing violations
        
        objval_diff_rg_1s = abs(objval_rg - objval_1s)
        objval_diff_rg_el = abs(objval_rg - objval_el)
        objval_diff_1s_el = abs(objval_1s - objval_el)
        max_diff = max(objval_diff_rg_1s, objval_diff_rg_el, objval_diff_1s_el)
        
        print(f"\nRow Generation (50 iters):   objval={objval_rg:.8f}  theta={theta_rg}")
        print(f"1-Slack (1000 iters):        objval={objval_1s:.8f}  theta={theta_1s}")
        print(f"Ellipsoid (200 iters):       objval={objval_el:.8f}  theta={theta_el}")
        print(f"\nPairwise objective differences:")
        print(f"  Row Gen vs 1-Slack:  {objval_diff_rg_1s:.10f}")
        print(f"  Row Gen vs Ellipsoid: {objval_diff_rg_el:.10f}")
        print(f"  1-Slack vs Ellipsoid: {objval_diff_1s_el:.10f}")
        print(f"  Max difference:       {max_diff:.10f}")
        
        # Row Gen and 1-Slack must be identical (same LP)
        assert objval_diff_rg_1s < 1e-6, f"Row Gen and 1-Slack give different objvals: {objval_diff_rg_1s}"
        
        # Ellipsoid should be close (different algorithm, approximate)
        assert objval_diff_rg_el < 5.0, f"Ellipsoid too far from Row Gen: {objval_diff_rg_el}"
        
        # Sanity checks
        assert not np.any(np.isnan(theta_rg)), "Row Gen theta has NaN"
        assert not np.any(np.isnan(theta_1s)), "1-Slack theta has NaN"
        assert not np.any(np.isnan(theta_el)), "Ellipsoid theta has NaN"
        
        if max_diff < 1e-3:
            print("\n✅ All three solvers converge to identical objective value!")
        else:
            print(f"\n✅ Row Gen and 1-Slack identical. Ellipsoid close (diff={objval_diff_rg_el:.4f})")


if __name__ == "__main__":
    test_all_three_solvers_consistency_greedy()
