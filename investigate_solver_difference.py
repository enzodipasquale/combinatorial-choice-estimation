"""
Investigate difference between Row Generation and 1-Slack objective values.
"""

import numpy as np
from mpi4py import MPI
from bundlechoice.core import BundleChoice
from bundlechoice.estimation import RowGeneration1SlackSolver

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Exact setup from working test
num_agents = 20
num_items = 50
num_features = 6
num_simuls = 1

def features_oracle(i_id, B_j, data):
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

# Generate data (exactly as in test_estimation_row_generation_greedy.py)
if rank == 0:
    modular = np.random.normal(0, 1, (num_agents, num_items, num_features-1))
    errors = np.random.normal(0, 1, size=(num_simuls, num_agents, num_items))
    agent_data = {"modular": modular}
    input_data = {"agent_data": agent_data, "errors": errors}
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
greedy_gen = BundleChoice()
greedy_gen.load_config(cfg_base)
greedy_gen.data.load_and_scatter(input_data)
greedy_gen.features.set_oracle(features_oracle)

theta_0 = np.ones(num_features)
observed_bundles = greedy_gen.subproblems.init_and_solve(theta_0)

# Update data with observations
if rank == 0:
    input_data["obs_bundle"] = observed_bundles

if rank == 0:
    print("\n" + "="*80)
    print("COMPARING ROW GENERATION vs 1-SLACK with varying max_iters")
    print("="*80)
    print(f"Problem: {num_agents} agents, {num_items} items, {num_features} features")
    print(f"Observed bundles: min={observed_bundles.sum(1).min()}, max={observed_bundles.sum(1).max()}")

# Test different iteration limits
for max_iters in [10, 20, 50, 100, 500, 1000]:
    
    # --- Row Generation ---
    cfg_rg = cfg_base.copy()
    cfg_rg["row_generation"] = {
        "max_iters": max_iters,
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
        constrs_rg = bc_rg.row_generation.master_model.NumConstrs
    
    # --- 1-Slack Row Generation ---
    cfg_1s = cfg_base.copy()
    cfg_1s["row_generation"] = {
        "max_iters": max_iters,
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
        constrs_1s = solver_1s.master_model.NumConstrs
        
        diff = abs(objval_rg - objval_1s)
        theta_diff = np.linalg.norm(theta_rg - theta_1s)
        
        print(f"\nmax_iters={max_iters:4d}:")
        print(f"  Row Gen:  objval={objval_rg:12.8f}  constraints={constrs_rg:4d}  theta={theta_rg}")
        print(f"  1-Slack:  objval={objval_1s:12.8f}  constraints={constrs_1s:4d}  theta={theta_1s}")
        print(f"  Objval diff:   {diff:.10f}  Theta diff: {theta_diff:.10f}", end="")
        
        if diff < 1e-8:
            print("  ✅ IDENTICAL")
        elif diff < 1e-4:
            print("  ⚠️  Very close")
        elif diff < 1e-2:
            print("  ⚠️  Close")
        else:
            print("  ❌ DIFFERENT")

if rank == 0:
    print("\n" + "="*80)
