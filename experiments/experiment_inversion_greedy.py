"""
Row Generation Greedy Experiment

This script runs an experiment using the RowGenerationSolver with greedy subproblem manager.
It's based on the test but adapted for standalone execution and experimentation.
"""
import numpy as np
import time
from mpi4py import MPI
from bundlechoice.core import BundleChoice

def run_row_generation_greedy_experiment():
    """Run the row generation greedy experiment."""
    # Experiment parameters
    num_agents = 200
    num_items = 100
    num_features = 4
    num_simuls = 1
    sigma = 6
    
    # Configuration
    cfg = {
        "dimensions": {
            "num_agents": num_agents,
            "num_items": num_items,
            "num_features": num_features,
        },
        "subproblem": {
            "name": "Greedy",
        },
        "row_generation": {
            "max_iters": 500,
            "tolerance_optimality": 0.001,
            "gurobi_settings": {
                "OutputFlag": 0
            },
            "max_slack_counter": 20
        }
    }
      
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    ########## Generate data ##########
    if rank == 0:
        modular = np.abs(np.random.normal(0, 1, (num_items, num_features-1)))
        endogenous_errors = np.random.normal(0, 1, size=(num_items,)) 
        instrument = np.random.normal(0, 1, size=(num_items,)) 
        modular[:,0] = instrument + endogenous_errors + np.random.normal(0, 1, size=(num_items,))
        errors = sigma * np.random.normal(0, 1, size=(num_agents, num_items)) + endogenous_errors[None,:]
        estimation_errors = sigma * np.random.normal(0, 1, size=(num_simuls, num_agents, num_items))
        item_data = {"modular": modular}
        input_data = {"item_data": item_data, "errors": errors}
    else:
        input_data = None

    greedy_demo = BundleChoice()
    greedy_demo.load_config(cfg)
    greedy_demo.data.load_and_scatter(input_data)
    greedy_demo.features.set_oracle(features_oracle)

    theta_0 = np.ones(num_features) * 2
    theta_0[-1] = .1
    start_time = time.time()
    obs_bundles = greedy_demo.subproblems.init_and_solve(theta_0)
    bundle_time = time.time() - start_time


    ########## Estimation ##########
    # if rank == 0:
    #     print(f"[Rank 0] Bundle generation completed in {bundle_time:.2f} seconds")
    #     print(f"[Rank 0] Aggregate demands: {obs_bundles.sum(1).min():.2f} to {obs_bundles.sum(1).max():.2f}")
    #     print(f"[Rank 0] Total aggregate: {obs_bundles.sum():.2f}")
    #     input_data["obs_bundle"] = obs_bundles
    #     input_data["errors"] = estimation_errors
    #     cfg["dimensions"]["num_simuls"] = num_simuls
    # else:
    #     input_data = None

    # if rank == 0:
    #     print(f"[Rank {rank}] Setting up for parameter estimation...")
    # greedy_demo.load_config(cfg)
    # greedy_demo.data.load_and_scatter(input_data)
    # greedy_demo.features.set_oracle(features_oracle)
    # greedy_demo.subproblems.load()
    
    
    # # Run row generation method
    # if rank == 0:
    #     print(f"[Rank {rank}] Starting row generation optimization...")
    # start_time = time.time()
    # theta_hat = greedy_demo.row_generation.solve()
    # optimization_time = time.time() - start_time
    #   # Compute objective values on all ranks
    # try:
    #     obj_at_star = greedy_demo.row_generation.objective(theta_0)
    #     obj_at_hat = greedy_demo.row_generation.objective(theta_hat)
    # except AttributeError:
    #     obj_at_star = None
    #     obj_at_hat = None
    
    # if rank == 0:
    #     print(f"[Rank 0] Optimization completed in {optimization_time:.2f} seconds")
    #     print(f"[Rank 0] Estimated parameters (theta_hat): {theta_hat}")
    #     print(f"[Rank 0] True parameters (theta_0): {theta_0}")
    #     print(f"[Rank 0] Parameter difference: {np.round(np.abs(theta_hat - theta_0), 2)}")
        
    #     # Print objective values if available
    #     if obj_at_star is not None and obj_at_hat is not None:
    #         print(f"[Rank 0] Objective at true parameters: {obj_at_star:.4f}")
    #         print(f"[Rank 0] Objective at estimated parameters: {obj_at_hat:.4f}")
    #         print(f"[Rank 0] Objective improvement: {obj_at_star - obj_at_hat:.4f}")
    #     else:
    #         print("[Rank 0] Objective function not available for row generation solver")



    ########## Modular BLP inversion ##########
    if rank == 0:
        print(f"[Rank 0] Bundle generation completed in {bundle_time:.2f} seconds")
        print(f"[Rank 0] Aggregate demands: {obs_bundles.sum(1).min():.2f} to {obs_bundles.sum(1).max():.2f}")
        print(f"[Rank 0] Total aggregate: {obs_bundles.sum():.2f}")
        print(f"[Rank 0] Demands_j: {obs_bundles.sum(0)}")
        input_data["item_data"]["modular"] = np.eye(num_items)
        input_data["obs_bundle"] = obs_bundles
        input_data["errors"] = estimation_errors 
    else:
        input_data = None
    cfg["dimensions"]["num_simuls"] = num_simuls
    cfg["dimensions"]["num_features"] = num_items + 1
    cfg["row_generation"]["theta_lbs"] = np.array([-500] * num_items + [0])
    cfg["row_generation"]["theta_ubs"] = 500
    
    greedy_demo.load_config(cfg)
    greedy_demo.data.load_and_scatter(input_data)
    greedy_demo.features.set_oracle(features_oracle)
    greedy_demo.subproblems.load()
    start_time = time.time()
    theta_hat = greedy_demo.row_generation.solve()
    if rank == 0:
        delta_hat = theta_hat[:-1]
        import statsmodels.api as sm
        from statsmodels.sandbox.regression.gmm import IV2SLS
        instruments = np.concatenate([instrument[:, None], modular], axis=1)
        iv_model = IV2SLS(delta_hat, modular, instruments)
        iv_results = iv_model.fit()
        print(f"Coefficients: {iv_results.params}")
        print(f"Standard errors: {iv_results.bse}")

def features_oracle(i_id, B_j, data):
    """
    Compute features for a given agent and bundle(s).
    Supports both single (1D) and multiple (2D) bundles.
    Returns array of shape (num_features,) for a single bundle,
    or (num_features, m) for m bundles.
    """
    modular_item = data["item_data"]["modular"]

    modular_item = np.atleast_2d(modular_item)

    single_bundle = False
    if B_j.ndim == 1:
        B_j = B_j[:, None]
        single_bundle = True
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        agent_sum = modular_item.T @ B_j
    neg_sq = -np.sum(B_j, axis=0, keepdims=True) ** 2

    features = np.vstack((agent_sum, neg_sq))
    if single_bundle:
        return features[:, 0]  # Return as 1D array for a single bundle
    return features


if __name__ == "__main__":
    run_row_generation_greedy_experiment() 