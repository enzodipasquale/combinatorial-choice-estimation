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
    num_agent_features = 1
    num_item_features = 1

    num_agents = 400
    num_items = 100
    num_features = num_agent_features + num_item_features +1
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
            "max_slack_counter": 30
        }
    }
      
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    ########## Generate data ##########
    if rank == 0:
        modular_item = np.abs(np.random.normal(0, 1, (num_items, num_item_features)))
        endogenous_errors = np.random.normal(0, 1, size=(num_items,)) 
        instrument = np.random.normal(0, 1, size=(num_items,)) 
        modular_item[:,0] = instrument + endogenous_errors + np.random.normal(0, .5, size=(num_items,))

        modular_agent = np.abs(np.random.normal(0, 1, (num_agents, num_items, num_agent_features)))
        errors = sigma * np.random.normal(0, 1, size=(num_agents, num_items)) + endogenous_errors[None,:]
        estimation_errors = sigma * np.random.normal(0, 1, size=(num_simuls, num_agents, num_items))
        item_data = {"modular": modular_item}
        agent_data = {"modular": modular_agent}
        input_data = {"item_data": item_data, 
                      "agent_data": agent_data,
                      "errors": errors}
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


    ######### Estimation ##########
    if rank == 0:
        print(f"[Rank 0] Bundle generation completed in {bundle_time:.2f} seconds")
        print(f"[Rank 0] Aggregate demands: {obs_bundles.sum(1).min():.2f} to {obs_bundles.sum(1).max():.2f}")
        print(f"[Rank 0] Total aggregate: {obs_bundles.sum():.2f}")
        input_data["obs_bundle"] = obs_bundles
        input_data["errors"] = estimation_errors
        cfg["dimensions"]["num_simuls"] = num_simuls
    else:
        input_data = None

    if rank == 0:
        print(f"[Rank {rank}] Setting up for parameter estimation...")
    greedy_demo.load_config(cfg)
    greedy_demo.data.load_and_scatter(input_data)
    greedy_demo.features.set_oracle(features_oracle)
    greedy_demo.subproblems.load()
    
    
    # Run row generation method
    theta_hat = greedy_demo.row_generation.solve()

    ########## Modular BLP inversion ##########
    if rank == 0:
        print(f"[Rank 0] theta_hat: {theta_hat}")
        input_data["item_data"]["modular"] = np.eye(num_items)
        input_data["obs_bundle"] = obs_bundles
        input_data["errors"] = estimation_errors 
    else:
        input_data = None
    cfg["dimensions"]["num_simuls"] = num_simuls
    cfg["dimensions"]["num_features"] = num_agent_features + num_items + 1
    cfg["row_generation"]["theta_lbs"] = [0] * num_agent_features + [-500] * num_items + [0]
    cfg["row_generation"]["theta_ubs"] = 500
    parameters_to_log = [i for i in range(num_agent_features)] + [-1]
    cfg["row_generation"]["parameters_to_log"] = parameters_to_log
    greedy_demo.load_config(cfg)
    greedy_demo.data.load_and_scatter(input_data)
    greedy_demo.features.set_oracle(features_oracle)
    greedy_demo.subproblems.load()
    start_time = time.time()
    theta_hat = greedy_demo.row_generation.solve()
    if rank == 0:
        delta_hat = theta_hat[num_agent_features:-1]
        import statsmodels.api as sm
        from statsmodels.sandbox.regression.gmm import IV2SLS
        # instruments = np.concatenate([instrument[:, None], modular_item[:,1:]], axis=1)
        # iv_model = IV2SLS(delta_hat, modular_item, instruments)
        # iv_results = iv_model.fit()
        # print(f"Coefficients: {iv_results.params}")
        # print(f"Standard errors: {iv_results.bse}")
        # print(f"theta_hat: {theta_hat}")    
        # do a simple OLS
        ols_model = sm.OLS(delta_hat, modular_item)
        ols_results = ols_model.fit()
        print(f"Coefficients: {ols_results.params}")
        print(f"Standard errors: {ols_results.bse}")
        # print(f"theta_hat: {theta_hat}")    
        iv_model = IV2SLS(delta_hat, modular_item, instrument)
        iv_results = iv_model.fit()
        print(f"Coefficients: {iv_results.params}")
        print(f"Standard errors: {iv_results.bse}")
        print(f"theta_hat: {theta_hat[parameters_to_log]}")    


def features_oracle(i_id, B_j, data):
    """
    Compute features for a given agent and bundle(s).
    Supports both single (1D) and multiple (2D) bundles.
    Returns array of shape (num_features,) for a single bundle,
    or (num_features, m) for m bundles.
    """
    
    modular_agent = data["agent_data"]["modular"][i_id]
    modular_item = data["item_data"]["modular"]

    modular_item = np.atleast_2d(modular_item)
    modular_agent = np.atleast_2d(modular_agent)
    modular = np.concatenate([modular_agent, modular_item], axis=1)

    single_bundle = False
    if B_j.ndim == 1:
        B_j = B_j[:, None]
        single_bundle = True
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        modular_features = modular.T @ B_j
    neg_sq = -np.sum(B_j, axis=0, keepdims=True) ** 2

    features = np.vstack((modular_features, neg_sq))
    if single_bundle:
        return features[:, 0]  # Return as 1D array for a single bundle
    return features


if __name__ == "__main__":
    run_row_generation_greedy_experiment() 