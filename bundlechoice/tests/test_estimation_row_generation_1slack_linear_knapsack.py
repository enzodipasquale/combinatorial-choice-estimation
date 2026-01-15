import numpy as np
import pytest
from mpi4py import MPI
from bundlechoice.core import BundleChoice
from bundlechoice.estimation import RowGeneration1SlackManager
from bundlechoice.scenarios import ScenarioLibrary

def test_row_generation_1slack_linear_knapsack():
    num_obs = 250
    num_items = 20
    num_modular_agent_features = 2
    num_modular_item_features = 2
    num_features = num_modular_agent_features + num_modular_item_features
    num_simulations = 1
    seed = 42
    scenario = ScenarioLibrary.linear_knapsack().with_dimensions(num_obs=num_obs, num_items=num_items).with_feature_counts(num_agent_features=num_modular_agent_features, num_item_features=num_modular_item_features).with_num_simulations(num_simulations).with_sigma(1.0).with_random_capacity(low=1, high=100).build()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    prepared = scenario.prepare(comm=comm, timeout_seconds=300, seed=seed)
    theta_0 = prepared.theta_star
    knapsack_demo = BundleChoice()
    prepared.apply(knapsack_demo, comm=comm, stage='generation')
    observed_bundles = knapsack_demo.subproblems.initialize_and_solve_subproblems(theta_0)
    if rank == 0 and observed_bundles is not None:
        print('Total demand:', observed_bundles.sum(1).min(), observed_bundles.sum(1).max())
    prepared.apply(knapsack_demo, comm=comm, stage='estimation')
    knapsack_demo.subproblems.load()
    solver = RowGeneration1SlackManager(comm_manager=knapsack_demo.comm_manager, dimensions_cfg=knapsack_demo.config.dimensions, row_generation_cfg=knapsack_demo.config.row_generation, data_manager=knapsack_demo.data_manager, oracles_manager=knapsack_demo.oracles_manager, subproblem_manager=knapsack_demo.subproblem_manager)
    result = solver.solve()
    if rank == 0:
        theta_hat = result.theta_hat
        print('theta_hat:', theta_hat)
        print('theta_0:', theta_0)
        assert theta_hat.shape == (num_features,)
        assert not np.any(np.isnan(theta_hat))
        param_error = np.linalg.norm(theta_hat - theta_0)
        print(f'Parameter recovery error (L2 norm): {param_error:.6f}')
        assert param_error < 1.0, f'Parameter recovery error too large: {param_error}'
        relative_errors = np.abs(theta_hat - theta_0) / (np.abs(theta_0) + 1e-08)
        max_relative_error = np.max(relative_errors)
        print(f'Max relative error: {max_relative_error:.6f}')
        assert max_relative_error < 0.5, f'Max relative error too large: {max_relative_error}'