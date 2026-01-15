import numpy as np
import pytest
from mpi4py import MPI
from bundlechoice.core import BundleChoice
from bundlechoice.estimation.ellipsoid import EllipsoidManager
from bundlechoice.scenarios import ScenarioLibrary

def test_ellipsoid_linear_knapsack():
    num_obs = 100
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
    observed_bundles = knapsack_demo.subproblems.init_and_solve(theta_0)
    if rank == 0 and observed_bundles is not None:
        print('Total demand:', observed_bundles.sum(1).min(), observed_bundles.sum(1).max())
    prepared.apply(knapsack_demo, comm=comm, stage='estimation')
    knapsack_demo.subproblems.load()
    result = knapsack_demo.ellipsoid.solve()
    theta_hat = result.theta_hat
    obj_at_theta_0 = knapsack_demo.ellipsoid.objective(theta_0)
    obj_at_theta_hat = knapsack_demo.ellipsoid.objective(theta_hat)
    if rank == 0:
        print('theta_hat:', theta_hat)
        print('theta_0:', theta_0)
        print('obj_at_theta_0', obj_at_theta_0)
        print('obj_at_theta_hat', obj_at_theta_hat)