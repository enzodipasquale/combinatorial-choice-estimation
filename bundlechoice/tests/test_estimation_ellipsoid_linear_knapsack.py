import numpy as np
import pytest
from mpi4py import MPI
from bundlechoice.core import BundleChoice
from bundlechoice.estimation.ellipsoid import EllipsoidManager
from bundlechoice.factory import ScenarioLibrary


def test_ellipsoid_linear_knapsack():
    """Test EllipsoidSolver using observed bundles generated from linear knapsack subproblem manager."""
    num_agents = 100
    num_items = 20
    num_modular_agent_features = 2
    num_modular_item_features = 2
    num_features = num_modular_agent_features + num_modular_item_features
    num_simuls = 1
    seed = 42
    
    # Use factory to generate data (matches manual: abs(normal), weights 1-10, random capacity 1-100)
    scenario = (
        ScenarioLibrary.linear_knapsack()
        .with_dimensions(num_agents=num_agents, num_items=num_items)
        .with_feature_counts(num_agent_features=num_modular_agent_features, num_item_features=num_modular_item_features)
        .with_num_simuls(num_simuls)
        .with_sigma(1.0)
        .with_random_capacity(low=1, high=100)  # Match manual: random capacity
        .build()
    )
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    prepared = scenario.prepare(comm=comm, timeout_seconds=300, seed=seed)
    theta_0 = prepared.theta_star
        
    knapsack_demo = BundleChoice()
    prepared.apply(knapsack_demo, comm=comm, stage="generation")
    
    observed_bundles = knapsack_demo.subproblems.init_and_solve(theta_0)
    
    if rank == 0 and observed_bundles is not None:
        print("Total demand:", observed_bundles.sum(1).min(), observed_bundles.sum(1).max())

    # Apply estimation data
    prepared.apply(knapsack_demo, comm=comm, stage="estimation")
    knapsack_demo.subproblems.load()

    result = knapsack_demo.ellipsoid.solve()
    
    # Extract theta_hat on all ranks (result object exists on all ranks)
    theta_hat = result.theta_hat
    
    # Check objective values on all ranks
    obj_at_theta_0 = knapsack_demo.ellipsoid.objective(theta_0)
    obj_at_theta_hat = knapsack_demo.ellipsoid.objective(theta_hat)
    if rank == 0:
        print("theta_hat:", theta_hat)
        print("theta_0:", theta_0)
        print("obj_at_theta_0", obj_at_theta_0)
        print("obj_at_theta_hat", obj_at_theta_hat) 
    
