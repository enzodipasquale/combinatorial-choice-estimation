import numpy as np
import pytest
from mpi4py import MPI
from bundlechoice.core import BundleChoice
from bundlechoice.estimation.ellipsoid import EllipsoidManager
from bundlechoice.scenarios import ScenarioLibrary


def test_ellipsoid_greedy():
    """Test EllipsoidSolver using observed bundles generated from greedy subproblem manager."""
    num_agents = 300
    num_items = 50
    num_features = 4
    num_simulations = 1
    seed = 42
    
    # Use factory to generate data (matches manual: abs(normal(0,1)))
    scenario = (
        ScenarioLibrary.greedy()
        .with_dimensions(num_agents=num_agents, num_items=num_items)
        .with_num_features(num_features)
        .with_num_simulations(num_simulations)
        .with_sigma(1.0)
        .build()
    )
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    prepared = scenario.prepare(comm=comm, timeout_seconds=300, seed=seed)
    theta_0 = prepared.theta_star

    # Initialize BundleChoice
    greedy_demo = BundleChoice()
    prepared.apply(greedy_demo, comm=comm, stage="generation")

    # Generate observed bundles
    observed_bundles = greedy_demo.subproblems.init_and_solve(theta_0)

    # Estimate parameters using ellipsoid method
    if rank == 0:
        print(f"aggregate demands: {observed_bundles.sum(1).min()}, {observed_bundles.sum(1).max()}")
        print(f"aggregate: {observed_bundles.sum()}")

    # Apply estimation data
    prepared.apply(greedy_demo, comm=comm, stage="estimation")
    greedy_demo.subproblems.load()
    theta_ = np.ones(num_features) 
    # theta_[0] = .99
    gradient = greedy_demo.ellipsoid.obj_gradient(theta_)
 
    
    result = greedy_demo.ellipsoid.solve()
    
    # Extract theta_hat on all ranks (result object exists on all ranks)
    theta_hat = result.theta_hat
    
    if rank == 0:
        print("theta_hat:", theta_hat)
        print("theta_0:", theta_0)
        assert theta_hat.shape == (num_features,)
        assert not np.any(np.isnan(theta_hat))
        # Additional assertions for ellipsoid method
        assert np.all(np.isfinite(theta_hat))
        # Check that the solution is reasonable (not all zeros or extreme values)
        assert np.any(theta_hat != 0)
        assert np.all(np.abs(theta_hat) < 100)  # Reasonable bounds
    
    # Check objective values on all ranks
    obj_at_theta_0 = greedy_demo.ellipsoid.objective(theta_0)
    obj_at_theta_hat = greedy_demo.ellipsoid.objective(theta_hat)
    if rank == 0:
        print("obj_at_theta_0", obj_at_theta_0)
        print("obj_at_theta_hat", obj_at_theta_hat)

