import numpy as np
import pytest
from mpi4py import MPI
from bundlechoice.core import BundleChoice
from bundlechoice.estimation.ellipsoid import EllipsoidManager
from bundlechoice.scenarios import ScenarioLibrary

def test_ellipsoid_greedy():
    num_obs = 300
    num_items = 50
    num_features = 4
    num_simulations = 1
    seed = 42
    scenario = ScenarioLibrary.greedy().with_dimensions(num_obs=num_obs, num_items=num_items).with_num_features(num_features).with_num_simulations(num_simulations).with_sigma(1.0).build()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    prepared = scenario.prepare(comm=comm, timeout_seconds=300, seed=seed)
    theta_0 = prepared.theta_star
    greedy_demo = BundleChoice()
    prepared.apply(greedy_demo, comm=comm, stage='generation')
    observed_bundles = greedy_demo.subproblems.initialize_and_solve_subproblems(theta_0)
    if rank == 0:
        print(f'aggregate demands: {observed_bundles.sum(1).min()}, {observed_bundles.sum(1).max()}')
        print(f'aggregate: {observed_bundles.sum()}')
    prepared.apply(greedy_demo, comm=comm, stage='estimation')
    greedy_demo.subproblems.load()
    result = greedy_demo.ellipsoid.solve()
    theta_hat = result.theta_hat
    if rank == 0:
        print('theta_hat:', theta_hat)
        print('theta_0:', theta_0)
        assert theta_hat.shape == (num_features,)
        assert not np.any(np.isnan(theta_hat))
        assert np.all(np.isfinite(theta_hat))
        assert np.any(theta_hat != 0)
        assert np.all(np.abs(theta_hat) < 100)
    obj_at_theta_0 = greedy_demo.ellipsoid.compute_obj(theta_0)
    obj_at_theta_hat = greedy_demo.ellipsoid.compute_obj(theta_hat)
    if rank == 0:
        print('obj_at_theta_0', obj_at_theta_0)
        print('obj_at_theta_hat', obj_at_theta_hat)