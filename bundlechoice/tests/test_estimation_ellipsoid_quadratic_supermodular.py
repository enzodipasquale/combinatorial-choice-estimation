import numpy as np
import pytest
from mpi4py import MPI
from bundlechoice.core import BundleChoice
from bundlechoice.estimation.ellipsoid import EllipsoidManager
from bundlechoice.scenarios import ScenarioLibrary
from bundlechoice.scenarios.data_generator import QuadraticGenerationMethod

def test_ellipsoid_quadsupermodular():
    num_obs = 100
    num_items = 30
    num_modular_agent_features = 2
    num_modular_item_features = 2
    num_quadratic_agent_features = 0
    num_quadratic_item_features = 2
    num_features = num_modular_agent_features + num_modular_item_features + num_quadratic_agent_features + num_quadratic_item_features
    num_simulations = 1
    sigma = 5.0
    seed = 42
    scenario = ScenarioLibrary.quadratic_supermodular().with_dimensions(num_obs=num_obs, num_items=num_items).with_feature_counts(num_mod_agent=num_modular_agent_features, num_mod_item=num_modular_item_features, num_quad_agent=num_quadratic_agent_features, num_quad_item=num_quadratic_item_features).with_num_simulations(num_simulations).with_sigma(sigma).with_agent_modular_config(multiplier=-2.0, mean=2.0, std=1.0).with_quadratic_method(method=QuadraticGenerationMethod.EXPONENTIAL, mask_threshold=0.3).build()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    prepared = scenario.prepare(comm=comm, timeout_seconds=300, seed=seed)
    theta_0 = prepared.theta_star
    quad_demo = BundleChoice()
    prepared.apply(quad_demo, comm=comm, stage='generation')
    observed_bundles = quad_demo.subproblems.init_and_solve(theta_0)
    if rank == 0 and observed_bundles is not None:
        total_demand = observed_bundles.sum(1)
        print('Demand range:', total_demand.min(), total_demand.max())
    prepared.apply(quad_demo, comm=comm, stage='estimation')
    quad_demo.subproblems.load()
    result = quad_demo.ellipsoid.solve()
    theta_hat = result.theta_hat
    obj_at_theta_0 = quad_demo.ellipsoid.objective(theta_0)
    obj_at_theta_hat = quad_demo.ellipsoid.objective(theta_hat)
    if rank == 0:
        print('theta_hat:', theta_hat)
        print('theta_0:', theta_0)
        assert theta_hat.shape == (num_features,)
        assert not np.any(np.isnan(theta_hat))
        assert np.all(np.isfinite(theta_hat))
        assert np.any(theta_hat != 0)
        assert np.all(np.abs(theta_hat) < 100)
        print('obj_at_theta_0', obj_at_theta_0)
        print('obj_at_theta_hat', obj_at_theta_hat)