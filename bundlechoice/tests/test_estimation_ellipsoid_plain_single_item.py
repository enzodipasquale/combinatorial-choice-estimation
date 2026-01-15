import numpy as np
import pytest
import time
from mpi4py import MPI
from bundlechoice.core import BundleChoice
from bundlechoice.estimation.ellipsoid import EllipsoidManager
from bundlechoice.scenarios import ScenarioLibrary

def test_ellipsoid_plain_single_item():
    num_obs = 1000
    num_items = 100
    num_modular_agent_features = 10
    num_modular_item_features = 1
    num_features = num_modular_agent_features + num_modular_item_features
    num_simulations = 1
    sigma = 1
    seed = 42
    scenario = ScenarioLibrary.plain_single_item().with_dimensions(num_obs=num_obs, num_items=num_items).with_feature_counts(num_agent_features=num_modular_agent_features, num_item_features=num_modular_item_features).with_num_simulations(num_simulations).with_sigma(sigma).with_correlation(enabled=False).build()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    prepared = scenario.prepare(comm=comm, timeout_seconds=300, seed=seed)
    theta_0 = prepared.theta_star
    demo = BundleChoice()
    prepared.apply(demo, comm=comm, stage='generation')
    observed_bundles = demo.subproblems.initialize_and_solve_subproblems(theta_0)
    if rank == 0:
        assert observed_bundles is not None, 'observed_bundles is None!'
        assert observed_bundles.shape == (num_obs, num_items)
        assert np.all(observed_bundles.sum(axis=1) <= 1), 'Each agent should select at most one item.'
        modular_agent = prepared.generation_data['agent_data']['modular']
        modular_item = prepared.generation_data['item_data']['modular']
        errors = prepared.generation_data['errors']
        agent_util = np.einsum('aij,j->ai', modular_agent, theta_0[:num_modular_agent_features])
        item_util = np.dot(modular_item, theta_0[num_modular_agent_features:])
        total_util = agent_util + item_util + errors
        j_star = np.argmax(total_util, axis=1)
        for i in range(num_obs):
            if observed_bundles[i, :].sum() == 1:
                assert observed_bundles[i, j_star[i]] == 1, f'Agent {i} did not select the max utility item.'
            else:
                assert np.all(total_util[i, :] <= 0), f'Agent {i} made no selection, but has positive utility: {total_util[i, :]}'
        print('theta_0:\n', theta_0)
    prepared.apply(demo, comm=comm, stage='estimation')
    demo.subproblems.load()
    tic = time.time()
    result = demo.ellipsoid.solve()
    toc = time.time()
    theta_hat = result.theta_hat