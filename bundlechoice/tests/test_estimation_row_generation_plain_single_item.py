import numpy as np
from mpi4py import MPI
from bundlechoice.core import BundleChoice
from bundlechoice.scenarios import ScenarioLibrary

def test_row_generation_plain_single_item():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    scenario = ScenarioLibrary.plain_single_item().with_dimensions(num_obs=500, num_items=2).with_feature_counts(num_agent_features=4, num_item_features=1).with_num_simulations(1).build()
    prepared = scenario.prepare(comm=comm, timeout_seconds=300, seed=21)
    bundlechoice = BundleChoice()
    prepared.apply(bundlechoice, comm=comm, stage='estimation')
    result = bundlechoice.row_generation.solve()
    if rank == 0:
        theta_hat = result.theta_hat
        theta_0 = prepared.theta_star
        obs_bundles = prepared.estimation_data['obs_bundle']
        assert obs_bundles is not None
        num_obs = prepared.metadata['num_obs']
        num_items = prepared.metadata['num_items']
        assert obs_bundles.shape == (num_obs, num_items)
        assert (obs_bundles.sum(axis=1) <= 1).all()
        print('theta_hat:', theta_hat)
        print('theta_0:', theta_0)
        assert theta_hat.shape == theta_0.shape
        assert not np.any(np.isnan(theta_hat))