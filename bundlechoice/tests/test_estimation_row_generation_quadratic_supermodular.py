import numpy as np
from mpi4py import MPI

from bundlechoice.core import BundleChoice
from bundlechoice.factory import ScenarioLibrary


def test_row_generation_quadsupermodular():
    """Quadratic supermodular estimation using the synthetic scenario helper."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    scenario = (
        ScenarioLibrary.quadratic_supermodular()
        .with_dimensions(num_agents=20, num_items=50)
        .with_feature_counts(
            num_mod_agent=2,
            num_mod_item=2,
            num_quad_agent=0,
            num_quad_item=2,
        )
        .with_num_simuls(1)
        .build()
    )

    prepared = scenario.prepare(comm=comm, timeout_seconds=300, seed=7)

    bundlechoice = BundleChoice()
    prepared.apply(bundlechoice, comm=comm, stage="estimation")

    theta_hat = bundlechoice.row_generation.solve()

    if rank == 0:
        theta_0 = prepared.theta_star
        obs_bundles = prepared.estimation_data["obs_bundle"]
        print("theta_hat:", theta_hat)
        print("theta_0:", theta_0)
        assert obs_bundles is not None
        num_agents = prepared.metadata["num_agents"]
        num_items = prepared.metadata["num_items"]
        assert obs_bundles.shape == (num_agents, num_items)
        assert theta_hat.shape == theta_0.shape
        assert not np.any(np.isnan(theta_hat))
