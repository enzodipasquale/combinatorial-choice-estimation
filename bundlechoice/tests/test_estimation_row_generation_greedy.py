import numpy as np
from mpi4py import MPI

from bundlechoice.core import BundleChoice
from bundlechoice.scenarios import ScenarioLibrary


def test_row_generation_greedy():
    """Row generation estimation using the shared greedy synthetic scenario."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    scenario = (
        ScenarioLibrary.greedy()
        .with_dimensions(num_agents=250, num_items=50)
        .with_num_features(6)
        .with_num_simuls(1)
        .build()
    )

    prepared = scenario.prepare(comm=comm, timeout_seconds=300, seed=42)

    bundlechoice = BundleChoice()
    prepared.apply(bundlechoice, comm=comm, stage="estimation")

    result = bundlechoice.row_generation.solve()

    if rank == 0:
        theta_hat = result.theta_hat
        theta_0 = prepared.theta_star
        obs_bundles = prepared.estimation_data["obs_bundle"]
        print(
            "aggregate demands:",
            obs_bundles.sum(1).min(),
            obs_bundles.sum(1).max(),
        )
        print("aggregate:", obs_bundles.sum())
        print("theta_hat:", theta_hat)
        print("theta_0:", theta_0)
        assert theta_hat.shape == theta_0.shape
        assert not np.any(np.isnan(theta_hat))

