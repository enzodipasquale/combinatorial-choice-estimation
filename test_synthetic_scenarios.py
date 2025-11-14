#!/usr/bin/env python3
"""Test script for synthetic scenario helpers with timeout wrapper."""

import signal
import sys
import time

import numpy as np
from mpi4py import MPI

from bundlechoice.core import BundleChoice
from bundlechoice.factory import ScenarioLibrary

TIMEOUT_SECONDS = 5


def timeout_handler(signum, frame):
    """Handle timeout signal."""
    print(f"\n❌ TIMEOUT after {TIMEOUT_SECONDS} seconds", flush=True)
    sys.exit(124)


def main():
    """Test synthetic scenario helpers."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Set up timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(TIMEOUT_SECONDS)

    try:
        if rank == 0:
            print("Testing Greedy scenario...", flush=True)

        # Test greedy scenario
        scenario = (
            ScenarioLibrary.greedy()
            .with_dimensions(num_agents=50, num_items=20)
            .with_num_features(4)
            .with_num_simuls(1)
            .build()
        )

        if rank == 0:
            print("Preparing scenario...", flush=True)

        prepared = scenario.prepare(comm=comm, timeout_seconds=3, seed=42)

        if rank == 0:
            print("Applying to BundleChoice...", flush=True)

        bc = BundleChoice()
        prepared.apply(bc, comm=comm, stage="estimation")

        if rank == 0:
            print("Solving with row generation...", flush=True)

        theta_hat = bc.row_generation.solve()

        if rank == 0:
            theta_0 = prepared.theta_star
            obs_bundles = prepared.estimation_data["obs_bundle"]
            print(f"✅ Test passed!")
            print(f"   theta_hat shape: {theta_hat.shape}")
            print(f"   theta_0 shape: {theta_0.shape}")
            print(f"   obs_bundles shape: {obs_bundles.shape}")
            print(f"   aggregate demands: {obs_bundles.sum(1).min():.2f} to {obs_bundles.sum(1).max():.2f}")

        signal.alarm(0)
        return 0

    except Exception as e:
        signal.alarm(0)
        if rank == 0:
            print(f"❌ Error: {e}", flush=True)
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

