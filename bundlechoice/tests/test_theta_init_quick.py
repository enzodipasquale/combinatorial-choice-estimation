#!/usr/bin/env python
"""Quick test for theta_init functionality - standalone to avoid pytest cleanup issues."""
import numpy as np
from mpi4py import MPI
from bundlechoice.core import BundleChoice
from bundlechoice.factory import ScenarioLibrary

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    scenario = (
        ScenarioLibrary.greedy()
        .with_dimensions(num_agents=100, num_items=100)
        .with_num_features(6)
        .with_num_simuls(1)
        .build()
    )
    
    prepared = scenario.prepare(comm=comm, timeout_seconds=300, seed=42)
    
    # First run
    bc1 = BundleChoice()
    prepared.apply(bc1, comm=comm, stage="estimation")
    bc1.config.row_generation.max_iters = 100
    bc1.config.row_generation.min_iters = 1
    
    theta1 = bc1.row_generation.solve()
    
    if rank == 0:
        iter1 = bc1.row_generation_manager.timing_stats['num_iterations']
        print(f"First run: {iter1} iterations")
        print(f"Theta1: {theta1}")
    
    # Second run with theta_init
    bc2 = BundleChoice()
    prepared.apply(bc2, comm=comm, stage="estimation")
    bc2.config.row_generation.max_iters = 10
    bc2.config.row_generation.min_iters = 1
    
    theta2 = bc2.row_generation.solve(theta_init=theta1)
    
    if rank == 0:
        iter2 = bc2.row_generation_manager.timing_stats['num_iterations']
        print(f"\nSecond run with theta_init: {iter2} iterations")
        print(f"Theta2: {theta2}")
        print(f"Max diff: {np.abs(theta2 - theta1).max()}")
        print(f"Converged in <= 2 iterations: {iter2 <= 2}")

if __name__ == "__main__":
    main()

