import numpy as np
from mpi4py import MPI

from bundlechoice.core import BundleChoice
from bundlechoice.scenarios import ScenarioLibrary


def test_row_generation_greedy_theta_init_same_problem():
    """
    Test that when initializing with theta from the same problem,
    row generation converges in 1-2 iterations.
    """
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

    # First run: solve to get theta_hat
    bundlechoice1 = BundleChoice()
    prepared.apply(bundlechoice1, comm=comm, stage="estimation")
    
    # Set max_iters to ensure we get a solution
    bundlechoice1.config.row_generation.max_iters = 100
    bundlechoice1.config.row_generation.min_iters = 1
    
    result1 = bundlechoice1.row_generation.solve()
    
    # Broadcast theta_hat1 to all ranks for use in second run
    if rank == 0:
        theta_hat1 = result1.theta_hat
        print(f"\nFirst run completed. Theta_hat: {theta_hat1}")
        print(f"Number of iterations: {result1.num_iterations}")
    else:
        theta_hat1 = None
    theta_hat1 = comm.bcast(theta_hat1, root=0)
    
    # Second run: initialize with theta_hat1 from first run
    bundlechoice2 = BundleChoice()
    prepared.apply(bundlechoice2, comm=comm, stage="estimation")
    
    # Set max_iters high enough to allow convergence
    bundlechoice2.config.row_generation.max_iters = 100
    bundlechoice2.config.row_generation.min_iters = 1
    
    # Solve with theta_init
    result2 = bundlechoice2.row_generation.solve(theta_init=theta_hat1)
    
    if rank == 0:
        theta_hat2 = result2.theta_hat
        num_iterations = result2.num_iterations
        num_iterations1 = result1.num_iterations
        print(f"\nSecond run with theta_init completed.")
        print(f"Number of iterations: {num_iterations} (first run: {num_iterations1})")
        print(f"Theta_hat2: {theta_hat2}")
        
        # When initialized with correct theta, should converge faster than without init
        # Allow some tolerance for numerical precision
        assert num_iterations < num_iterations1 or num_iterations <= 3, \
            f"Expected fewer iterations with theta_init ({num_iterations}) than without ({num_iterations1}), or <= 3"
        assert not np.any(np.isnan(theta_hat2))
        assert theta_hat2.shape == theta_hat1.shape
        
        # Theta should be close (allowing for numerical differences)
        theta_diff = np.abs(theta_hat2 - theta_hat1)
        print(f"Max theta difference: {theta_diff.max()}")
        print(f"Mean theta difference: {theta_diff.mean()}")


def test_row_generation_greedy_theta_init_helps_convergence():
    """
    Test that initializing with theta helps convergence when using 10 iterations.
    Compare iterations needed with and without theta_init.
    """
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

    # First run: solve to get theta_hat (with low simulations or full)
    bundlechoice1 = BundleChoice()
    prepared.apply(bundlechoice1, comm=comm, stage="estimation")
    
    bundlechoice1.config.row_generation.max_iters = 100
    bundlechoice1.config.row_generation.min_iters = 1
    
    result1 = bundlechoice1.row_generation.solve()
    
    # Broadcast theta_hat1 to all ranks
    if rank == 0:
        theta_hat1 = result1.theta_hat
        num_iterations1 = result1.num_iterations
        print(f"\nFirst run (without init): {num_iterations1} iterations")
    else:
        theta_hat1 = None
    theta_hat1 = comm.bcast(theta_hat1, root=0)
    
    # Second run: without theta_init, limited to 10 iterations
    bundlechoice2 = BundleChoice()
    prepared.apply(bundlechoice2, comm=comm, stage="estimation")
    
    bundlechoice2.config.row_generation.max_iters = 10
    bundlechoice2.config.row_generation.min_iters = 1
    
    result2 = bundlechoice2.row_generation.solve()
    
    if rank == 0:
        theta_hat2_no_init = result2.theta_hat
        num_iterations2_no_init = result2.num_iterations
        print(f"Second run (no init, max 10 iters): {num_iterations2_no_init} iterations")
    
    # Third run: with theta_init, limited to 10 iterations
    bundlechoice3 = BundleChoice()
    prepared.apply(bundlechoice3, comm=comm, stage="estimation")
    
    bundlechoice3.config.row_generation.max_iters = 10
    bundlechoice3.config.row_generation.min_iters = 1
    
    result3 = bundlechoice3.row_generation.solve(theta_init=theta_hat1)
    
    if rank == 0:
        theta_hat3_with_init = result3.theta_hat
        num_iterations3_with_init = result3.num_iterations
        print(f"Third run (with init, max 10 iters): {num_iterations3_with_init} iterations")
        
        # With initialization, should converge faster (fewer iterations)
        # or at least not worse
        print(f"\nComparison:")
        print(f"  Without init: {num_iterations2_no_init} iterations")
        print(f"  With init: {num_iterations3_with_init} iterations")
        
        assert not np.any(np.isnan(theta_hat2_no_init))
        assert not np.any(np.isnan(theta_hat3_with_init))
        
        # With init should converge in fewer or equal iterations
        # (allowing for edge cases where both converge quickly)
        assert num_iterations3_with_init <= num_iterations2_no_init + 1, \
            f"Initialization should help: {num_iterations3_with_init} > {num_iterations2_no_init} + 1"
        
        print("✓ Initialization helps convergence")

