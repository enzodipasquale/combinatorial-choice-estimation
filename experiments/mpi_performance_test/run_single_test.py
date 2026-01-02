"""
Single test runner - outputs timing statistics for a specific configuration.
Used by the comparison script.
"""
import argparse
import sys
import numpy as np
from bundlechoice import BundleChoice
from bundlechoice.factory import ScenarioLibrary

def main():
    parser = argparse.ArgumentParser(description='Run single performance test')
    parser.add_argument('--agents', type=int, required=True)
    parser.add_argument('--items', type=int, required=True)
    parser.add_argument('--simuls', type=int, default=5)
    parser.add_argument('--features', type=int, default=6)
    parser.add_argument('--max-iters', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Generate data using builder pattern - use same config as run_timing_test.py
    # This ensures meaningful variation in choices (not all zeros)
    num_mod_agent = 2
    num_mod_item = 2
    num_quad_item = 2
    
    scenario = (
        ScenarioLibrary.quadratic_supermodular()
        .with_dimensions(num_agents=args.agents, num_items=args.items)
        .with_feature_counts(
            num_mod_agent=num_mod_agent,
            num_mod_item=num_mod_item,
            num_quad_item=num_quad_item,
        )
        .with_sigma(5.0)  # Keep default sigma
        .with_num_simulations(args.simuls)
        .build()
    )
    
    # Prepare scenario
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    theta_star = np.ones(args.features) * 2.0
    prepared = scenario.prepare(comm=comm, seed=args.seed, theta=theta_star)
    
    # Update config with num_simulations
    config = prepared.config.copy()
    config["dimensions"]["num_simulations"] = args.simuls
    config["row_generation"]["max_iters"] = args.max_iters
    
    # Create BundleChoice instance
    bc = BundleChoice()
    bc.load_config(config)
    bc.data.load_and_scatter(prepared.estimation_data)
    bc.features.build_from_data()
    bc.subproblems.load()
    
    # Run estimation
    result = bc.row_generation.solve()
    
    # Extract and print timing statistics
    # result is an EstimationResult object with timing field
    timing_stats = result.timing if result.timing else {}
    total_time = timing_stats.get('total_time', 0)
    num_iterations = result.num_iterations
    
    print(f"Total time: {total_time:.3f}s")
    print(f"Total iterations: {num_iterations}")
    
    # Also try to get detailed timing from row_generation object if available
    if hasattr(bc.row_generation, 'timing_stats') and bc.row_generation.timing_stats:
        detailed_timing = bc.row_generation.timing_stats
        # Print key timing components
        components = ['pricing_time', 'mpi_time', 'master_time']
        for comp in components:
            val = detailed_timing.get(comp, 0)
            if val > 0:
                print(f"{comp}: {val:.3f}s")

if __name__ == "__main__":
    main()

