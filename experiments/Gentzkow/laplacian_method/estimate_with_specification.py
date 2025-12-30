#!/usr/bin/env python3
"""
Estimate parameters with correctly specified and misspecified models.

Correctly specified: Accounts for correlation in time-invariant errors.
Misspecified: Ignores correlation in time-invariant errors (treats them as i.i.d.).
"""
import numpy as np
from mpi4py import MPI

from bundlechoice.core import BundleChoice
from bundlechoice.factory import ScenarioLibrary
from bundlechoice.factory.data_generator import QuadraticGenerationMethod


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("=" * 80)
        print("Estimation with Correctly Specified vs Misspecified Model")
        print("=" * 80)
        print()
    
    # Setup scenario
    true_alpha = 1.0
    custom_theta = np.array([0.9, 0.6, -0.4, 0.5, 0.3, 0.2])
    
    scenario = (
        ScenarioLibrary.gentzkow()
        .with_dimensions(num_agents=300, num_items_per_period=30)
        .with_feature_counts(num_mod_agent=2, num_mod_item=2, num_quad_item=2)
        .with_num_simuls(1)
        .with_sigma(3.0)
        .with_sigma_time_invariant(2.0)
        .with_time_invariant_alpha(true_alpha)
        .with_agent_modular_config(multiplier=-1.0, mean=2.0, std=1.0)
        .with_item_modular_config(multiplier=-1.0, mean=2.0, std=1.0)
        .with_quadratic_method(
            method=QuadraticGenerationMethod.BINARY_CHOICE,
            binary_prob=0.25,
            binary_value=1.0,
            mask_threshold=0.4,
        )
        .with_theta(custom_theta)
        .build()
    )
    
    prepared = scenario.prepare(comm=comm, timeout_seconds=600, seed=42)
    
    if rank == 0:
        print(f"Problem dimensions: {prepared.metadata['num_agents']} agents, "
              f"{prepared.metadata['num_items']} items ({prepared.metadata['num_periods']} periods)")
        print(f"True alpha: {true_alpha}")
        print(f"True theta: {custom_theta}\n")
    
    # Store the generated data (observed bundles and errors with correlation)
    comm.Barrier()  # Ensure all ranks are synchronized
    
    if rank == 0:
        correct_data = prepared.estimation_data.copy()
        observed_bundles = correct_data["obs_bundle"].copy()
        correct_errors = correct_data["errors"].copy()
        
        # Generate i.i.d. errors for misspecified model
        from bundlechoice.factory.data_generator import DataGenerator
        generator = DataGenerator(seed=999)
        
        num_simuls, num_agents, num_items = correct_errors.shape
        num_items_per_period = prepared.metadata['num_items_per_period']
        num_periods = prepared.metadata['num_periods']
        sigma = 3.0
        sigma_time_invariant = 2.0
        
        # Generate i.i.d. time-invariant errors (no correlation)
        iid_time_invariant = generator.rng.normal(
            0, sigma_time_invariant, (num_agents, num_items_per_period)
        )
        
        # Copy to all periods (time-invariant)
        time_invariant_errors_iid = np.zeros((num_agents, num_items))
        for period in range(num_periods):
            start_idx = period * num_items_per_period
            end_idx = (period + 1) * num_items_per_period
            time_invariant_errors_iid[:, start_idx:end_idx] = iid_time_invariant
        
        # Generate new errors: i.i.d. simulation + i.i.d. time-invariant
        misspec_errors = np.zeros_like(correct_errors)
        for simul in range(num_simuls):
            iid_simul = generator.rng.normal(0, sigma, (num_agents, num_items))
            misspec_errors[simul] = iid_simul + time_invariant_errors_iid
        
        misspec_data = correct_data.copy()
        misspec_data["errors"] = misspec_errors
        misspec_data["obs_bundle"] = observed_bundles  # Same observed bundles
    else:
        correct_data = None
        misspec_data = None
    
    comm.Barrier()  # Ensure data preparation is complete
    
    # ============================================================================
    # ESTIMATION 1: CORRECTLY SPECIFIED (with correlation)
    # ============================================================================
    if rank == 0:
        print("=" * 80)
        print("ESTIMATION 1: CORRECTLY SPECIFIED (errors with correlation)")
        print("=" * 80)
        print()
    
    bundlechoice_correct = BundleChoice()
    prepared.apply(bundlechoice_correct, comm=comm, stage="estimation")
    
    # Replace with correct errors (with correlation) - same as generated data
    if rank == 0:
        bundlechoice_correct.data_manager.load_and_scatter(correct_data)
    else:
        bundlechoice_correct.data_manager.scatter()
    
    # Reinitialize subproblems and row generation with new data
    bundlechoice_correct.subproblems.initialize_local()
    bundlechoice_correct.row_generation._initialize_master_problem(initial_constraints=None)
    
    result_correct = bundlechoice_correct.row_generation.solve(callback=None)
    theta_hat_correct = result_correct.theta_hat
    
    if rank == 0:
        print(f"\nEstimated theta (correctly specified): {theta_hat_correct}")
        print(f"True theta: {custom_theta}")
        l2_error_correct = np.linalg.norm(theta_hat_correct - custom_theta)
        print(f"L2 error: {l2_error_correct:.6f}")
        
        # Print row generation stats
        if result_correct.timing is not None:
            stats = result_correct.timing
            print(f"\nRow Generation Stats (Correctly Specified):")
            print(f"  Total time: {stats.get('total_time', 0):.2f}s")
            print(f"  Init time: {stats.get('init_time', 0):.2f}s")
            print(f"  Iterations: {result_correct.num_iterations}")
            print(f"  Pricing time: {stats.get('pricing_time', 0):.2f}s ({stats.get('pricing_time_pct', 0):.1f}%)")
            print(f"  Master time: {stats.get('master_time', 0):.2f}s ({stats.get('master_time_pct', 0):.1f}%)")
            print(f"  MPI time: {stats.get('mpi_time', 0):.2f}s ({stats.get('mpi_time_pct', 0):.1f}%)")
        print()
    
    # ============================================================================
    # ESTIMATION 2: MISSPECIFIED (i.i.d. errors, no correlation)
    # ============================================================================
    if rank == 0:
        print("=" * 80)
        print("ESTIMATION 2: MISSPECIFIED (i.i.d. errors, no correlation)")
        print("=" * 80)
        print()
    
    bundlechoice_misspec = BundleChoice()
    prepared.apply(bundlechoice_misspec, comm=comm, stage="estimation")
    
    # Replace with i.i.d. errors (no correlation) but same observed bundles
    if rank == 0:
        bundlechoice_misspec.data_manager.load_and_scatter(misspec_data)
    else:
        bundlechoice_misspec.data_manager.scatter()
    
    # Reinitialize subproblems and row generation with new data
    bundlechoice_misspec.subproblems.initialize_local()
    bundlechoice_misspec.row_generation._initialize_master_problem(initial_constraints=None)
    
    result_misspec = bundlechoice_misspec.row_generation.solve(callback=None)
    theta_hat_misspec = result_misspec.theta_hat
    
    if rank == 0:
        print(f"\nEstimated theta (misspecified): {theta_hat_misspec}")
        print(f"True theta: {custom_theta}")
        l2_error_misspec = np.linalg.norm(theta_hat_misspec - custom_theta)
        print(f"L2 error: {l2_error_misspec:.6f}")
        
        # Print row generation stats
        if result_misspec.timing is not None:
            stats = result_misspec.timing
            print(f"\nRow Generation Stats (Misspecified):")
            print(f"  Total time: {stats.get('total_time', 0):.2f}s")
            print(f"  Init time: {stats.get('init_time', 0):.2f}s")
            print(f"  Iterations: {result_misspec.num_iterations}")
            print(f"  Pricing time: {stats.get('pricing_time', 0):.2f}s ({stats.get('pricing_time_pct', 0):.1f}%)")
            print(f"  Master time: {stats.get('master_time', 0):.2f}s ({stats.get('master_time_pct', 0):.1f}%)")
            print(f"  MPI time: {stats.get('mpi_time', 0):.2f}s ({stats.get('mpi_time_pct', 0):.1f}%)")
        print()
        
        print("=" * 80)
        print("COMPARISON")
        print("=" * 80)
        print(f"Correctly specified L2 error: {l2_error_correct:.6f}")
        print(f"Misspecified L2 error:       {l2_error_misspec:.6f}")
        print(f"Difference:                   {l2_error_misspec - l2_error_correct:.6f}")
        if l2_error_misspec > l2_error_correct:
            print(f"Misspecified model is {l2_error_misspec/l2_error_correct:.2f}x worse")
        print()


if __name__ == "__main__":
    main()

