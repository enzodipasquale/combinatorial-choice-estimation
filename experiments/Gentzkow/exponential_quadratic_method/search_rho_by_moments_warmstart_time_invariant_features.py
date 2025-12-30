#!/usr/bin/env python3
"""
Search over rho to minimize L2 distance between observed and generated moments.
Uses rho-based correlation method: y = sqrt(rho) * (A @ x) + sqrt(1 - rho) * z
Uses EXPONENTIAL quadratic method (instead of BINARY_CHOICE).
Uses warm-starting: reuses constraints from previous rho for faster estimation.
ALL modular features (agent and item) are time-invariant (constant across periods).
"""
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

from bundlechoice.core import BundleChoice
from bundlechoice.factory import ScenarioLibrary
from bundlechoice.factory.data_generator import QuadraticGenerationMethod, DataGenerator


def _generate_time_invariant_errors_with_rho(
    generator: DataGenerator,
    num_agents: int,
    num_items: int,
    num_items_per_period: int,
    num_periods: int,
    sigma: float,
    rho: float,
    item_quadratic: np.ndarray,
) -> np.ndarray:
    """
    Generate time-invariant errors with rho-based correlation structure.
    
    Given:
    - A is an n x n matrix with unit Euclidean norm rows (derived from Q)
    - rho is a parameter in [0,1]
    - x ~ N(0, I_n) is a standard normal vector
    - z ~ N(0, I_n) is an independent standard normal vector
    
    Define: y = sqrt(rho) * (A @ x) + sqrt(1 - rho) * z
    
    Properties:
    - Each entry of y has variance 1
    - Cov(y) = rho * (A @ A.T) + (1 - rho) * I
    - rho = 1 gives fully correlated, rho = 0 gives independent
    """
    errors = np.zeros((num_agents, num_items))
    
    # Time-invariant errors are the SAME across periods.
    period = 0
    start_idx = period * num_items_per_period
    end_idx = (period + 1) * num_items_per_period
    period_items = num_items_per_period
    
    # Extract Q block for period 0
    Q_period = item_quadratic[start_idx:end_idx, start_idx:end_idx, :]
    Q_sum = Q_period.sum(axis=2)  # Sum across features: (period_items, period_items)
    
    # Make Q_sum symmetric
    Q_symmetric = Q_sum + Q_sum.T - np.diag(np.diag(Q_sum))
    
    # Create matrix A from Q: normalize rows to unit Euclidean norm
    A = Q_symmetric.copy()
    row_norms = np.linalg.norm(A, axis=1, keepdims=True)
    # Avoid division by zero (if a row is all zeros, keep it as zeros)
    row_norms = np.where(row_norms > 1e-10, row_norms, 1.0)
    A = A / row_norms  # Now each row has unit norm
    
    # Generate correlated errors ONCE for period 0
    time_invariant_errors = np.zeros((num_agents, period_items))
    for agent in range(num_agents):
        # Generate independent standard normal vectors
        x = generator.rng.normal(0, 1, period_items)  # x ~ N(0, I)
        z = generator.rng.normal(0, 1, period_items)  # z ~ N(0, I), independent of x
        
        # Compute: y = sqrt(rho) * (A @ x) + sqrt(1 - rho) * z
        y = np.sqrt(rho) * (A @ x) + np.sqrt(1 - rho) * z
        
        # Scale by sigma to get desired variance
        time_invariant_errors[agent, :] = sigma * y
    
    # Copy the SAME time-invariant errors to all periods
    for period in range(num_periods):
        start_idx = period * num_items_per_period
        end_idx = (period + 1) * num_items_per_period
        errors[:, start_idx:end_idx] = time_invariant_errors
    
    return errors


def compute_choice_pattern_moments(
    bundles: np.ndarray,
    num_items_per_period: int,
    num_periods: int,
) -> dict:
    """Compute choice pattern moments for all pairs of items across 2 periods."""
    if num_periods != 2:
        raise ValueError("This function is designed for 2-period models")
    
    num_agents, num_items = bundles.shape
    
    # Extract period bundles
    period1_items = slice(0, num_items_per_period)
    period2_items = slice(num_items_per_period, 2 * num_items_per_period)
    period1_bundles = bundles[:, period1_items].astype(bool)
    period2_bundles = bundles[:, period2_items].astype(bool)
    
    # For each unordered pair of items (i, j) where i < j
    num_pairs = num_items_per_period * (num_items_per_period - 1) // 2
    pattern_counts = np.zeros((num_pairs, 16), dtype=np.int32)
    
    pair_idx = 0
    for i in range(num_items_per_period):
        for j in range(i + 1, num_items_per_period):
            # Extract choice pattern for items i and j across both periods
            item_i_p1 = period1_bundles[:, i].astype(int)
            item_i_p2 = period2_bundles[:, i].astype(int)
            item_j_p1 = period1_bundles[:, j].astype(int)
            item_j_p2 = period2_bundles[:, j].astype(int)
            
            # Encode pattern as 4-bit integer
            patterns = (item_i_p1 + 2 * item_i_p2 + 4 * item_j_p1 + 8 * item_j_p2).astype(int)
            
            # Count occurrences of each pattern (0-15)
            for pattern in range(16):
                pattern_counts[pair_idx, pattern] = np.sum(patterns == pattern)
            
            pair_idx += 1
    
    # Compute fractions (moments) for each pair
    pattern_fractions = pattern_counts.astype(float) / num_agents
    
    # Average over all pairs
    avg_pattern_moments = pattern_fractions.mean(axis=0)
    
    # Create pattern labels
    pattern_labels = []
    for p in range(16):
        item_i_p1 = (p >> 0) & 1
        item_i_p2 = (p >> 1) & 1
        item_j_p1 = (p >> 2) & 1
        item_j_p2 = (p >> 3) & 1
        pattern_labels.append(f"{item_i_p1}{item_i_p2}{item_j_p1}{item_j_p2}")
    
    return {
        'num_pairs': num_pairs,
        'num_agents': num_agents,
        'pattern_moments': avg_pattern_moments,
        'pattern_labels': pattern_labels,
    }


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("=" * 80)
        print("Search Rho by Minimizing Moment Distance (with Warm-Starting)")
        print("Using EXPONENTIAL quadratic method")
        print("ALL modular features (agent and item) are TIME-INVARIANT")
        print("=" * 80)
        print()
    
    # Setup scenario with rho correlation method, EXPONENTIAL quadratic, and time-invariant modular features
    true_rho = 0.5
    custom_theta = np.array([0.9, 0.6, -0.4, 0.5, 0.3, 0.2])
    
    scenario = (
        ScenarioLibrary.gentzkow()
        .with_dimensions(num_agents=1500, num_items_per_period=30)
        .with_feature_counts(num_mod_agent=2, num_mod_item=2, num_quad_item=2)
        .with_num_simuls(1)
        .with_sigma(3.0)
        .with_sigma_time_invariant(2.0)
        .with_correlation_method("rho")
        .with_time_invariant_rho(true_rho)
        .with_agent_modular_config(multiplier=-1.0, mean=2.0, std=1.0)
        .with_item_modular_config(multiplier=-1.0, mean=2.0, std=1.0)
        .with_time_invariant_modular_features()  # Make ALL modular features time-invariant
        .with_quadratic_method(
            method=QuadraticGenerationMethod.EXPONENTIAL,
        )
        .with_theta(custom_theta)
        .build()
    )
    
    prepared = scenario.prepare(comm=comm, timeout_seconds=600, seed=42)
    
    if rank == 0:
        print(f"Problem dimensions: {prepared.metadata['num_agents']} agents, "
              f"{prepared.metadata['num_items']} items ({prepared.metadata['num_periods']} periods)")
        print(f"True rho: {true_rho}")
        print(f"True theta: {custom_theta}\n")
    
    comm.Barrier()
    
    # Check observed bundles for degenerate cases
    if rank == 0:
        observed_bundles = prepared.estimation_data['obs_bundle'].copy()
        bundle_sizes = observed_bundles.sum(axis=1)
        num_items = observed_bundles.shape[1]
        num_empty = np.sum(bundle_sizes == 0)
        num_full = np.sum(bundle_sizes == num_items)
        print("=" * 80)
        print("OBSERVED BUNDLE DIAGNOSTICS")
        print("=" * 80)
        print(f"Empty bundles: {num_empty}/{len(bundle_sizes)} ({100*num_empty/len(bundle_sizes):.1f}%)")
        print(f"Full bundles: {num_full}/{len(bundle_sizes)} ({100*num_full/len(bundle_sizes):.1f}%)")
        print(f"Bundle sizes: min={bundle_sizes.min()}, max={bundle_sizes.max()}, mean={bundle_sizes.mean():.2f}, std={bundle_sizes.std():.2f}")
        print()
        
        if num_empty > len(bundle_sizes) * 0.1 or num_full > len(bundle_sizes) * 0.1:
            print("WARNING: High fraction of degenerate bundles! Consider adjusting parameters.")
            print()
    
    comm.Barrier()
    
    # Extract observed bundles and store i.i.d. errors for transformation
    if rank == 0:
        num_simuls = prepared.metadata['num_simuls']
        num_agents = prepared.metadata['num_agents']
        num_items = prepared.metadata['num_items']
        num_items_per_period = prepared.metadata['num_items_per_period']
        num_periods = prepared.metadata['num_periods']
        sigma = 3.0
        sigma_time_invariant = 2.0
        
        # Get observed bundles (fixed)
        observed_bundles = prepared.estimation_data['obs_bundle'].copy()
        
        # Get item quadratic features (needed for correlation structure)
        item_quadratic = prepared.estimation_data['item_data']['quadratic']
        
        # Generate i.i.d. errors ONCE for estimation - these will be transformed for each rho
        # Use DIFFERENT seed from data generation (999) so estimation errors are independent
        generator_estimation = DataGenerator(seed=999)
        
        # Generate i.i.d. simulation errors for estimation
        iid_simulation_errors = np.zeros((num_simuls, num_agents, num_items))
        for simul in range(num_simuls):
            iid_simulation_errors[simul] = generator_estimation.rng.normal(0, sigma, (num_agents, num_items))
        
        # Generate base random draws (x, z) ONCE for estimation - these will be reused for all rho values
        # Extract Q block for period 0 to get the same A matrix
        period = 0
        start_idx = period * num_items_per_period
        end_idx = (period + 1) * num_items_per_period
        period_items = num_items_per_period
        Q_period = item_quadratic[start_idx:end_idx, start_idx:end_idx, :]
        Q_sum = Q_period.sum(axis=2)
        Q_symmetric = Q_sum + Q_sum.T - np.diag(np.diag(Q_sum))
        A = Q_symmetric.copy()
        row_norms = np.linalg.norm(A, axis=1, keepdims=True)
        row_norms = np.where(row_norms > 1e-10, row_norms, 1.0)
        A = A / row_norms
        
        # Generate x and z ONCE for all agents - these will be reused for all rho in estimation
        base_x = np.zeros((num_agents, period_items))
        base_z = np.zeros((num_agents, period_items))
        for agent in range(num_agents):
            base_x[agent] = generator_estimation.rng.normal(0, 1, period_items)
            base_z[agent] = generator_estimation.rng.normal(0, 1, period_items)
        
        # Compute observed moments (fixed reference)
        obs_moments = compute_choice_pattern_moments(
            observed_bundles,
            num_items_per_period=num_items_per_period,
            num_periods=num_periods,
        )
        obs_pattern_moments = obs_moments['pattern_moments']
        
        print("=" * 80)
        print("OBSERVED MOMENTS (from data generation)")
        print("=" * 80)
        print(f"Computed from {num_agents} agents, {num_items_per_period} items per period")
        print()
    else:
        observed_bundles = None
        item_quadratic = None
        iid_simulation_errors = None
        obs_pattern_moments = None
        num_simuls = None
        num_agents = None
        num_items = None
        num_items_per_period = None
        num_periods = None
        sigma = None
        sigma_time_invariant = None
        base_x = None
        base_z = None
        A = None
    
    comm.Barrier()
    
    # Search over rho values: 20 points from 0.0 to 1.0, ensuring 1.0 is included
    rho_values = np.linspace(0.0, 1.0, 20).tolist()
    # Ensure 1.0 is exactly included (replace closest value)
    closest_idx = min(range(len(rho_values)), key=lambda i: abs(rho_values[i] - 1.0))
    rho_values[closest_idx] = 1.0
    rho_values = sorted(rho_values)  # Keep sorted
    
    if rank == 0:
        print("=" * 80)
        print("SEARCHING OVER RHO VALUES (with warm-starting)")
        print("=" * 80)
        print(f"Rho values to test: {rho_values}")
        print()
    
    results = []
    prev_constraints = None  # Store constraints from previous iteration for warm-starting
    
    for rho_idx, rho in enumerate(rho_values):
        if rank == 0:
            print("=" * 80)
            print(f"RHO = {rho} (iteration {rho_idx + 1}/{len(rho_values)})")
            if prev_constraints is not None:
                num_prev_constraints = len(prev_constraints.get('indices', []))
                print(f"Using warm-start with {num_prev_constraints} constraints from previous rho")
            print("=" * 80)
            print()
        
        # Transform errors for this rho
        if rank == 0:
            # Generate time-invariant errors with correlation based on rho
            # Reuse the SAME base_x and base_z for all rho values
            time_invariant_errors = np.zeros((num_agents, num_items))
            period = 0
            start_idx = period * num_items_per_period
            end_idx = (period + 1) * num_items_per_period
            period_items = num_items_per_period
            
            time_invariant_errors_period = np.zeros((num_agents, period_items))
            for agent in range(num_agents):
                # Use the SAME x and z for all rho values
                x = base_x[agent]
                z = base_z[agent]
                
                # Compute: y = sqrt(rho) * (A @ x) + sqrt(1 - rho) * z
                y = np.sqrt(rho) * (A @ x) + np.sqrt(1 - rho) * z
                
                # Scale by sigma to get desired variance
                time_invariant_errors_period[agent, :] = sigma_time_invariant * y
            
            # Copy to all periods
            for period in range(num_periods):
                start_idx = period * num_items_per_period
                end_idx = (period + 1) * num_items_per_period
                time_invariant_errors[:, start_idx:end_idx] = time_invariant_errors_period
            
            # Combine: i.i.d. simulation + time-invariant (with correlation)
            estimation_errors = np.zeros((num_simuls, num_agents, num_items))
            for simul in range(num_simuls):
                estimation_errors[simul] = iid_simulation_errors[simul] + time_invariant_errors
            
            # Prepare estimation data
            estimation_data = prepared.estimation_data.copy()
            estimation_data["errors"] = estimation_errors
            estimation_data["obs_bundle"] = observed_bundles
        else:
            estimation_data = None
        
        comm.Barrier()
        
        # Estimate theta with this rho
        bundlechoice = BundleChoice()
        prepared.apply(bundlechoice, comm=comm, stage="estimation")
        
        # Replace with errors generated using this rho
        if rank == 0:
            bundlechoice.data_manager.load_and_scatter(estimation_data)
        else:
            bundlechoice.data_manager.scatter()
        
        # Reinitialize subproblems and row generation
        bundlechoice.subproblems.initialize_local()
        
        # Use warm-starting for second and subsequent rho
        if rho_idx > 0 and prev_constraints is not None:
            bundlechoice.row_generation._initialize_master_problem(initial_constraints=prev_constraints)
        else:
            bundlechoice.row_generation._initialize_master_problem(initial_constraints=None)
        
        result = bundlechoice.row_generation.solve(callback=None)
        theta_hat = result.theta_hat
        
        # Extract constraints for next iteration (warm-starting)
        if rho_idx < len(rho_values) - 1:  # Don't extract on last iteration
            prev_constraints = bundlechoice.row_generation.get_binding_constraints()
            if rank == 0 and prev_constraints is not None:
                num_constraints = len(prev_constraints.get('indices', []))
                print(f"Extracted {num_constraints} binding constraints for warm-starting next rho")
                print()
        
        if rank == 0:
            l2_error_theta = np.linalg.norm(theta_hat - custom_theta)
            print(f"Estimated theta: {theta_hat}")
            print(f"True theta: {custom_theta}")
            print(f"L2 error (theta): {l2_error_theta:.6f}")
            
            # Print timing stats
            if result.timing is not None:
                stats = result.timing
                print(f"Row generation time: {stats.get('total_time', 0):.2f}s")
            print()
        
        comm.Barrier()
        
        # Generate bundles at estimated theta
        if rank == 0:
            theta_hat_broadcast = theta_hat.copy()
        else:
            theta_hat_broadcast = np.empty(len(theta_hat), dtype=np.float64)
        comm.Bcast(theta_hat_broadcast, root=0)
        
        generated_bundles = bundlechoice.subproblems.init_and_solve(theta_hat_broadcast, return_values=False)
        
        if rank == 0:
            # Check for degenerate cases
            bundle_sizes = generated_bundles.sum(axis=1)
            num_items = generated_bundles.shape[1]
            num_empty = np.sum(bundle_sizes == 0)
            num_full = np.sum(bundle_sizes == num_items)
            if rho_idx == 0:  # Only print for first rho to avoid clutter
                print(f"Bundle size diagnostics (rho={rho}):")
                print(f"  Empty bundles: {num_empty}/{len(bundle_sizes)} ({100*num_empty/len(bundle_sizes):.1f}%)")
                print(f"  Full bundles: {num_full}/{len(bundle_sizes)} ({100*num_full/len(bundle_sizes):.1f}%)")
                print(f"  Bundle sizes: min={bundle_sizes.min()}, max={bundle_sizes.max()}, mean={bundle_sizes.mean():.2f}")
                print()
        
        if rank == 0:
            # Compute moments for generated bundles
            gen_moments = compute_choice_pattern_moments(
                generated_bundles,
                num_items_per_period=num_items_per_period,
                num_periods=num_periods,
            )
            gen_pattern_moments = gen_moments['pattern_moments']
            
            # Compute L2 distance between observed and generated moments
            moment_l2_distance = np.linalg.norm(gen_pattern_moments - obs_pattern_moments)
            
            print(f"L2 distance (moments): {moment_l2_distance:.6f}")
            print()
            
            results.append({
                'rho': rho,
                'theta_hat': theta_hat,
                'l2_error_theta': l2_error_theta,
                'moment_l2_distance': moment_l2_distance,
                'gen_moments': gen_pattern_moments,
                'timing': result.timing.get('total_time') if result.timing else None,
            })
        
        comm.Barrier()
    
    # Plot results
    if rank == 0:
        print("=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        print()
        print("Rho | L2 Error (theta) | L2 Distance (moments) | Time (s)")
        print("----|-------------------|----------------------|----------")
        for r in results:
            time_str = f"{r['timing']:.2f}" if r['timing'] is not None else "N/A"
            print(f" {r['rho']:.2f} | {r['l2_error_theta']:.6f}        | {r['moment_l2_distance']:.6f}    | {time_str}")
        print()
        
        # Plot
        rhos = [r['rho'] for r in results]
        moment_distances = [r['moment_l2_distance'] for r in results]
        theta_errors = [r['l2_error_theta'] for r in results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Moment L2 distance vs rho
        ax1.plot(rhos, moment_distances, 'o-', linewidth=1.5, markersize=4)
        ax1.axvline(true_rho, color='r', linestyle='--', linewidth=2, label=f'True rho = {true_rho}')
        # Highlight minimum
        min_idx = np.argmin(moment_distances)
        ax1.plot(rhos[min_idx], moment_distances[min_idx], 'g*', markersize=15, label=f'Minimum at ρ={rhos[min_idx]:.2f}')
        ax1.set_xlabel('Rho', fontsize=12)
        ax1.set_ylabel('L2 Distance (Moments)', fontsize=12)
        ax1.set_title('Moment Distance vs Rho (EXPONENTIAL, Time-Invariant Features, Warm-Start)', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Theta L2 error vs rho
        ax2.plot(rhos, theta_errors, 's-', linewidth=1.5, markersize=4, color='orange')
        ax2.axvline(true_rho, color='r', linestyle='--', linewidth=2, label=f'True rho = {true_rho}')
        ax2.set_xlabel('Rho', fontsize=12)
        ax2.set_ylabel('L2 Error (Theta)', fontsize=12)
        ax2.set_title('Theta Estimation Error vs Rho', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('experiments/Gentzkow/exponential_quadratic_method/rho_search_moments_warmstart_time_invariant_features.png', dpi=150, bbox_inches='tight')
        print("Plot saved to: experiments/Gentzkow/exponential_quadratic_method/rho_search_moments_warmstart_time_invariant_features.png")
        print()
        
        # Create single plot focusing on moment distance
        fig2, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(rhos, moment_distances, 'o-', linewidth=2, markersize=6, color='blue')
        ax.axvline(true_rho, color='r', linestyle='--', linewidth=2, label=f'True rho = {true_rho}')
        # Highlight minimum
        min_idx = np.argmin(moment_distances)
        ax.plot(rhos[min_idx], moment_distances[min_idx], 'g*', markersize=20, label=f'Minimum at ρ={rhos[min_idx]:.2f}')
        ax.set_xlabel('Rho', fontsize=14)
        ax.set_ylabel('L2 Distance (Moments)', fontsize=14)
        ax.set_title('Moment Distance vs Rho (EXPONENTIAL, Time-Invariant Features, Warm-Start)', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        ax.set_xlim(min(rhos) - 0.05, max(rhos) + 0.05)
        plt.tight_layout()
        plt.savefig('experiments/Gentzkow/exponential_quadratic_method/rho_search_moment_distance_warmstart_time_invariant_features.png', dpi=150, bbox_inches='tight')
        print("Single plot saved to: experiments/Gentzkow/exponential_quadratic_method/rho_search_moment_distance_warmstart_time_invariant_features.png")
        print()


if __name__ == "__main__":
    main()


















