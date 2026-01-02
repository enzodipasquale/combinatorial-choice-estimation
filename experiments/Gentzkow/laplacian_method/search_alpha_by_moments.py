#!/usr/bin/env python3
"""
Search over alpha to minimize L2 distance between observed and generated moments.

For each alpha:
1. Transform errors with correlation based on alpha
2. Estimate theta using row generation
3. Generate bundles at estimated theta
4. Compute choice pattern moments
5. Compare to observed moments (L2 distance)
"""
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

from bundlechoice.core import BundleChoice
from bundlechoice.scenarios import ScenarioLibrary
from bundlechoice.scenarios.data_generator import QuadraticGenerationMethod, DataGenerator
from scipy.linalg import expm


def _generate_time_invariant_errors_with_alpha(
    generator: DataGenerator,
    num_agents: int,
    num_items: int,
    num_items_per_period: int,
    num_periods: int,
    sigma: float,
    alpha: float,
    item_quadratic: np.ndarray,
) -> np.ndarray:
    """Generate time-invariant errors with correlation structure from graph Laplacian of Q."""
    errors = np.zeros((num_agents, num_items))
    
    # Time-invariant errors are the SAME across periods.
    period = 0
    start_idx = period * num_items_per_period
    end_idx = (period + 1) * num_items_per_period
    period_items = num_items_per_period
    
    # Extract Q block for period 0
    Q_period = item_quadratic[start_idx:end_idx, start_idx:end_idx, :]
    Q_sum = Q_period.sum(axis=2)
    
    # Make Q_sum symmetric
    Q_symmetric = Q_sum + Q_sum.T - np.diag(np.diag(Q_sum))
    
    # Compute degree matrix D_ii = sum_j Q_ij
    D = np.diag(Q_symmetric.sum(axis=1))
    
    # Form graph Laplacian L = D - Q
    L = D - Q_symmetric
    
    # Compute kernel K = exp(-alpha * L)
    K = expm(-alpha * L)
    
    # Normalize to correlation matrix
    K_diag_sqrt = np.sqrt(np.diag(K))
    corr_matrix = K / np.outer(K_diag_sqrt, K_diag_sqrt)
    
    # Ensure positive semidefinite
    corr_matrix = (corr_matrix + corr_matrix.T) / 2
    eigenvals = np.linalg.eigvals(corr_matrix)
    if np.any(eigenvals < -1e-10):
        corr_matrix += np.eye(period_items) * (1e-10 - np.min(eigenvals))
    
    # Generate correlated errors ONCE for period 0
    time_invariant_errors = np.zeros((num_agents, period_items))
    for agent in range(num_agents):
        try:
            period_errors = generator.rng.multivariate_normal(
                np.zeros(period_items),
                sigma**2 * corr_matrix,
            )
        except np.linalg.LinAlgError:
            period_errors = generator.rng.normal(0, sigma, period_items)
        time_invariant_errors[agent, :] = period_errors
    
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
        print("Search Alpha by Minimizing Moment Distance")
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
        
        # Generate i.i.d. errors ONCE - these will be transformed for each alpha
        generator = DataGenerator(seed=999)
        
        # Generate i.i.d. simulation errors
        iid_simulation_errors = np.zeros((num_simuls, num_agents, num_items))
        for simul in range(num_simuls):
            iid_simulation_errors[simul] = generator.rng.normal(0, sigma, (num_agents, num_items))
        
        # Generate i.i.d. time-invariant errors (will be transformed with correlation)
        iid_time_invariant_errors = generator.rng.normal(
            0, sigma_time_invariant, (num_agents, num_items_per_period)
        )
        
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
        iid_time_invariant_errors = None
        obs_pattern_moments = None
        num_simuls = None
        num_agents = None
        num_items = None
        num_items_per_period = None
        num_periods = None
        sigma = None
        sigma_time_invariant = None
    
    comm.Barrier()
    
    # Search over alpha values
    alpha_values = [0.9, 1.0, 1.1]
    
    if rank == 0:
        print("=" * 80)
        print("SEARCHING OVER ALPHA VALUES")
        print("=" * 80)
        print(f"Alpha values to test: {alpha_values}")
        print()
    
    results = []
    
    for alpha in alpha_values:
        if rank == 0:
            print("=" * 80)
            print(f"ALPHA = {alpha}")
            print("=" * 80)
            print()
        
        # Transform errors for this alpha
        if rank == 0:
            # Generate time-invariant errors with correlation based on alpha
            generator_alpha = DataGenerator(seed=999)
            time_invariant_errors = _generate_time_invariant_errors_with_alpha(
                generator_alpha,
                num_agents,
                num_items,
                num_items_per_period,
                num_periods,
                sigma_time_invariant,
                alpha,
                item_quadratic,
            )
            
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
        
        # Estimate theta with this alpha
        bundlechoice = BundleChoice()
        prepared.apply(bundlechoice, comm=comm, stage="estimation")
        
        # Replace with errors generated using this alpha
        if rank == 0:
            bundlechoice.data_manager.load_and_scatter(estimation_data)
        else:
            bundlechoice.data_manager.scatter()
        
        # Reinitialize subproblems and row generation
        bundlechoice.subproblems.initialize_local()
        bundlechoice.row_generation._initialize_master_problem(initial_constraints=None)
        
        result = bundlechoice.row_generation.solve(callback=None)
        theta_hat = result.theta_hat
        
        if rank == 0:
            l2_error_theta = np.linalg.norm(theta_hat - custom_theta)
            print(f"Estimated theta: {theta_hat}")
            print(f"True theta: {custom_theta}")
            print(f"L2 error (theta): {l2_error_theta:.6f}")
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
                'alpha': alpha,
                'theta_hat': theta_hat,
                'l2_error_theta': l2_error_theta,
                'moment_l2_distance': moment_l2_distance,
                'gen_moments': gen_pattern_moments,
            })
        
        comm.Barrier()
    
    # Plot results
    if rank == 0:
        print("=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        print()
        print("Alpha | L2 Error (theta) | L2 Distance (moments)")
        print("------|-------------------|----------------------")
        for r in results:
            print(f" {r['alpha']:.1f}  | {r['l2_error_theta']:.6f}        | {r['moment_l2_distance']:.6f}")
        print()
        
        # Plot
        alphas = [r['alpha'] for r in results]
        moment_distances = [r['moment_l2_distance'] for r in results]
        theta_errors = [r['l2_error_theta'] for r in results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Moment L2 distance vs alpha
        ax1.plot(alphas, moment_distances, 'o-', linewidth=2, markersize=8)
        ax1.axvline(true_alpha, color='r', linestyle='--', label=f'True alpha = {true_alpha}')
        ax1.set_xlabel('Alpha', fontsize=12)
        ax1.set_ylabel('L2 Distance (Moments)', fontsize=12)
        ax1.set_title('Moment Distance vs Alpha', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Theta L2 error vs alpha
        ax2.plot(alphas, theta_errors, 's-', linewidth=2, markersize=8, color='orange')
        ax2.axvline(true_alpha, color='r', linestyle='--', label=f'True alpha = {true_alpha}')
        ax2.set_xlabel('Alpha', fontsize=12)
        ax2.set_ylabel('L2 Error (Theta)', fontsize=12)
        ax2.set_title('Theta Estimation Error vs Alpha', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('experiments/Gentzkow/alpha_search_moments.png', dpi=150, bbox_inches='tight')
        print("Plot saved to: experiments/Gentzkow/alpha_search_moments.png")
        print()
        
        # Create single plot focusing on moment distance
        fig2, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(alphas, moment_distances, 'o-', linewidth=2, markersize=6, color='blue')
        ax.axvline(true_alpha, color='r', linestyle='--', linewidth=2, label=f'True alpha = {true_alpha}')
        # Highlight minimum
        min_idx = np.argmin(moment_distances)
        ax.plot(alphas[min_idx], moment_distances[min_idx], 'g*', markersize=20, label=f'Minimum at α={alphas[min_idx]:.2f}')
        ax.set_xlabel('Alpha', fontsize=14)
        ax.set_ylabel('L2 Distance (Moments)', fontsize=14)
        ax.set_title('Moment Distance vs Alpha', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        ax.set_xlim(min(alphas) - 0.05, max(alphas) + 0.05)
        plt.tight_layout()
        plt.savefig('experiments/Gentzkow/alpha_search_moment_distance.png', dpi=150, bbox_inches='tight')
        print("Single plot saved to: experiments/Gentzkow/alpha_search_moment_distance.png")
        print()


if __name__ == "__main__":
    main()

