#!/usr/bin/env python3
"""
Estimate parameters with correctly specified model using flexible alpha parameter.

The script generates data with a true alpha, then allows estimation with errors
generated using a specified alpha (starting with the true alpha).
"""
import numpy as np
from mpi4py import MPI
from scipy.linalg import expm

from bundlechoice.core import BundleChoice
from bundlechoice.factory import ScenarioLibrary
from bundlechoice.factory.data_generator import QuadraticGenerationMethod, DataGenerator


def generate_errors_with_alpha(
    generator: DataGenerator,
    num_agents: int,
    num_items: int,
    num_items_per_period: int,
    num_periods: int,
    sigma: float,
    sigma_time_invariant: float,
    alpha: float,
    item_quadratic: np.ndarray,
    num_simuls: int = 1,
) -> np.ndarray:
    """
    Generate errors with time-invariant component having correlation structure based on alpha.
    
    Args:
        generator: Random number generator
        num_agents: Number of agents
        num_items: Total number of items (num_periods * num_items_per_period)
        num_items_per_period: Number of items per period
        num_periods: Number of periods
        sigma: Standard deviation for i.i.d. errors
        sigma_time_invariant: Standard deviation for time-invariant errors
        alpha: Correlation parameter (alpha > 0)
        item_quadratic: (num_items, num_items, num_features) quadratic features
        num_simuls: Number of simulation draws
    
    Returns:
        errors: (num_simuls, num_agents, num_items) array of errors
    """
    errors = np.zeros((num_simuls, num_agents, num_items))
    
    # Generate i.i.d. errors for each simulation
    for simul in range(num_simuls):
        iid_errors = generator.rng.normal(0, sigma, (num_agents, num_items))
        
        # Generate time-invariant errors with correlation structure
        if sigma_time_invariant > 0:
            time_invariant_errors = _generate_time_invariant_errors_with_alpha(
                generator,
                num_agents,
                num_items,
                num_items_per_period,
                num_periods,
                sigma_time_invariant,
                alpha,
                item_quadratic,
            )
            errors[simul] = iid_errors + time_invariant_errors
        else:
            errors[simul] = iid_errors
    
    return errors


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
    """
    Generate time-invariant errors with correlation structure from graph Laplacian of Q.
    
    Uses the same method as GentzkowScenarioBuilder._generate_time_invariant_errors_with_q_structure.
    """
    errors = np.zeros((num_agents, num_items))
    
    # Time-invariant errors are the SAME across periods.
    # Generate errors once for period 0, then copy to all periods.
    period = 0
    start_idx = period * num_items_per_period
    end_idx = (period + 1) * num_items_per_period
    period_items = num_items_per_period
    
    # Extract Q block for period 0 (same for all periods since Q is time-invariant)
    # Sum Q across all features to get connection strength matrix
    Q_period = item_quadratic[start_idx:end_idx, start_idx:end_idx, :]
    Q_sum = Q_period.sum(axis=2)  # Sum across features: (period_items, period_items)
    
    # Make Q_sum symmetric (since it's upper triangular, reflect it)
    Q_symmetric = Q_sum + Q_sum.T - np.diag(np.diag(Q_sum))
    
    # Step 1: Compute degree matrix D_ii = sum_j Q_ij
    D = np.diag(Q_symmetric.sum(axis=1))
    
    # Step 2: Form graph Laplacian L = D - Q
    L = D - Q_symmetric
    
    # Step 3: Compute kernel K = exp(-alpha * L)
    K = expm(-alpha * L)
    
    # Step 4: Normalize to correlation matrix R_ij = K_ij / sqrt(K_ii * K_jj)
    K_diag_sqrt = np.sqrt(np.diag(K))
    corr_matrix = K / np.outer(K_diag_sqrt, K_diag_sqrt)
    
    # Ensure positive semidefinite (should be by construction, but add small regularization if needed)
    corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Ensure symmetry
    eigenvals = np.linalg.eigvals(corr_matrix)
    if np.any(eigenvals < -1e-10):
        # Add small regularization to ensure positive semidefinite
        corr_matrix += np.eye(period_items) * (1e-10 - np.min(eigenvals))
    
    # Generate correlated errors ONCE for period 0
    # For each agent, generate period_items correlated errors
    time_invariant_errors = np.zeros((num_agents, period_items))
    for agent in range(num_agents):
        try:
            period_errors = generator.rng.multivariate_normal(
                np.zeros(period_items),
                sigma**2 * corr_matrix,
            )
        except np.linalg.LinAlgError:
            # If correlation matrix is not positive definite, use independent errors
            period_errors = generator.rng.normal(0, sigma, period_items)
        time_invariant_errors[agent, :] = period_errors
    
    # Copy the SAME time-invariant errors to all periods
    for period in range(num_periods):
        start_idx = period * num_items_per_period
        end_idx = (period + 1) * num_items_per_period
        errors[:, start_idx:end_idx] = time_invariant_errors
    
    return errors


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("=" * 80)
        print("Estimation with Correctly Specified Model (Flexible Alpha)")
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
    
    comm.Barrier()  # Ensure all ranks are synchronized
    
    # Generate errors with the true alpha (correctly specified)
    estimation_alpha = true_alpha  # Start with true alpha
    
    # Extract metadata and quadratic features for error generation
    if rank == 0:
        num_simuls = prepared.metadata['num_simuls']
        num_agents = prepared.metadata['num_agents']
        num_items = prepared.metadata['num_items']
        num_items_per_period = prepared.metadata['num_items_per_period']
        num_periods = prepared.metadata['num_periods']
        sigma = 3.0
        sigma_time_invariant = 2.0
        
        # Get item quadratic features (needed for correlation structure)
        item_quadratic = prepared.estimation_data['item_data']['quadratic']
        observed_bundles = prepared.estimation_data['obs_bundle'].copy()
        
        generator = DataGenerator(seed=999)
        estimation_errors = generate_errors_with_alpha(
            generator=generator,
            num_agents=num_agents,
            num_items=num_items,
            num_items_per_period=num_items_per_period,
            num_periods=num_periods,
            sigma=sigma,
            sigma_time_invariant=sigma_time_invariant,
            alpha=estimation_alpha,
            item_quadratic=item_quadratic,
            num_simuls=num_simuls,
        )
        
        # Prepare estimation data
        estimation_data = prepared.estimation_data.copy()
        estimation_data["errors"] = estimation_errors
        estimation_data["obs_bundle"] = observed_bundles
    else:
        estimation_data = None
    
    comm.Barrier()  # Ensure data preparation is complete
    
    # ============================================================================
    # ESTIMATION: CORRECTLY SPECIFIED (with correlation)
    # ============================================================================
    if rank == 0:
        print("=" * 80)
        print(f"ESTIMATION: CORRECTLY SPECIFIED (alpha = {estimation_alpha})")
        print("=" * 80)
        print()
    
    bundlechoice = BundleChoice()
    prepared.apply(bundlechoice, comm=comm, stage="estimation")
    
    # Replace with errors generated using estimation_alpha
    if rank == 0:
        bundlechoice.data_manager.load_and_scatter(estimation_data)
    else:
        bundlechoice.data_manager.scatter()
    
    # Reinitialize subproblems and row generation with new data
    bundlechoice.subproblems.initialize_local()
    bundlechoice.row_generation._initialize_master_problem(initial_constraints=None)
    
    result = bundlechoice.row_generation.solve(callback=None)
    theta_hat = result.theta_hat
    
    if rank == 0:
        print(f"\nEstimated theta: {theta_hat}")
        print(f"True theta: {custom_theta}")
        l2_error = np.linalg.norm(theta_hat - custom_theta)
        print(f"L2 error: {l2_error:.6f}")
        
        # Print row generation stats
        if result.timing is not None:
            stats = result.timing
            print(f"\nRow Generation Stats:")
            print(f"  Total time: {stats.get('total_time', 0):.2f}s")
            print(f"  Init time: {stats.get('init_time', 0):.2f}s")
            print(f"  Iterations: {result.num_iterations}")
            print(f"  Pricing time: {stats.get('pricing_time', 0):.2f}s ({stats.get('pricing_time_pct', 0):.1f}%)")
            print(f"  Master time: {stats.get('master_time', 0):.2f}s ({stats.get('master_time_pct', 0):.1f}%)")
            print(f"  MPI time: {stats.get('mpi_time', 0):.2f}s ({stats.get('mpi_time_pct', 0):.1f}%)")
        print()
    
    comm.Barrier()
    
    # Store observed bundles for comparison
    if rank == 0:
        observed_bundles = prepared.estimation_data['obs_bundle'].copy()
    else:
        observed_bundles = None
    
    comm.Barrier()
    
    # ============================================================================
    # GENERATE BUNDLES AT ESTIMATED THETA
    # ============================================================================
    if rank == 0:
        print("=" * 80)
        print("GENERATING BUNDLES AT ESTIMATED THETA")
        print("=" * 80)
        print()
    
    # Generate bundles using estimated theta and estimation errors
    # The errors are already in the data manager from estimation
    # We need to solve subproblems with theta_hat
    if rank == 0:
        theta_hat_broadcast = theta_hat.copy()
    else:
        theta_hat_broadcast = np.empty(len(theta_hat), dtype=np.float64)
    comm.Bcast(theta_hat_broadcast, root=0)
    
    # Solve subproblems to get optimal bundles at estimated theta
    generated_bundles = bundlechoice.subproblems.init_and_solve(theta_hat_broadcast, return_values=False)
    
    if rank == 0:
        if generated_bundles is not None:
            print(f"Generated bundles shape: {generated_bundles.shape}")
            print(f"Number of agents: {generated_bundles.shape[0]}")
            print(f"Number of items: {generated_bundles.shape[1]}")
            print()
        else:
            print("WARNING: No bundles generated!")
            print()
    
    # ============================================================================
    # COMPUTE STATISTICS ON GENERATED BUNDLES
    # ============================================================================
    if rank == 0:
        print("=" * 80)
        print("COMPUTING STATISTICS ON GENERATED BUNDLES")
        print("=" * 80)
        print()
        
        # Set up statistics computation
        stats = compute_bundle_statistics(
            generated_bundles,
            num_items_per_period=prepared.metadata['num_items_per_period'],
            num_periods=prepared.metadata['num_periods'],
        )
        
        print_statistics(stats)
        print()
        
        # ============================================================================
        # COMPUTE CHOICE PATTERN MOMENTS
        # ============================================================================
        print("=" * 80)
        print("CHOICE PATTERN MOMENTS")
        print("=" * 80)
        print()
        
        # Compute moments for observed bundles
        print("OBSERVED BUNDLES (from data generation):")
        print("-" * 80)
        obs_pattern_moments = compute_choice_pattern_moments(
            observed_bundles,
            num_items_per_period=prepared.metadata['num_items_per_period'],
            num_periods=prepared.metadata['num_periods'],
        )
        print_choice_pattern_moments(obs_pattern_moments)
        print()
        
        # Compute moments for generated bundles at estimated theta
        print("GENERATED BUNDLES (at estimated theta):")
        print("-" * 80)
        gen_pattern_moments = compute_choice_pattern_moments(
            generated_bundles,
            num_items_per_period=prepared.metadata['num_items_per_period'],
            num_periods=prepared.metadata['num_periods'],
        )
        print_choice_pattern_moments(gen_pattern_moments)
        print()
        
        # Compare moments
        print("COMPARISON:")
        print("-" * 80)
        compare_pattern_moments(obs_pattern_moments, gen_pattern_moments)
        print()


def compute_bundle_statistics(
    bundles: np.ndarray,
    num_items_per_period: int,
    num_periods: int,
) -> dict:
    """
    Compute statistics on generated bundles.
    
    Args:
        bundles: (num_agents, num_items) array of binary bundles
        num_items_per_period: Number of items per period
        num_periods: Number of periods
    
    Returns:
        Dictionary of statistics
    """
    num_agents, num_items = bundles.shape
    
    # Basic statistics
    bundle_sizes = bundles.sum(axis=1)
    item_demands = bundles.sum(axis=0)
    
    stats = {
        'num_agents': num_agents,
        'num_items': num_items,
        'num_items_per_period': num_items_per_period,
        'num_periods': num_periods,
        'bundle_sizes': {
            'min': float(bundle_sizes.min()),
            'max': float(bundle_sizes.max()),
            'mean': float(bundle_sizes.mean()),
            'std': float(bundle_sizes.std()),
        },
        'item_demands': {
            'min': float(item_demands.min()),
            'max': float(item_demands.max()),
            'mean': float(item_demands.mean()),
            'std': float(item_demands.std()),
        },
        'total_items_selected': int(bundles.sum()),
        'total_possible': num_agents * num_items,
        'selection_rate': float(bundles.sum() / (num_agents * num_items)),
    }
    
    # Period-specific statistics (if multiple periods)
    if num_periods > 1:
        period_stats = []
        for period in range(num_periods):
            start_idx = period * num_items_per_period
            end_idx = (period + 1) * num_items_per_period
            period_bundles = bundles[:, start_idx:end_idx]
            period_sizes = period_bundles.sum(axis=1)
            period_demands = period_bundles.sum(axis=0)
            
            period_stats.append({
                'period': period,
                'bundle_sizes': {
                    'min': float(period_sizes.min()),
                    'max': float(period_sizes.max()),
                    'mean': float(period_sizes.mean()),
                    'std': float(period_sizes.std()),
                },
                'item_demands': {
                    'min': float(period_demands.min()),
                    'max': float(period_demands.max()),
                    'mean': float(period_demands.mean()),
                    'std': float(period_demands.std()),
                },
            })
        stats['period_stats'] = period_stats
    
    return stats


def print_statistics(stats: dict) -> None:
    """Print bundle statistics in a readable format."""
    print(f"Overall Statistics:")
    print(f"  Agents: {stats['num_agents']}")
    print(f"  Items: {stats['num_items']} ({stats['num_periods']} periods × {stats['num_items_per_period']} items/period)")
    print(f"  Total items selected: {stats['total_items_selected']} out of {stats['total_possible']}")
    print(f"  Selection rate: {stats['selection_rate']:.2%}")
    print()
    
    print(f"Bundle Sizes:")
    print(f"  Min: {stats['bundle_sizes']['min']:.1f}")
    print(f"  Max: {stats['bundle_sizes']['max']:.1f}")
    print(f"  Mean: {stats['bundle_sizes']['mean']:.2f}")
    print(f"  Std: {stats['bundle_sizes']['std']:.2f}")
    print()
    
    print(f"Item Demands:")
    print(f"  Min: {stats['item_demands']['min']:.1f}")
    print(f"  Max: {stats['item_demands']['max']:.1f}")
    print(f"  Mean: {stats['item_demands']['mean']:.2f}")
    print(f"  Std: {stats['item_demands']['std']:.2f}")
    print()
    
    if 'period_stats' in stats:
        print(f"Period-Specific Statistics:")
        for period_stat in stats['period_stats']:
            period = period_stat['period']
            print(f"  Period {period}:")
            print(f"    Bundle sizes: mean={period_stat['bundle_sizes']['mean']:.2f}, "
                  f"std={period_stat['bundle_sizes']['std']:.2f}")
            print(f"    Item demands: mean={period_stat['item_demands']['mean']:.2f}, "
                  f"std={period_stat['item_demands']['std']:.2f}")


def compute_choice_pattern_moments(
    bundles: np.ndarray,
    num_items_per_period: int,
    num_periods: int,
) -> dict:
    """
    Compute choice pattern moments for all pairs of items across 2 periods.
    
    For each pair of items (i, j), compute the fraction of agents with each of the
    16 possible choice patterns across 2 periods and 2 items.
    
    Pattern encoding: [item1_period1, item1_period2, item2_period1, item2_period2]
    Pattern index: 0-15 (binary encoding of the 4 bits)
    
    Args:
        bundles: (num_agents, num_items) array of binary bundles
        num_items_per_period: Number of items per period
        num_periods: Number of periods (should be 2)
    
    Returns:
        Dictionary with average moments over all item pairs
    """
    if num_periods != 2:
        raise ValueError("This function is designed for 2-period models")
    
    num_agents, num_items = bundles.shape
    
    # Extract period bundles
    period1_items = slice(0, num_items_per_period)
    period2_items = slice(num_items_per_period, 2 * num_items_per_period)
    period1_bundles = bundles[:, period1_items].astype(bool)  # (num_agents, num_items_per_period)
    period2_bundles = bundles[:, period2_items].astype(bool)  # (num_agents, num_items_per_period)
    
    # For each unordered pair of items (i, j) where i < j
    num_pairs = num_items_per_period * (num_items_per_period - 1) // 2
    pattern_counts = np.zeros((num_pairs, 16), dtype=np.int32)
    
    pair_idx = 0
    for i in range(num_items_per_period):
        for j in range(i + 1, num_items_per_period):
            # Extract choice pattern for items i and j across both periods
            # Pattern: [item_i_period1, item_i_period2, item_j_period1, item_j_period2]
            item_i_p1 = period1_bundles[:, i].astype(int)  # (num_agents,)
            item_i_p2 = period2_bundles[:, i].astype(int)  # (num_agents,)
            item_j_p1 = period1_bundles[:, j].astype(int)  # (num_agents,)
            item_j_p2 = period2_bundles[:, j].astype(int)  # (num_agents,)
            
            # Encode pattern as 4-bit integer: [item_i_p1, item_i_p2, item_j_p1, item_j_p2]
            # bit 0 (LSB): item_i_p1
            # bit 1: item_i_p2
            # bit 2: item_j_p1
            # bit 3 (MSB): item_j_p2
            patterns = (item_i_p1 + 
                       2 * item_i_p2 + 
                       4 * item_j_p1 + 
                       8 * item_j_p2).astype(int)  # (num_agents,)
            
            # Count occurrences of each pattern (0-15)
            for pattern in range(16):
                pattern_counts[pair_idx, pattern] = np.sum(patterns == pattern)
            
            pair_idx += 1
    
    # Compute fractions (moments) for each pair
    pattern_fractions = pattern_counts.astype(float) / num_agents  # (num_pairs, 16)
    
    # Average over all pairs
    avg_pattern_moments = pattern_fractions.mean(axis=0)  # (16,)
    
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
        'per_pair_fractions': pattern_fractions,  # (num_pairs, 16) - for detailed analysis
    }


def print_choice_pattern_moments(moments: dict) -> None:
    """Print choice pattern moments in a readable format."""
    print("Choice Pattern Moments (averaged over all item pairs):")
    print(f"  Number of item pairs: {moments['num_pairs']}")
    print(f"  Number of agents: {moments['num_agents']}")
    print()
    print("  Pattern | Fraction")
    print("  --------|---------")
    
    pattern_moments = moments['pattern_moments']
    pattern_labels = moments['pattern_labels']
    
    for pattern_idx in range(16):
        label = pattern_labels[pattern_idx]
        fraction = pattern_moments[pattern_idx]
        print(f"  {label}     | {fraction:.4f}")
    
    print()
    print(f"  Sum of all moments: {pattern_moments.sum():.4f} (should be 1.0)")


def compare_pattern_moments(obs_moments: dict, gen_moments: dict) -> None:
    """Compare observed and generated pattern moments."""
    obs_patterns = obs_moments['pattern_moments']
    gen_patterns = gen_moments['pattern_moments']
    pattern_labels = obs_moments['pattern_labels']
    
    print("  Pattern | Observed | Generated | Difference")
    print("  --------|----------|-----------|-----------")
    
    differences = []
    for pattern_idx in range(16):
        label = pattern_labels[pattern_idx]
        obs_val = obs_patterns[pattern_idx]
        gen_val = gen_patterns[pattern_idx]
        diff = gen_val - obs_val
        differences.append(abs(diff))
        print(f"  {label}     | {obs_val:.4f}   | {gen_val:.4f}    | {diff:+.4f}")
    
    print()
    print(f"  Mean absolute difference: {np.mean(differences):.4f}")
    print(f"  Max absolute difference: {np.max(differences):.4f}")
    print(f"  L2 distance: {np.linalg.norm(gen_patterns - obs_patterns):.4f}")


if __name__ == "__main__":
    main()

