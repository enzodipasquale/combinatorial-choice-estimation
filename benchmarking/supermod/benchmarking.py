#!/usr/bin/env python3
"""
Comparison script for estimation results between two datasets.
Compares the first 100 lines of each dataset and provides statistical analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load the first 100 lines from both datasets and prepare them for comparison."""
    
    # File paths
    current_results = Path("benchmarking/supermod/results.csv")
    score_estimator_results = Path("/Users/enzo-macbookpro/MyProjects/score-estimator/supermod/AddDrop/estimation_results.csv")
    
    print("Loading datasets...")
    print(f"Current results: {current_results}")
    print(f"Score estimator results: {score_estimator_results}")
    
    # Load first 100 lines from each dataset
    try:
        df_current = pd.read_csv(current_results, nrows=100)
        print(f"✓ Loaded {len(df_current)} rows from current results")
    except Exception as e:
        print(f"✗ Error loading current results: {e}")
        return None, None
    
    try:
        df_score = pd.read_csv(score_estimator_results, nrows=100)
        print(f"✓ Loaded {len(df_score)} rows from score estimator results")
    except Exception as e:
        print(f"✗ Error loading score estimator results: {e}")
        return None, None
    
    return df_current, df_score

def extract_time_comparison(df_current, df_score):
    """Extract and compare execution times between methods."""
    # Current method uses 'elapsed' column
    current_times = df_current['elapsed'].values
    
    # Score estimator uses 'timeTaken' column
    score_times = df_score['timeTaken'].values
    
    # Calculate statistics
    time_stats = {
        'current_mean': current_times.mean(),
        'current_std': current_times.std(),
        'score_mean': score_times.mean(),
        'score_std': score_times.std(),
        'time_ratio': score_times.mean() / current_times.mean()  # score/current
    }
    
    return time_stats

def extract_normalized_betas(df_current, df_score):
    """Extract and normalize beta_hat_* columns from current results, and extract k-1 columns from score estimator."""
    # Find all beta_hat_* columns and sort
    beta_hat_cols = [col for col in df_current.columns if col.startswith('beta_hat_')]
    beta_hat_cols.sort()
    k = len(beta_hat_cols)
    if k < 2:
        raise ValueError("Need at least two beta_hat columns for normalization.")
    
    # Normalize: divide all beta_hat_* by the first, then drop the first
    betas = df_current[beta_hat_cols].values
    normalized = betas / betas[:, [0]]  # divide each row by its first beta_hat
    normalized = normalized[:, 1:]      # exclude the first column
    norm_cols = [f'beta_hat_{i}' for i in range(1, k)]
    norm_betas_df = pd.DataFrame(normalized, columns=norm_cols)
    
    # Score estimator columns (assume b2, b3, ...)
    score_beta_cols = [col for col in df_score.columns if col.startswith('b') and col[1:].isdigit()]
    score_beta_cols.sort(key=lambda x: int(x[1:]))
    score_betas = df_score[score_beta_cols].iloc[:, :k-1].copy()  # ensure same number of columns
    score_betas.columns = norm_cols
    
    return norm_betas_df, score_betas

def extract_true_normalized_values(df_current):
    """Extract the true normalized theta values for MSE computation."""
    # Find all theta_0_* columns and sort
    theta_0_cols = [col for col in df_current.columns if col.startswith('theta_0_')]
    theta_0_cols.sort()
    k = len(theta_0_cols)
    if k < 2:
        raise ValueError("Need at least two theta_0 columns for normalization.")
    
    # Get the true theta values from the first row (they should be the same across all rows)
    true_thetas = df_current[theta_0_cols].iloc[0].values
    
    # Normalize: divide all theta_0_* by the first, then drop the first
    normalized_true = true_thetas / true_thetas[0]  # divide by first theta
    normalized_true = normalized_true[1:]           # exclude the first column
    
    # Create mapping from normalized beta column names to their true values
    true_values = {}
    for i, col in enumerate([f'beta_hat_{j}' for j in range(1, k)]):
        true_values[col] = normalized_true[i]
    
    return true_values

def boxplot_normalized(norm_betas_df, score_betas, results_folder):
    """Boxplot for normalized k-1 betas from both methods."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from datetime import datetime
    
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create side-by-side boxplots for each beta estimate
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get the number of beta estimates
    n_betas = len(norm_betas_df.columns)
    x_positions = np.arange(n_betas)
    width = 0.35
    
    # Create boxplot data
    current_data = [norm_betas_df[col] for col in norm_betas_df.columns]
    score_data = [score_betas[col] for col in score_betas.columns]
    
    # Create proper beta labels (beta_2, ..., beta_k)
    beta_labels = [f'$\\beta_{i+2}$' for i in range(n_betas)]
    
    # Create boxplots side by side without labels
    bp1 = ax.boxplot(current_data, positions=x_positions - width/2, patch_artist=True, 
                     widths=width)
    bp2 = ax.boxplot(score_data, positions=x_positions + width/2, patch_artist=True, 
                     widths=width)
    
    # Set the x-axis labels to be centered between the pairs
    ax.set_xticks(x_positions)
    ax.set_xticklabels(beta_labels)
    
    # Color the boxes
    for patch in bp1['boxes']:
        patch.set_facecolor('plum')
        patch.set_alpha(0.7)
    for patch in bp2['boxes']:
        patch.set_facecolor('lightcoral')
        patch.set_alpha(0.7)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='plum', alpha=0.7, label='Current (normalized)'),
                      Patch(facecolor='lightcoral', alpha=0.7, label='Score estimator')]
    ax.legend(handles=legend_elements)
    
    ax.set_ylabel('Normalized Beta Estimate')
    ax.set_title('Boxplot of Normalized Beta Estimates (k-1) - Supermodular')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot (no need for timestamp in filename since we're in a timestamped folder)
    plot_file = results_folder / "boxplot_normalized.png"
    plt.savefig(plot_file, dpi=300)
    print(f"✓ Boxplot saved to: {plot_file}")
    plt.show()

def mean_squared_error(estimates, true_value=1.0):
    """Compute mean squared error to the true value (default 1.0)."""
    return ((estimates - true_value) ** 2).mean()

def relative_mean_squared_error(estimates, true_value=1.0):
    """Compute relative mean squared error to the true value (default 1.0)."""
    return mean_squared_error(estimates, true_value) / true_value**2

def relative_bias(estimates, true_value=1.0):
    """Compute relative bias to the true value (default 1.0)."""
    return (estimates.mean() - true_value) / true_value

def compute_individual_mse(norm_betas_df, score_betas, true_values=None):
    """Compute MSE, RMSE, absolute bias and relative bias for each individual beta estimate."""
    mse_results = []
    
    # If no true values provided, use the mean of each column as the target
    # This is more reasonable than assuming 1.0 for all normalized betas
    if true_values is None:
        true_values = {}
        for col in norm_betas_df.columns:
            # Use the mean of the current estimates as a reasonable target
            # This assumes the current method is unbiased
            true_values[col] = norm_betas_df[col].mean()
    
    for col in norm_betas_df.columns:
        true_value = true_values.get(col, 1.0)  # fallback to 1.0 if not specified
        
        current_mse = mean_squared_error(norm_betas_df[col], true_value)
        score_mse = mean_squared_error(score_betas[col], true_value)
        
        # Calculate RMSE
        current_rmse = relative_mean_squared_error(norm_betas_df[col], true_value)
        score_rmse = relative_mean_squared_error(score_betas[col], true_value)
        
        # Calculate absolute bias (average difference from true value)
        current_abs_bias = (norm_betas_df[col] - true_value).mean()
        score_abs_bias = (score_betas[col] - true_value).mean()
        
        # Calculate relative bias
        current_rel_bias = relative_bias(norm_betas_df[col], true_value)
        score_rel_bias = relative_bias(score_betas[col], true_value)
        
        mse_results.append({
            'Beta_Index': col,
            'True_Value': true_value,
            'Current_MSE': current_mse,
            'Score_MSE': score_mse,
            'Current_RMSE': current_rmse,
            'Score_RMSE': score_rmse,
            'Current_Abs_Bias': current_abs_bias,
            'Score_Abs_Bias': score_abs_bias,
            'Current_Rel_Bias': current_rel_bias,
            'Score_Rel_Bias': score_rel_bias
        })
    
    return pd.DataFrame(mse_results)

def create_results_folder():
    """Create timestamped results subfolder for this run."""
    from datetime import datetime
    
    # Create base comparison_results folder
    base_folder = Path("benchmarking/supermod/comparison_results")
    base_folder.mkdir(exist_ok=True)
    
    # Create timestamped subfolder for this run
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_folder = base_folder / timestamp
    run_folder.mkdir(exist_ok=True)
    
    return run_folder

def extract_metadata(df_current):
    """Extract metadata from the first row of current results."""
    if len(df_current) == 0:
        return {}
    
    first_row = df_current.iloc[0]
    metadata = {
        'time': first_row.get('time', ''),
        'num_agents': first_row.get('num_agents', ''),
        'num_items': first_row.get('num_items', ''),
        'num_features': first_row.get('num_features', ''),
        'num_simuls': first_row.get('num_simuls', ''),
        'subproblem': first_row.get('subproblem', ''),
        'sigma': first_row.get('sigma', '')
    }
    return metadata

def save_results_with_metadata(results_folder, individual_mse_df, metadata, mse_current, mse_score):
    """Save results with metadata and append to existing files."""
    from datetime import datetime
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Prepare summary results with metadata
    summary_row = {
        'timestamp': timestamp,
        'time': metadata.get('time', ''),
        'num_agents': metadata.get('num_agents', ''),
        'num_items': metadata.get('num_items', ''),
        'num_features': metadata.get('num_features', ''),
        'num_simuls': metadata.get('num_simuls', ''),
        'subproblem': metadata.get('subproblem', ''),
        'sigma': metadata.get('sigma', ''),
        'current_mse': mse_current,
        'score_mse': mse_score,
        'mse_difference': mse_current - mse_score
    }
    
    # Save individual MSEs with metadata
    individual_results = individual_mse_df.copy()
    for key, value in metadata.items():
        individual_results[key] = value
    individual_results['timestamp'] = timestamp
    
    # Save files in the timestamped run folder (no appending needed)
    summary_file = results_folder / "summary_mse.csv"
    individual_file = results_folder / "individual_mse.csv"
    
    # Save results directly (overwrite for this run)
    summary_df = pd.DataFrame([summary_row])
    summary_df.to_csv(summary_file, index=False)
    
    individual_results.to_csv(individual_file, index=False)
    
    return summary_file, individual_file

def save_mse_results(results_folder, individual_mse_df, metadata, time_stats):
    """Save MSE results with metadata and time comparison in a single file."""
    from datetime import datetime
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Add metadata and time stats to the MSE dataframe
    mse_results = individual_mse_df.copy()
    for key, value in metadata.items():
        mse_results[key] = value
    
    # Add time comparison columns
    mse_results['current_time_mean'] = time_stats['current_mean']
    mse_results['current_time_std'] = time_stats['current_std']
    mse_results['score_time_mean'] = time_stats['score_mean']
    mse_results['score_time_std'] = time_stats['score_std']
    mse_results['time_ratio'] = time_stats['time_ratio']
    mse_results['timestamp'] = timestamp
    
    # Save to mse.csv
    mse_file = results_folder / "mse.csv"
    mse_results.to_csv(mse_file, index=False)

def main():
    print("="*60)
    print("NORMALIZED ESTIMATION RESULTS COMPARISON - SUPERMODULAR")
    print("="*60)
    print("Comparing normalized (k-1) beta estimates of each dataset")
    print("="*60)
    
    # Create results folder
    results_folder = create_results_folder()
    print(f"Results will be saved to: {results_folder}")
    
    # Load data
    df_current, df_score = load_and_prepare_data()
    if df_current is None or df_score is None:
        print("Failed to load data. Exiting.")
        return
    
    # Extract metadata
    metadata = extract_metadata(df_current)
    print(f"Metadata extracted: {metadata}")
    
    # Extract normalized betas
    norm_betas_df, score_betas = extract_normalized_betas(df_current, df_score)
    
    # Extract true normalized values for MSE computation
    true_normalized_values = extract_true_normalized_values(df_current)
    
    # Extract time comparison
    time_stats = extract_time_comparison(df_current, df_score)
    
    # Boxplot
    boxplot_normalized(norm_betas_df, score_betas, results_folder)
    
    # MSE calculation for each beta estimate using true values
    individual_mse_df = compute_individual_mse(norm_betas_df, score_betas, true_normalized_values)
    print("\nIndividual Mean Squared Errors:")
    print("=" * 50)
    print(individual_mse_df.round(6))
    
    # Overall MSE calculation
    mse_current = individual_mse_df['Current_MSE'].mean()
    mse_score = individual_mse_df['Score_MSE'].mean()
    print(f"\nOverall Mean Squared Error (Current, normalized): {mse_current:.6f}")
    print(f"Overall Mean Squared Error (Score estimator): {mse_score:.6f}")
    
    # Overall RMSE calculation
    rmse_current = individual_mse_df['Current_RMSE'].mean()
    rmse_score = individual_mse_df['Score_RMSE'].mean()
    print(f"\nOverall Relative Mean Squared Error (Current, normalized): {rmse_current:.6f}")
    print(f"Overall Relative Mean Squared Error (Score estimator): {rmse_score:.6f}")
    
    # Overall bias calculation
    bias_current = individual_mse_df['Current_Abs_Bias'].mean()
    bias_score = individual_mse_df['Score_Abs_Bias'].mean()
    print(f"\nOverall Average Absolute Bias (Current, normalized): {bias_current:.6f}")
    print(f"Overall Average Absolute Bias (Score estimator): {bias_score:.6f}")
    
    # Overall relative bias calculation
    rel_bias_current = individual_mse_df['Current_Rel_Bias'].mean()
    rel_bias_score = individual_mse_df['Score_Rel_Bias'].mean()
    print(f"\nOverall Average Relative Bias (Current, normalized): {rel_bias_current:.6f}")
    print(f"Overall Average Relative Bias (Score estimator): {rel_bias_score:.6f}")
    
    # Time comparison
    print(f"\nExecution Time Comparison:")
    print(f"Current method: {time_stats['current_mean']:.3f} ± {time_stats['current_std']:.3f} seconds")
    print(f"Score estimator: {time_stats['score_mean']:.3f} ± {time_stats['score_std']:.3f} seconds")
    print(f"Time ratio (score/current): {time_stats['time_ratio']:.2f}x")
    
    print(f"\nNote: MSE and RMSE computed against actual normalized true theta values")
    print(f"      RMSE = MSE / (true_value^2) - lower is better")
    print(f"      Relative Bias = (mean(estimated) - true) / true - positive = overestimation")
    
    # Save only the MSE table with metadata
    save_mse_results(results_folder, individual_mse_df, metadata, time_stats)
    print(f"✓ MSE results saved to: {results_folder}/mse.csv")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main() 