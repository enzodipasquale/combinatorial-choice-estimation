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
    current_results = Path("benchmarking/knapsack/results.csv")
    score_estimator_results = Path("/Users/enzo-macbookpro/MyProjects/score-estimator/knapsack/estimation_results.csv")
    
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
        patch.set_facecolor('lightgreen')
        patch.set_alpha(0.7)
    for patch in bp2['boxes']:
        patch.set_facecolor('lightcoral')
        patch.set_alpha(0.7)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='lightgreen', alpha=0.7, label='Current (normalized)'),
                      Patch(facecolor='lightcoral', alpha=0.7, label='Score estimator')]
    ax.legend(handles=legend_elements)
    
    ax.set_ylabel('Normalized Beta Estimate')
    ax.set_title('Boxplot of Normalized Beta Estimates (k-1) - Knapsack')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plot_file = results_folder / f"boxplot_normalized_{timestamp}.png"
    plt.savefig(plot_file, dpi=300)
    print(f"✓ Boxplot saved to: {plot_file}")
    plt.show()

def mean_squared_error(estimates, true_value=1.0):
    """Compute mean squared error to the true value (default 1.0)."""
    return ((estimates - true_value) ** 2).mean()

def compute_individual_mse(norm_betas_df, score_betas, true_value=1.0):
    """Compute MSE for each individual beta estimate."""
    mse_results = []
    
    for col in norm_betas_df.columns:
        current_mse = mean_squared_error(norm_betas_df[col], true_value)
        score_mse = mean_squared_error(score_betas[col], true_value)
        
        mse_results.append({
            'Beta_Index': col,
            'Current_MSE': current_mse,
            'Score_MSE': score_mse,
            'Difference': current_mse - score_mse
        })
    
    return pd.DataFrame(mse_results)

def create_results_folder():
    """Create results subfolder if it doesn't exist."""
    results_folder = Path("benchmarking/knapsack/comparison_results")
    results_folder.mkdir(exist_ok=True)
    return results_folder

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
        'subproblem': first_row.get('subproblem', '')
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
        'current_mse': mse_current,
        'score_mse': mse_score,
        'mse_difference': mse_current - mse_score
    }
    
    # Save individual MSEs with metadata
    individual_results = individual_mse_df.copy()
    for key, value in metadata.items():
        individual_results[key] = value
    individual_results['timestamp'] = timestamp
    
    # Append to existing files or create new ones
    summary_file = results_folder / "summary_mse.csv"
    individual_file = results_folder / "individual_mse.csv"
    
    # Save/append summary results
    if summary_file.exists():
        existing_summary = pd.read_csv(summary_file)
        updated_summary = pd.concat([existing_summary, pd.DataFrame([summary_row])], ignore_index=True)
    else:
        updated_summary = pd.DataFrame([summary_row])
    
    updated_summary.to_csv(summary_file, index=False)
    
    # Save/append individual results
    if individual_file.exists():
        existing_individual = pd.read_csv(individual_file)
        updated_individual = pd.concat([existing_individual, individual_results], ignore_index=True)
    else:
        updated_individual = individual_results
    
    updated_individual.to_csv(individual_file, index=False)
    
    return summary_file, individual_file

def main():
    print("="*60)
    print("NORMALIZED ESTIMATION RESULTS COMPARISON - KNAPSACK")
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
    
    # Boxplot
    boxplot_normalized(norm_betas_df, score_betas, results_folder)
    
    # MSE calculation for each beta estimate
    individual_mse_df = compute_individual_mse(norm_betas_df, score_betas)
    print("\nIndividual Mean Squared Errors:")
    print("=" * 50)
    print(individual_mse_df.round(6))
    
    # Overall MSE calculation
    mse_current = individual_mse_df['Current_MSE'].mean()
    mse_score = individual_mse_df['Score_MSE'].mean()
    print(f"\nOverall Mean Squared Error (Current, normalized): {mse_current:.6f}")
    print(f"Overall Mean Squared Error (Score estimator): {mse_score:.6f}")
    
    # Save results with metadata
    summary_file, individual_file = save_results_with_metadata(
        results_folder, individual_mse_df, metadata, mse_current, mse_score
    )
    print(f"✓ Summary results appended to: {summary_file}")
    print(f"✓ Individual results appended to: {individual_file}")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main() 