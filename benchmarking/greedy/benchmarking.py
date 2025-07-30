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
    current_results = Path("benchmarking/greedy/results.csv")
    score_estimator_results = Path("/Users/enzo-macbookpro/MyProjects/score-estimator/greedy/estimation_results.csv")
    
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

def extract_beta_estimates(df_current, df_score):
    """Extract beta estimates from both datasets for comparison."""
    
    # Extract beta_hat columns from current dataset
    beta_hat_cols = [col for col in df_current.columns if col.startswith('beta_hat_')]
    beta_hat_cols.sort()
    
    # Extract b2-b5 columns from score estimator dataset (corresponding to beta_hat_1 to beta_hat_4)
    score_beta_cols = ['b2', 'b3', 'b4', 'b5']
    
    print(f"\nBeta estimate columns found:")
    print(f"Current dataset: {beta_hat_cols}")
    print(f"Score estimator dataset: {score_beta_cols}")
    
    # Create comparison dataframes
    current_betas = df_current[beta_hat_cols].copy()
    current_betas.columns = [f'beta_hat_{i}' for i in range(len(beta_hat_cols))]
    
    score_betas = df_score[score_beta_cols].copy()
    score_betas.columns = [f'beta_hat_{i+1}' for i in range(len(score_beta_cols))]
    
    return current_betas, score_betas

def compare_statistics(current_betas, score_betas):
    """Compare statistical properties of beta estimates."""
    
    print("\n" + "="*60)
    print("STATISTICAL COMPARISON")
    print("="*60)
    
    # Basic statistics
    print("\n1. BASIC STATISTICS")
    print("-" * 40)
    
    print("\nCurrent Dataset Beta Estimates:")
    print(current_betas.describe())
    
    print("\nScore Estimator Dataset Beta Estimates:")
    print(score_betas.describe())
    
    # Mean comparison
    print("\n2. MEAN COMPARISON")
    print("-" * 40)
    current_means = current_betas.mean()
    score_means = score_betas.mean()
    
    comparison_df = pd.DataFrame({
        'Current_Mean': current_means,
        'Score_Mean': score_means,
        'Difference': current_means - score_means,
        'Relative_Diff_%': ((current_means - score_means) / score_means * 100)
    })
    
    print(comparison_df.round(4))
    
    # Variance comparison
    print("\n3. VARIANCE COMPARISON")
    print("-" * 40)
    current_vars = current_betas.var()
    score_vars = score_betas.var()
    
    var_comparison = pd.DataFrame({
        'Current_Variance': current_vars,
        'Score_Variance': score_vars,
        'Variance_Ratio': current_vars / score_vars
    })
    
    print(var_comparison.round(4))
    
    return comparison_df, var_comparison

def create_visualizations(current_betas, score_betas):
    """Create visualizations for comparison."""
    
    print("\n4. CREATING VISUALIZATIONS")
    print("-" * 40)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Beta Estimate Comparison: Current vs Score Estimator', fontsize=16)
    
    # 1. Box plot comparison
    ax1 = axes[0, 0]
    current_data = [current_betas[col] for col in current_betas.columns]
    score_data = [score_betas[col] for col in score_betas.columns]
    
    bp1 = ax1.boxplot(current_data, positions=range(1, len(current_betas.columns)+1), 
                      patch_artist=True, alpha=0.7, label='Current')
    bp2 = ax1.boxplot(score_data, positions=range(1, len(score_betas.columns)+1), 
                      patch_artist=True, alpha=0.7, label='Score Estimator')
    
    # Color the boxes
    for patch in bp1['boxes']:
        patch.set_facecolor('lightblue')
    for patch in bp2['boxes']:
        patch.set_facecolor('lightcoral')
    
    ax1.set_xlabel('Beta Index')
    ax1.set_ylabel('Beta Estimate Value')
    ax1.set_title('Box Plot Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Histogram comparison
    ax2 = axes[0, 1]
    for i, col in enumerate(current_betas.columns):
        ax2.hist(current_betas[col], alpha=0.6, bins=20, label=f'{col} (Current)', density=True)
        ax2.hist(score_betas[col], alpha=0.6, bins=20, label=f'{col} (Score)', density=True)
    
    ax2.set_xlabel('Beta Estimate Value')
    ax2.set_ylabel('Density')
    ax2.set_title('Distribution Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Scatter plot of means
    ax3 = axes[1, 0]
    current_means = current_betas.mean()
    score_means = score_betas.mean()
    
    ax3.scatter(score_means, current_means, alpha=0.7, s=100)
    ax3.plot([score_means.min(), score_means.max()], [score_means.min(), score_means.max()], 
             'r--', alpha=0.8, label='Perfect Agreement')
    
    ax3.set_xlabel('Score Estimator Mean')
    ax3.set_ylabel('Current Mean')
    ax3.set_title('Mean Comparison Scatter Plot')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Time series of estimates (if available)
    ax4 = axes[1, 1]
    for i, col in enumerate(current_betas.columns):
        ax4.plot(current_betas[col].values, alpha=0.7, label=f'{col} (Current)', linewidth=1)
        ax4.plot(score_betas[col].values, alpha=0.7, label=f'{col} (Score)', linewidth=1, linestyle='--')
    
    ax4.set_xlabel('Sample Index')
    ax4.set_ylabel('Beta Estimate Value')
    ax4.set_title('Time Series Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = "benchmarking/greedy/comparison_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to: {output_path}")
    
    plt.show()

def perform_hypothesis_tests(current_betas, score_betas):
    """Perform statistical hypothesis tests to compare the distributions."""
    
    print("\n5. HYPOTHESIS TESTS")
    print("-" * 40)
    
    from scipy import stats
    
    results = []
    
    for i, col in enumerate(current_betas.columns):
        current_data = current_betas[col].dropna()
        score_data = score_betas[col].dropna()
        
        # Ensure same length for comparison
        min_len = min(len(current_data), len(score_data))
        current_data = current_data.iloc[:min_len]
        score_data = score_data.iloc[:min_len]
        
        # T-test for difference in means
        t_stat, p_value = stats.ttest_ind(current_data, score_data)
        
        # Kolmogorov-Smirnov test for distribution difference
        ks_stat, ks_p_value = stats.ks_2samp(current_data, score_data)
        
        # Mann-Whitney U test (non-parametric)
        mw_stat, mw_p_value = stats.mannwhitneyu(current_data, score_data, alternative='two-sided')
        
        results.append({
            'Beta_Index': col,
            'T_Statistic': t_stat,
            'T_P_Value': p_value,
            'KS_Statistic': ks_stat,
            'KS_P_Value': ks_p_value,
            'MW_Statistic': mw_stat,
            'MW_P_Value': mw_p_value
        })
    
    results_df = pd.DataFrame(results)
    print(results_df.round(4))
    
    # Summary of significant differences
    print("\nSummary of Significant Differences (p < 0.05):")
    print("-" * 50)
    
    for _, row in results_df.iterrows():
        significant_tests = []
        if row['T_P_Value'] < 0.05:
            significant_tests.append("T-test")
        if row['KS_P_Value'] < 0.05:
            significant_tests.append("KS-test")
        if row['MW_P_Value'] < 0.05:
            significant_tests.append("MW-test")
        
        if significant_tests:
            print(f"{row['Beta_Index']}: {', '.join(significant_tests)}")
        else:
            print(f"{row['Beta_Index']}: No significant differences detected")
    
    return results_df

def main():
    """Main function to run the comparison analysis."""
    
    print("="*60)
    print("ESTIMATION RESULTS COMPARISON")
    print("="*60)
    print("Comparing first 100 lines of each dataset")
    print("="*60)
    
    # Load data
    df_current, df_score = load_and_prepare_data()
    
    if df_current is None or df_score is None:
        print("Failed to load data. Exiting.")
        return
    
    # Extract beta estimates
    current_betas, score_betas = extract_beta_estimates(df_current, df_score)
    
    if current_betas is None or score_betas is None:
        print("Failed to extract beta estimates. Exiting.")
        return
    
    # Perform comparisons
    comparison_df, var_comparison = compare_statistics(current_betas, score_betas)
    
    # Create visualizations
    create_visualizations(current_betas, score_betas)
    
    # Perform hypothesis tests
    test_results = perform_hypothesis_tests(current_betas, score_betas)
    
    # Save results to CSV
    output_file = "benchmarking/greedy/comparison_results.csv"
    comparison_df.to_csv(output_file, index=True)
    print(f"\n✓ Comparison results saved to: {output_file}")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
