import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # Import seaborn
import argparse
import os
import numpy as np # For log scale handling if needed

def visualize_match_count_distributions(csv_filepath: str, use_log_scale: bool = False):
    """
    Generates side-by-side violin plots with overlaid strip plots to visualize 
    the distribution of InfiniGram match_count based on LLM self-correction scores.

    Args:
        csv_filepath (str): Path to the input CSV file.
        use_log_scale (bool): Whether to use a logarithmic scale for the match_count axis.
    """
    if not os.path.exists(csv_filepath):
        print(f"Error: File not found at '{csv_filepath}'")
        return

    try:
        df = pd.read_csv(csv_filepath)
    except Exception as e:
        print(f"Error reading CSV file '{csv_filepath}': {e}")
        return

    required_columns = ['match_count', 'llm_correctness_score']
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        print(f"Error: CSV file must contain columns: {', '.join(missing_cols)}.")
        return

    df['match_count'] = pd.to_numeric(df['match_count'], errors='coerce')
    df_filtered = df[df['llm_correctness_score'].isin([0, 1]) & df['match_count'].notna()].copy()
    
    if df_filtered.empty:
        print("No data for visualization after filtering (LLM scores 0/1, valid match_counts).")
        return

    df_filtered.loc[:, 'llm_correctness_score'] = df_filtered['llm_correctness_score'].astype(int)
    # Map scores to more descriptive labels for the plot
    df_filtered.loc[:, 'LLM Correctness'] = df_filtered['llm_correctness_score'].map({0: 'Incorrect (0)', 1: 'Correct (1)'})

    plt.figure(figsize=(12, 8))
    
    # Define the order of categories for the x-axis
    order = ['Incorrect (0)', 'Correct (1)']
    palette = {'Incorrect (0)': 'red', 'Correct (1)': 'green'}

    # Create violin plots
    sns.violinplot(x='LLM Correctness', y='match_count', data=df_filtered, 
                   order=order, palette=palette, inner=None, linewidth=1.5)
                   # inner=None removes the default boxplot/quartiles inside the violin

    # Overlay strip plots (jittered individual points)
    sns.stripplot(x='LLM Correctness', y='match_count', data=df_filtered, 
                  order=order, jitter=True, alpha=0.5, size=5, 
                  palette=palette, dodge=True, edgecolor='gray', linewidth=0.5)

    plt.xlabel("LLM Self-Correction Outcome")
    plt.ylabel(f"InfiniGram Match Count{' (Log Scale)' if use_log_scale else ''}")
    plt.title(f"Distribution of InfiniGram Match Counts by LLM Correctness{' (Log Scale)' if use_log_scale else ''}")
    
    if use_log_scale:
        # Apply log scale. Add 1 to avoid log(0) issues if match_count can be 0.
        # Filter out non-positive values if any before log transformation if necessary
        # For plotting, it's often better to just set the scale on the axis.
        plt.yscale('log')
        # Adjust y-axis limits if counts are very small, to prevent display issues with log scale
        # For example, if min count is 0, log scale will have issues. Handle this by perhaps setting a small lower bound.
        # If df_filtered['match_count'].min() <= 0 and use_log_scale:
        #    plt.ylim(bottom=0.1) # Adjust as needed, or filter data further

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plot_filename = f"match_count_distribution{'_log' if use_log_scale else ''}.png"
    try:
        plt.savefig(plot_filename)
        print(f"Visualization saved to '{plot_filename}'")
    except Exception as e:
        print(f"Error saving plot: {e}")
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize InfiniGram match_count distributions by LLM self-correction scores.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "csv_filepath", 
        nargs='?',
        default="infinigram_query_results_heuristic_AND_llm.csv",
        help="Path to the input CSV results file."
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Use a logarithmic scale for the match_count axis."
    )
    
    args = parser.parse_args()
    visualize_match_count_distributions(args.csv_filepath, args.log) 