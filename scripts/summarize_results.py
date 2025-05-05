#!/usr/bin/env python3
# summarize_results.py
#
# Script for summarizing tree comparison results
# This script collects and analyzes comparison metrics across methods and datasets

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import re

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Summarize tree comparison results")
    parser.add_argument("--input-dir", required=True, help="Base directory containing results")
    parser.add_argument("--method-patterns", required=True, help="Comma-separated list of file_pattern:method_name pairs")
    parser.add_argument("--output-prefix", required=True, help="Prefix for output files")
    return parser.parse_args()

def parse_comparison_file(file_path):
    """Parse a comparison file and extract RF, FN, and FP values"""
    results = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if "RF distance:" in line:
                    value = line.split(":")[1].strip()
                    results['RF'] = float(value) if value != 'N/A' else np.nan
                elif "FN rate:" in line:
                    value = line.split(":")[1].strip()
                    results['FN'] = float(value) if value != 'N/A' else np.nan
                elif "FP rate:" in line:
                    value = line.split(":")[1].strip()
                    results['FP'] = float(value) if value != 'N/A' else np.nan
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        results = {'RF': np.nan, 'FN': np.nan, 'FP': np.nan}
    return results

def main():
    """Main function to summarize tree comparison results"""
    args = parse_args()
    
    # Parse method patterns
    method_patterns = []
    for pattern_pair in args.method_patterns.split(','):
        parts = pattern_pair.split(':')
        if len(parts) == 2:
            method_patterns.append((parts[0], parts[1]))
        else:
            print(f"Warning: Invalid method pattern format: {pattern_pair}")
    
    # Create a dataframe to store results
    columns = ['Model', 'Method', 'Replicate', 'RF', 'FN', 'FP']
    results_df = pd.DataFrame(columns=columns)
    
    # Process each model condition directory
    for model_dir in glob.glob(os.path.join(args.input_dir, "*")):
        if not os.path.isdir(model_dir):
            continue
            
        model = os.path.basename(model_dir)
        
        # Process each replicate directory
        for rep_dir in glob.glob(os.path.join(model_dir, "R*")):
            if not os.path.isdir(rep_dir):
                continue
                
            # Extract replicate number
            rep = os.path.basename(rep_dir)
            
            # Process each method
            for file_pattern, method_name in method_patterns:
                # Look for comparison files matching the pattern
                comparison_files = glob.glob(os.path.join(rep_dir, f"{file_pattern}*_comparison.txt"))
                comparison_files.extend(glob.glob(os.path.join(rep_dir, f"{file_pattern}*.txt")))
                
                for comparison_file in comparison_files:
                    results = parse_comparison_file(comparison_file)
                    
                    # Add to dataframe
                    new_row = {
                        'Model': model,
                        'Method': method_name,
                        'Replicate': rep,
                        'RF': results.get('RF', np.nan),
                        'FN': results.get('FN', np.nan),
                        'FP': results.get('FP', np.nan)
                    }
                    results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Save raw results to CSV
    raw_file = f"{args.output_prefix}_raw.csv"
    results_df.to_csv(raw_file, index=False)
    print(f"Raw results saved to {raw_file}")
    
    # Calculate average metrics for each method and model
    summary_df = results_df.groupby(['Model', 'Method']).agg({
        'RF': ['mean', 'std'],
        'FN': ['mean', 'std'],
        'FP': ['mean', 'std']
    }).reset_index()
    
    # Save summary to CSV
    summary_file = f"{args.output_prefix}_summary.csv"
    summary_df.to_csv(summary_file)
    print(f"Summary results saved to {summary_file}")
    
    # Create plots
    create_plots(results_df, args.output_prefix)

def create_plots(df, output_prefix):
    """Create plots to visualize results"""
    # Skip plotting if no valid data
    if df.empty or df['FN'].isna().all() or df['FP'].isna().all():
        print("Warning: Not enough valid data for plotting")
        return
        
    # Average FN rates by method and model
    plt.figure(figsize=(12, 6))
    
    # Group by model and method, then calculate mean FN
    fn_means = df.groupby(['Model', 'Method'])['FN'].mean().reset_index()
    
    # Pivot to create a table suitable for grouped bar chart
    fn_pivot = fn_means.pivot(index='Method', columns='Model', values='FN')
    
    # Plot grouped bar chart
    ax = fn_pivot.plot(kind='bar', capsize=4)
    
    plt.title('Average False Negative Rates by Method and Model')
    plt.ylabel('False Negative Rate')
    plt.xlabel('Method')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_fn_rates.png")
    
    # Average FP rates by method and model
    plt.figure(figsize=(12, 6))
    
    # Group by model and method, then calculate mean FP
    fp_means = df.groupby(['Model', 'Method'])['FP'].mean().reset_index()
    
    # Pivot to create a table suitable for grouped bar chart
    fp_pivot = fp_means.pivot(index='Method', columns='Model', values='FP')
    
    # Plot grouped bar chart
    ax = fp_pivot.plot(kind='bar', capsize=4)
    
    plt.title('Average False Positive Rates by Method and Model')
    plt.ylabel('False Positive Rate')
    plt.xlabel('Method')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_fp_rates.png")
    
    # Average RF distances by method and model
    plt.figure(figsize=(12, 6))
    
    # Group by model and method, then calculate mean RF
    rf_means = df.groupby(['Model', 'Method'])['RF'].mean().reset_index()
    
    # Pivot to create a table suitable for grouped bar chart
    rf_pivot = rf_means.pivot(index='Method', columns='Model', values='RF')
    
    # Plot grouped bar chart
    ax = rf_pivot.plot(kind='bar', capsize=4)
    
    plt.title('Average Robinson-Foulds Distances by Method and Model')
    plt.ylabel('RF Distance')
    plt.xlabel('Method')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_rf_distances.png")

if __name__ == "__main__":
    main()