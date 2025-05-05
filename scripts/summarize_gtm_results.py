#!/usr/bin/env python3
# summarize_gtm_results.py
#
# Script for summarizing GTM-specific comparison results
# This script collects and analyzes metrics for different GTM variants

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Summarize GTM comparison results")
    parser.add_argument("--original-gtm-dir", required=True, help="Directory containing Original GTM results")
    parser.add_argument("--hybrid-gtm1-dir", required=True, help="Directory containing Hybrid GTM 1 results")
    parser.add_argument("--hybrid-gtm2-dir", required=True, help="Directory containing Hybrid GTM 2 results")
    parser.add_argument("--hybrid-gtm3-dir", required=True, help="Directory containing Hybrid GTM 3 results")
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

def collect_gtm_results(gtm_dir, method_name):
    """Collect results from a GTM directory"""
    results = []
    
    for model_dir in glob.glob(os.path.join(gtm_dir, "*")):
        if not os.path.isdir(model_dir):
            continue
            
        model = os.path.basename(model_dir)
        
        for rep_dir in glob.glob(os.path.join(model_dir, "R*")):
            if not os.path.isdir(rep_dir):
                continue
                
            rep = os.path.basename(rep_dir)
            comparison_file = os.path.join(rep_dir, "comparison.txt")
            
            if os.path.exists(comparison_file):
                metric_results = parse_comparison_file(comparison_file)
                
                results.append({
                    'Model': model,
                    'Method': method_name,
                    'Replicate': rep,
                    'RF': metric_results.get('RF', np.nan),
                    'FN': metric_results.get('FN', np.nan),
                    'FP': metric_results.get('FP', np.nan)
                })
    
    return results

def main():
    """Main function to summarize GTM results"""
    args = parse_args()
    
    # Collect results from each GTM method
    results = []
    
    # Original GTM
    results.extend(collect_gtm_results(
        args.original_gtm_dir, 
        "Original GTM (FastTree guide + FastTree subset)"
    ))
    
    # Hybrid GTM 1
    results.extend(collect_gtm_results(
        args.hybrid_gtm1_dir, 
        "Hybrid GTM 1 (FastTree guide + NJ-LogDet subset)"
    ))
    
    # Hybrid GTM 2
    results.extend(collect_gtm_results(
        args.hybrid_gtm2_dir, 
        "Hybrid GTM 2 (NJ-LogDet guide + FastTree subset)"
    ))
    
    # Hybrid GTM 3
    results.extend(collect_gtm_results(
        args.hybrid_gtm3_dir, 
        "Hybrid GTM 3 (NJ-LogDet guide + NJ-LogDet subset)"
    ))
    
    # Create dataframe
    results_df = pd.DataFrame(results)
    
    # Save raw results to CSV
    raw_file = f"{args.output_prefix}_raw.csv"
    results_df.to_csv(raw_file, index=False)
    print(f"Raw GTM results saved to {raw_file}")
    
    # Calculate average metrics for each method and model
    summary_df = results_df.groupby(['Model', 'Method']).agg({
        'RF': ['mean', 'std'],
        'FN': ['mean', 'std'],
        'FP': ['mean', 'std']
    }).reset_index()
    
    # Save summary to CSV
    summary_file = f"{args.output_prefix}_summary.csv"
    summary_df.to_csv(summary_file)
    print(f"GTM summary results saved to {summary_file}")
    
    # Create plots
    create_plots(results_df, args.output_prefix)

def create_plots(df, output_prefix):
    """Create plots to visualize GTM results"""
    # Skip plotting if no valid data
    if df.empty or df['FN'].isna().all() or df['FP'].isna().all():
        print("Warning: Not enough valid data for plotting")
        return
        
    # Average FN rates by method and model
    plt.figure(figsize=(14, 7))
    
    # Group by model and method, then calculate mean FN
    fn_means = df.groupby(['Model', 'Method'])['FN'].mean().reset_index()
    
    # Pivot to create a table suitable for grouped bar chart
    fn_pivot = fn_means.pivot(index='Method', columns='Model', values='FN')
    
    # Plot grouped bar chart
    ax = fn_pivot.plot(kind='bar', capsize=4)
    
    plt.title('Average False Negative Rates by GTM Method and Model')
    plt.ylabel('False Negative Rate')
    plt.xlabel('Method')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_fn_rates.png")
    
    # Average FP rates by method and model
    plt.figure(figsize=(14, 7))
    
    # Group by model and method, then calculate mean FP
    fp_means = df.groupby(['Model', 'Method'])['FP'].mean().reset_index()
    
    # Pivot to create a table suitable for grouped bar chart
    fp_pivot = fp_means.pivot(index='Method', columns='Model', values='FP')
    
    # Plot grouped bar chart
    ax = fp_pivot.plot(kind='bar', capsize=4)
    
    plt.title('Average False Positive Rates by GTM Method and Model')
    plt.ylabel('False Positive Rate')
    plt.xlabel('Method')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_fp_rates.png")
    
    # Average RF distances by method and model
    plt.figure(figsize=(14, 7))
    
    # Group by model and method, then calculate mean RF
    rf_means = df.groupby(['Model', 'Method'])['RF'].mean().reset_index()
    
    # Pivot to create a table suitable for grouped bar chart
    rf_pivot = rf_means.pivot(index='Method', columns='Model', values='RF')
    
    # Plot grouped bar chart
    ax = rf_pivot.plot(kind='bar', capsize=4)
    
    plt.title('Average Robinson-Foulds Distances by GTM Method and Model')
    plt.ylabel('RF Distance')
    plt.xlabel('Method')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_rf_distances.png")

if __name__ == "__main__":
    main()