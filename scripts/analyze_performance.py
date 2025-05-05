#!/usr/bin/env python3
# analyze_performance.py
#
# Script for analyzing runtime and memory usage of different methods

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import argparse

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Analyze performance metrics")
    parser.add_argument("--perf-dir", required=True, help="Directory containing performance files")
    parser.add_argument("--output-dir", required=True, help="Output directory for plots")
    parser.add_argument("--accuracy-file", required=False, help="Path to accuracy summary file")
    return parser.parse_args()

def parse_performance_file(file_path):
    """Parse a performance measurement file"""
    try:
        with open(file_path, 'r') as f:
            content = f.read().strip().split(',')
            if len(content) >= 2:
                elapsed_time = float(content[0])
                memory_kb = float(content[1]) if content[1] != "NA" else None
                
                # Extract method, model, and replicate from filename
                filename = os.path.basename(file_path)
                parts = filename.replace('.perf', '').split('_')
                
                if len(parts) >= 3:
                    method = parts[0]
                    model = parts[1]
                    rep = parts[2].replace('R', '')
                    
                    return {
                        'Method': method,
                        'Model': model,
                        'Replicate': rep,
                        'Runtime': elapsed_time,
                        'Memory': memory_kb if memory_kb != "NA" else None
                    }
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    
    return None

def parse_dependencies(results_dir):
    """Parse method dependencies from dependency files"""
    dependencies = {}
    
    # Map of directory names to method identifiers
    dir_method_map = {
        'hybrid_gtm1': 'hybrid_gtm1',
        'hybrid_gtm2': 'hybrid_gtm2',
        'hybrid_gtm3': 'hybrid_gtm3',
        'original_gtm': 'original_gtm'
    }
    
    for method_dir, method_id in dir_method_map.items():
        for model_dir in glob.glob(os.path.join(results_dir, method_dir, "*")):
            model = os.path.basename(model_dir)
            
            for rep_dir in glob.glob(os.path.join(model_dir, "R*")):
                rep = os.path.basename(rep_dir)
                
                dep_file = os.path.join(rep_dir, "dependencies.txt")
                if os.path.exists(dep_file):
                    with open(dep_file, 'r') as f:
                        deps = [line.strip() for line in f.readlines() if line.strip()]
                        dependencies[(method_id, model, rep)] = deps
    
    return dependencies

def calculate_total_performance(df, dependencies):
    """Calculate total runtime accounting for dependencies"""
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Add a new column for total runtime
    result_df['Total_Runtime'] = result_df['Runtime']
    result_df['Total_Memory'] = result_df['Memory']
    
    # For each GTM method, add the runtime of dependencies
    for idx, row in result_df.iterrows():
        method = row['Method']
        model = row['Model']
        rep = row['Replicate']
        
        # Check if this method has dependencies
        key = (method, model, rep)
        if key in dependencies:
            dep_methods = dependencies[key]
            
            # Find the runtime of each dependency
            dep_runtime = 0
            dep_memory = 0
            for dep in dep_methods:
                dep_rows = df[(df['Method'] == dep) & 
                              (df['Model'] == model) & 
                              (df['Replicate'] == rep)]
                
                if not dep_rows.empty:
                    dep_runtime += dep_rows['Runtime'].iloc[0]
                    if not pd.isna(dep_rows['Memory'].iloc[0]):
                        dep_memory = max(dep_memory, dep_rows['Memory'].iloc[0])
            
            # Add dependency runtime to total
            result_df.at[idx, 'Total_Runtime'] += dep_runtime
            
            # Use maximum memory across all steps
            if not pd.isna(row['Memory']) and not pd.isna(dep_memory):
                result_df.at[idx, 'Total_Memory'] = max(row['Memory'], dep_memory)
    
    return result_df

def aggregate_gtm_steps(df):
    """Aggregate steps for GTM methods by summing step times"""
    # Create a copy for modification
    result_df = df.copy()
    
    # Add a column for the base method (first part of method name)
    result_df['Base_Method'] = result_df['Method'].apply(lambda x: x.split('_')[0] if '_' in x else x)
    
    # Group by base method, model, and replicate
    aggregated = result_df.groupby(['Base_Method', 'Model', 'Replicate']).agg({
        'Runtime': 'sum',
        'Memory': 'max',
        'Total_Runtime': 'sum',
        'Total_Memory': 'max'
    }).reset_index()
    
    # Rename Base_Method back to Method for consistency
    aggregated = aggregated.rename(columns={'Base_Method': 'Method'})
    
    # Create method display names mapping
    method_display = {
        'hybrid_gtm1': 'Hybrid GTM 1',
        'hybrid_gtm2': 'Hybrid GTM 2',
        'hybrid_gtm3': 'Hybrid GTM 3',
        'original_gtm': 'Original GTM',
        'fasttree_gtr': 'FastTree (GTR)',
        'fasttree_jc': 'FastTree (JC)',
        'nj_jc': 'NJ (JC)',
        'nj_logdet': 'NJ (LogDet)',
        'nj_pdist': 'NJ (p-distance)',
        'fastme_bme': 'FastME (BME)'
    }
    
    # Apply display names
    aggregated['Method'] = aggregated['Method'].map(lambda x: method_display.get(x, x))
    
    return aggregated

def create_plots(df, output_dir):
    """Create performance comparison plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the style
    plt.figure(figsize=(12, 8))
    
    # Group by Method and Model, calculate mean runtime
    runtime_df = df.groupby(['Method', 'Model'])['Total_Runtime'].mean().reset_index()
    
    # Pivot for easier plotting
    runtime_pivot = runtime_df.pivot(index='Method', columns='Model', values='Total_Runtime')
    
    # Plot grouped bar chart
    ax = runtime_pivot.plot(kind='bar', rot=45)
    
    plt.title('Average Runtime by Method and Model')
    plt.ylabel('Runtime (seconds)')
    plt.xlabel('Method')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "runtime_comparison.png"), dpi=300)
    
    # ----- Memory Usage Comparison -----
    plt.figure(figsize=(12, 8))
    
    # Filter rows with valid memory data
    memory_df = df[df['Total_Memory'].notna()]
    
    if not memory_df.empty:
        # Convert KB to MB for readability
        memory_df['Memory_MB'] = memory_df['Total_Memory'] / 1024
        
        # Group by Method and Model, calculate mean memory
        memory_df = memory_df.groupby(['Method', 'Model'])['Memory_MB'].mean().reset_index()
        
        # Pivot for easier plotting
        memory_pivot = memory_df.pivot(index='Method', columns='Model', values='Memory_MB')
        
        # Plot grouped bar chart
        ax = memory_pivot.plot(kind='bar', rot=45)
        
        plt.title('Average Memory Usage by Method and Model')
        plt.ylabel('Memory Usage (MB)')
        plt.xlabel('Method')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "memory_comparison.png"), dpi=300)
    else:
        print("WARNING: No valid memory data found for plotting")
        
def create_tradeoff_plots(perf_df, accuracy_df, output_dir):
    """Create plots showing tradeoffs between performance and accuracy"""
    # Prepare accuracy data - map column names to expected format
    accuracy_df = accuracy_df.copy()
    column_mapping = {}
    for col in accuracy_df.columns:
        if col.startswith('RF') and 'mean' in col:
            column_mapping[col] = 'RF_mean'
        elif col.startswith('FN') and 'mean' in col:
            column_mapping[col] = 'FN_mean'
        elif col.startswith('FP') and 'mean' in col:
            column_mapping[col] = 'FP_mean'
    
    accuracy_df = accuracy_df.rename(columns=column_mapping)
    
    # Get just the columns we need
    accuracy_cols = ['Method', 'Model', 'RF_mean', 'FN_mean', 'FP_mean']
    accuracy_df = accuracy_df[[col for col in accuracy_cols if col in accuracy_df.columns]]
    
    # Convert 'Replicate' column in perf_df to string
    perf_df['Replicate'] = perf_df['Replicate'].astype(str)
    
    # Group both DataFrames by Method and Model to get average values
    perf_grouped = perf_df.groupby(['Method', 'Model']).agg({
        'Total_Runtime': 'mean',
        'Total_Memory': 'mean'
    }).reset_index()
    
    # Merge performance and accuracy data
    combined_df = pd.merge(
        perf_grouped, 
        accuracy_df, 
        on=['Method', 'Model'],
        how='inner'
    )
    
    if combined_df.empty:
        print("WARNING: No matching data found between performance and accuracy results")
        return
    
    # Create scatter plots for each model
    plt.figure(figsize=(14, 10))
    
    models = combined_df['Model'].unique()
    markers = ['o', 's', '^', 'D', 'v']
    
    # Runtime vs FN Rate
    plt.subplot(2, 2, 1)
    for i, model in enumerate(models):
        model_data = combined_df[combined_df['Model'] == model]
        plt.scatter(
            model_data['Total_Runtime'], 
            model_data['FN_mean'],
            label=model,
            marker=markers[i % len(markers)],
            s=80,
            alpha=0.7
        )
        
        for _, row in model_data.iterrows():
            plt.annotate(
                row['Method'],
                (row['Total_Runtime'], row['FN_mean']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8
            )
    
    plt.title('Runtime vs False Negative Rate')
    plt.xlabel('Runtime (seconds)')
    plt.ylabel('False Negative Rate')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Memory vs FN Rate
    plt.subplot(2, 2, 2)
    for i, model in enumerate(models):
        model_data = combined_df[combined_df['Model'] == model]
        plt.scatter(
            model_data['Total_Memory'] / 1024 if 'Total_Memory' in model_data else 0, # Convert to MB
            model_data['FN_mean'],
            label=model,
            marker=markers[i % len(markers)],
            s=80,
            alpha=0.7
        )
        
        for _, row in model_data.iterrows():
            plt.annotate(
                row['Method'],
                (row['Total_Memory'] / 1024 if 'Total_Memory' in row else 0, row['FN_mean']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8
            )
    
    plt.title('Memory Usage vs False Negative Rate')
    plt.xlabel('Memory (MB)')
    plt.ylabel('False Negative Rate')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Runtime vs RF Distance (if available)
    if 'RF_mean' in combined_df.columns:
        plt.subplot(2, 2, 3)
        for i, model in enumerate(models):
            model_data = combined_df[combined_df['Model'] == model]
            plt.scatter(
                model_data['Total_Runtime'], 
                model_data['RF_mean'],
                label=model,
                marker=markers[i % len(markers)],
                s=80,
                alpha=0.7
            )
            
            for _, row in model_data.iterrows():
                plt.annotate(
                    row['Method'],
                    (row['Total_Runtime'], row['RF_mean']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8
                )
        
        plt.title('Runtime vs RF Distance')
        plt.xlabel('Runtime (seconds)')
        plt.ylabel('Robinson-Foulds Distance')
        plt.grid(True, linestyle='--', alpha=0.7)
    
    # Memory vs RF Distance (if available)
    if 'RF_mean' in combined_df.columns and 'Total_Memory' in combined_df.columns:
        plt.subplot(2, 2, 4)
        for i, model in enumerate(models):
            model_data = combined_df[combined_df['Model'] == model]
            plt.scatter(
                model_data['Total_Memory'] / 1024, # Convert to MB
                model_data['RF_mean'],
                label=model,
                marker=markers[i % len(markers)],
                s=80,
                alpha=0.7
            )
            
            for _, row in model_data.iterrows():
                plt.annotate(
                    row['Method'],
                    (row['Total_Memory'] / 1024, row['RF_mean']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8
                )
        
        plt.title('Memory Usage vs RF Distance')
        plt.xlabel('Memory (MB)')
        plt.ylabel('Robinson-Foulds Distance')
        plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "performance_accuracy_tradeoff.png"), dpi=300)
    
def main():
    """Main function to analyze performance data"""
    args = parse_args()
    
    # Collect all performance files
    performance_files = glob.glob(os.path.join(args.perf_dir, "*.perf"))
    
    if not performance_files:
        print(f"No performance files found in {args.perf_dir}")
        return
    
    # Parse performance data
    results = []
    for file_path in performance_files:
        result = parse_performance_file(file_path)
        if result:
            results.append(result)
    
    if not results:
        print("No valid performance data found")
        return
    
    # Create DataFrame
    perf_df = pd.DataFrame(results)
    
    # Parse dependencies
    dependencies = parse_dependencies(os.path.dirname(args.perf_dir))
    
    # Calculate total performance
    total_perf_df = calculate_total_performance(perf_df, dependencies)

    # Aggregate GTM steps by summing step times
    aggregated_perf_df = aggregate_gtm_steps(total_perf_df)

    os.makedirs(args.output_dir, exist_ok=True)
    aggregated_perf_df.to_csv(os.path.join(args.output_dir, "performance_data.csv"), index=False)
    create_plots(aggregated_perf_df, args.output_dir)    

    # Load accuracy data if provided
    if args.accuracy_file and os.path.exists(args.accuracy_file):
        try:
            accuracy_df = pd.read_csv(args.accuracy_file)
            # Create tradeoff plots
            create_tradeoff_plots(total_perf_df, accuracy_df, args.output_dir)
        except Exception as e:
            print(f"Error processing accuracy data: {e}")
    
    print(f"Performance analysis completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
