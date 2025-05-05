import argparse
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Create publication-quality comparison plots")
    parser.add_argument("--baseline-summary", required=True, help="Baseline summary CSV file")
    parser.add_argument("--gtm-summary", required=True, help="GTM summary CSV file")
    parser.add_argument("--output-dir", required=True, help="Output directory for plots")
    return parser.parse_args()

def main():
    """Main function to create beautiful interactive plots"""
    args = parse_args()
    
    # Define model colors
    model_colors = {
        '1000M1': '#1f77b4',  # Blue
        '1000M4': '#ff7f0e',  # Orange
    }

    # Read summary files
    print(f"Reading baseline summary: {args.baseline_summary}")
    baseline_df = pd.read_csv(args.baseline_summary)
    print(f"Reading GTM summary: {args.gtm_summary}")
    gtm_df = pd.read_csv(args.gtm_summary)

    # --- DROP 'Unnamed:*' columns if present ---
    baseline_df = baseline_df.loc[:, ~baseline_df.columns.str.startswith('Unnamed')]
    gtm_df = gtm_df.loc[:, ~gtm_df.columns.str.startswith('Unnamed')]
    
    # Debug prints
    print("Baseline DataFrame columns:", baseline_df.columns.tolist())
    print("GTM DataFrame columns:", gtm_df.columns.tolist())
    
    # Map original columns to expected names
    column_mapping = {
        'RF':  'RF_mean',
        'RF.1':'RF_std',
        'FN':  'FN_mean',
        'FN.1':'FN_std',
        'FP':  'FP_mean',
        'FP.1':'FP_std'
    }
    baseline_df = baseline_df.rename(columns=column_mapping)
    gtm_df = gtm_df.rename(columns=column_mapping)
    
    # Ensure numeric conversion
    numeric_columns = ['RF_mean','RF_std','FN_mean','FN_std','FP_mean','FP_std']
    for col in numeric_columns:
        if col in baseline_df.columns:
            baseline_df[col] = (baseline_df[col].astype(str)
                                .str.replace(r'[^\d.]', '', regex=True))
            baseline_df[col] = pd.to_numeric(baseline_df[col], errors='coerce')
        if col in gtm_df.columns:
            gtm_df[col] = (gtm_df[col].astype(str)
                           .str.replace(r'[^\d.]', '', regex=True))
            gtm_df[col] = pd.to_numeric(gtm_df[col], errors='coerce')
    
    # --- Normalize RF distances and their std to [0,1] ---
    all_rf = pd.concat([baseline_df['RF_mean'], gtm_df['RF_mean']])
    max_rf = all_rf.max() if not all_rf.empty else 0
    if max_rf > 0:
        for df in (baseline_df, gtm_df):
            df['RF_norm'] = df['RF_mean'] / max_rf
            df['RF_norm_std'] = df['RF_std'] / max_rf
    else:
        for df in (baseline_df, gtm_df):
            df['RF_norm'] = 0
            df['RF_norm_std'] = 0

    # Combine and fill NaNs
    combined_df = pd.concat([baseline_df, gtm_df], ignore_index=True).fillna(0)

    # ----- DROP any rows where Model or Method is the literal zero -----
    combined_df = combined_df[
        ~(
            (combined_df['Model'].astype(str) == '0') |
            (combined_df['Method'].astype(str) == '0')
        )
    ]
    
    # Save combined summary
    os.makedirs(args.output_dir, exist_ok=True)
    combined_df.to_csv(f"{args.output_dir}/all_methods_summary.csv", index=False)
    
    # Generate standard plots
    create_horizontal_bar_plots(combined_df, args.output_dir, model_colors)
    create_complexity_scatter_plot(combined_df, args.output_dir, model_colors)
    create_comparison_matrix(combined_df, args.output_dir, model_colors)
    
    # Generate improved plots
    create_gtm_methods_matrix(combined_df, args.output_dir, model_colors)
    create_normalized_metrics_comparison(combined_df, args.output_dir, model_colors)
    
    # If performance data is available
    perf_file = os.path.join(os.path.dirname(args.output_dir), "performance_data.csv")
    if os.path.exists(perf_file):
        try:
            perf_df = pd.read_csv(perf_file)
            create_improved_performance_comparisons(perf_df, args.output_dir)
        except Exception as e:
            print(f"Error processing performance data: {e}")
    
    print("All plots created successfully!")

def create_gtm_methods_matrix(df, output_dir, model_colors):
    """Create fixed comprehensive comparison matrix for GTM methods with proper spacing"""
    # Filter for only GTM methods
    gtm_methods = [
        'Original GTM (FastTree guide + FastTree subset)',
        'Hybrid GTM 1 (FastTree guide + NJ-LogDet subset)',
        'Hybrid GTM 2 (NJ-LogDet guide + FastTree subset)',
        'Hybrid GTM 3 (NJ-LogDet guide + NJ-LogDet subset)'
    ]
    
    # Use shorter display names
    method_display_names = {
        'Original GTM (FastTree guide + FastTree subset)': 'Original GTM',
        'Hybrid GTM 1 (FastTree guide + NJ-LogDet subset)': 'Hybrid GTM 1',
        'Hybrid GTM 2 (NJ-LogDet guide + FastTree subset)': 'Hybrid GTM 2',
        'Hybrid GTM 3 (NJ-LogDet guide + NJ-LogDet subset)': 'Hybrid GTM 3'
    }
    
    gtm_df = df[df['Method'].isin(gtm_methods)].copy()
    gtm_df['Display_Name'] = gtm_df['Method'].map(method_display_names)
    
    models = gtm_df['Model'].unique()
    
    # Create the figure with properly spaced subplots
    fig = make_subplots(
        rows=len(models), 
        cols=3,
        subplot_titles=[
            f"{model} - Normalized RF" for model in models
        ] + [
            f"{model} - False Negative Rate" for model in models
        ] + [
            f"{model} - False Positive Rate" for model in models
        ],
        shared_yaxes=True,
        vertical_spacing=0.15,  # Increased spacing to avoid overlaps
        horizontal_spacing=0.08
    )
    
    # Process data by aggregating
    agg_dict = {}
    for col in gtm_df.columns:
        if col not in ['Model', 'Method', 'Display_Name']:
            agg_dict[col] = 'mean' if pd.api.types.is_numeric_dtype(gtm_df[col]) else 'first'
    agg_dict['Display_Name'] = 'first'
    
    grouped_df = gtm_df.groupby(['Model', 'Method']).agg(agg_dict).reset_index()
    
    # Metric colors
    metric_colors = {
        'RF': 'rgba(31, 119, 180, 0.8)',   # Blue
        'FN': 'rgba(255, 127, 14, 0.8)',   # Orange
        'FP': 'rgba(44, 160, 44, 0.8)',    # Green
    }
    
    # Add traces for each model and metric
    for i, model in enumerate(models):
        md = grouped_df[grouped_df['Model']==model].sort_values('Display_Name')
        
        # RF column
        fig.add_trace(go.Bar(
            x=md['RF_norm'], y=md['Display_Name'], 
            name=f"{model} - RF",
            orientation='h',
            error_x=dict(
                type='data',
                array=md['RF_norm_std'],
                visible=True,
                color='rgba(0,0,0,0.3)',
                thickness=1.5,
                width=4
            ),
            marker=dict(color=metric_colors['RF']),
            hovertemplate='<b>%{y}</b><br>RF: %{x:.4f} ± %{error_x.array:.4f}<extra></extra>',
            showlegend=(i==0)
        ), row=i+1, col=1)
        
        # Add text annotations
        for j, (_, row) in enumerate(md.iterrows()):
            fig.add_annotation(
                x=row['RF_norm'] + 0.01,
                y=row['Display_Name'],
                text=f"{row['RF_norm']:.4f}",
                showarrow=False,
                font=dict(size=9, color='black'),
                xanchor='left',
                row=i+1, col=1
            )
        
        # FN column
        fig.add_trace(go.Bar(
            x=md['FN_mean'], y=md['Display_Name'], 
            name=f"{model} - FN",
            orientation='h',
            error_x=dict(
                type='data',
                array=md['FN_std'],
                visible=True,
                color='rgba(0,0,0,0.3)',
                thickness=1.5,
                width=4
            ),
            marker=dict(color=metric_colors['FN']),
            hovertemplate='<b>%{y}</b><br>FN: %{x:.4f} ± %{error_x.array:.4f}<extra></extra>',
            showlegend=(i==0)
        ), row=i+1, col=2)
        
        # Add text annotations
        for j, (_, row) in enumerate(md.iterrows()):
            fig.add_annotation(
                x=row['FN_mean'] + 0.01,
                y=row['Display_Name'],
                text=f"{row['FN_mean']:.4f}",
                showarrow=False,
                font=dict(size=9, color='black'),
                xanchor='left',
                row=i+1, col=2
            )
        
        # FP column
        fig.add_trace(go.Bar(
            x=md['FP_mean'], y=md['Display_Name'], 
            name=f"{model} - FP",
            orientation='h',
            error_x=dict(
                type='data',
                array=md['FP_std'],
                visible=True,
                color='rgba(0,0,0,0.3)',
                thickness=1.5,
                width=4
            ),
            marker=dict(color=metric_colors['FP']),
            hovertemplate='<b>%{y}</b><br>FP: %{x:.4f} ± %{error_x.array:.4f}<extra></extra>',
            showlegend=(i==0)
        ), row=i+1, col=3)
        
        # Add text annotations
        for j, (_, row) in enumerate(md.iterrows()):
            fig.add_annotation(
                x=row['FP_mean'] + 0.01,
                y=row['Display_Name'],
                text=f"{row['FP_mean']:.4f}",
                showarrow=False,
                font=dict(size=9, color='black'),
                xanchor='left',
                row=i+1, col=3
            )
    
    # Update layout with better spacing and formatting
    fig.update_layout(
        title=dict(
            text='GTM Methods Comprehensive Comparison Matrix', 
            x=0.5, 
            font=dict(size=20, family="Arial", color="#333333"),
            pad=dict(t=20, b=20)
        ),
        plot_bgcolor='rgba(250,250,250,0.9)',
        width=1200, 
        height=250 * len(models) + 100,  # Add extra space for title
        margin=dict(l=20, r=20, t=100, b=60),  # Increased top margin
        font=dict(family="Arial, sans-serif", size=12),
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=-0.18/len(models), 
            xanchor="center", 
            x=0.5,
            title=dict(text='Metrics'),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.1)',
            borderwidth=1
        ),
        shapes=[
            # Add subtle grid lines
            dict(
                type="line", xref="paper", yref="paper",
                x0=0.33, y0=0, x1=0.33, y1=1, line=dict(color="rgba(0,0,0,0.1)", width=1)
            ),
            dict(
                type="line", xref="paper", yref="paper",
                x0=0.66, y0=0, x1=0.66, y1=1, line=dict(color="rgba(0,0,0,0.1)", width=1)
            )
        ]
    )
    
    # Update axes with clearer formatting
    for i in range(len(models)):
        fig.update_xaxes(
            title_text="Normalized RF", 
            title_font=dict(size=12), 
            row=i+1, 
            col=1,
            showgrid=True, 
            gridwidth=1, 
            gridcolor='rgba(230,230,230,0.8)',
            zeroline=True,
            zerolinecolor='rgba(0,0,0,0.2)',
            zerolinewidth=1
        )
        fig.update_xaxes(
            title_text="FN Rate", 
            title_font=dict(size=12), 
            row=i+1, 
            col=2,
            showgrid=True, 
            gridwidth=1, 
            gridcolor='rgba(230,230,230,0.8)',
            zeroline=True,
            zerolinecolor='rgba(0,0,0,0.2)',
            zerolinewidth=1
        )
        fig.update_xaxes(
            title_text="FP Rate", 
            title_font=dict(size=12), 
            row=i+1, 
            col=3,
            showgrid=True, 
            gridwidth=1, 
            gridcolor='rgba(230,230,230,0.8)',
            zeroline=True,
            zerolinecolor='rgba(0,0,0,0.2)',
            zerolinewidth=1
        )
    
    # Save the updated plot
    fig.write_html(f"{output_dir}/gtm_methods_matrix_fixed.html")
    fig.write_image(f"{output_dir}/gtm_methods_matrix_fixed.png", scale=3)

def create_improved_performance_comparisons(perf_df, output_dir):
    """Create visually appealing runtime and memory comparisons between method types"""
    # Process performance data
    if perf_df.empty:
        print("No performance data available")
        return
    
    # Add method type classification
    method_types = {
        'fasttree_gtr': 'Baseline',
        'fasttree_jc': 'Baseline',
        'nj_jc': 'Baseline',
        'nj_logdet': 'Baseline',
        'nj_pdist': 'Baseline',
        'fastme_bme': 'Baseline',
        'original_gtm': 'Original GTM',
        'hybrid_gtm1': 'Hybrid GTM',
        'hybrid_gtm2': 'Hybrid GTM',
        'hybrid_gtm3': 'Hybrid GTM'
    }
    
    # Add method type and pretty names
    perf_df['Method_Type'] = perf_df['Method'].apply(lambda x: method_types.get(x, 'Other'))
    
    # Method display names
    method_display = {
        'fasttree_gtr': 'FastTree (GTR)',
        'fasttree_jc': 'FastTree (JC)',
        'nj_jc': 'NJ (JC)',
        'nj_logdet': 'NJ (LogDet)',
        'nj_pdist': 'NJ (p-distance)',
        'fastme_bme': 'FastME (BME)',
        'original_gtm': 'Original GTM',
        'hybrid_gtm1': 'Hybrid GTM 1',
        'hybrid_gtm2': 'Hybrid GTM 2',
        'hybrid_gtm3': 'Hybrid GTM 3'
    }
    perf_df['Display_Name'] = perf_df['Method'].map(method_display)
    
    # Method type colors for consistency
    type_colors = {
        'Baseline': '#2c3e50',
        'Original GTM': '#e74c3c',
        'Hybrid GTM': '#3498db'
    }
    
    # Group by method and model, calculate average performance
    grouped_perf = perf_df.groupby(['Method', 'Model', 'Method_Type', 'Display_Name']).agg({
        'Total_Runtime': 'mean',
        'Total_Memory': 'mean'
    }).reset_index()
    
    # Create Runtime Comparison Plot - Grouped by Method Types
    fig = go.Figure()
    
    # First sort by method type and then by runtime
    grouped_perf = grouped_perf.sort_values(['Method_Type', 'Total_Runtime'])
    
    # Get unique models for color assignment
    models = grouped_perf['Model'].unique()
    model_colors = {
        '1000M1': '#2ecc71',  # Green
        '1000M4': '#9b59b6'   # Purple
    }
    
    # Add traces for each method type
    for method_type in ['Baseline', 'Original GTM', 'Hybrid GTM']:
        type_data = grouped_perf[grouped_perf['Method_Type'] == method_type]
        
        for model in models:
            model_data = type_data[type_data['Model'] == model]
            
            # Skip if no data for this combination
            if model_data.empty:
                continue
                
            fig.add_trace(go.Bar(
                x=model_data['Display_Name'],
                y=model_data['Total_Runtime'],
                name=f"{model} - {method_type}",
                marker=dict(
                    color=model_colors.get(model, '#1f77b4'),
                    line=dict(
                        color=type_colors.get(method_type, '#333333'),
                        width=2
                    ),
                    pattern=dict(
                        shape=['/', '\\', 'x', '-', '|', '+', '.'][list(method_types.values()).index(method_type) % 7],
                        solidity=0.7
                    )
                ),
                opacity=0.85,
                hovertemplate="<b>%{x}</b><br>" +
                              "Model: " + model + "<br>" +
                              "Type: " + method_type + "<br>" +
                              "Runtime: %{y:.6f} seconds<extra></extra>"  # Changed to 6 decimal places
            ))
    
    # Customize layout
    fig.update_layout(
        title=dict(
            text='Runtime Comparison by Method Type',
            font=dict(size=24, family='Arial, sans-serif', color='#333333'),
            x=0.5
        ),
        xaxis=dict(
            title='Method',
            titlefont=dict(size=16),
            tickangle=45,
            tickfont=dict(size=12),
            gridcolor='rgba(230,230,230,0.5)',
            showgrid=False,
            automargin=True
        ),
        yaxis=dict(
            title='Runtime (seconds)',
            titlefont=dict(size=16),
            tickfont=dict(size=12),
            gridcolor='rgba(230,230,230,0.5)',
            showgrid=True,
            zeroline=True,
            zerolinecolor='rgba(0,0,0,0.2)',
            zerolinewidth=1,
            tickformat='.6f'  # Format y-axis ticks with 6 decimal places
        ),
        barmode='group',
        bargap=0.15,
        bargroupgap=0.1,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1,
            font=dict(size=12)
        ),
        plot_bgcolor='rgba(250,250,250,0.9)',
        width=1200,
        height=800,
        margin=dict(l=60, r=60, t=100, b=120),
        shapes=[
            # Add subtle section dividers
            dict(
                type="rect",
                xref="paper", yref="paper",
                x0=0, y0=0, x1=1, y1=1,
                line=dict(color="rgba(0,0,0,0.05)", width=1),
                layer="below"
            )
        ],
        annotations=[
            # Add section headers
            dict(
                xref="paper", yref="paper",
                x=0.5, y=-0.15,
                text="Method Types: Baseline = Standard Methods | Original GTM = FastTree Guide + FastTree Subset | Hybrid GTM = Combinations of FastTree and NJ-LogDet",
                showarrow=False,
                font=dict(size=12, color="#555555")
            )
        ]
    )
    
    # Add value annotations for important methods with higher precision
    for i, row in grouped_perf.iterrows():
        if 'GTM' in row['Method_Type']:  # Only annotate GTM methods
            fig.add_annotation(
                x=row['Display_Name'],
                y=row['Total_Runtime'],
                text=f"{row['Total_Runtime']:.6f}s",  # Changed to 6 decimal places
                showarrow=False,
                yshift=10,
                font=dict(size=10, color="#333333"),
                bgcolor='rgba(255,255,255,0.7)',
                bordercolor='rgba(0,0,0,0.1)',
                borderwidth=1,
                borderpad=3,
                opacity=0.8
            )
    
    # Save runtime comparison
    fig.write_html(f"{output_dir}/improved_runtime_comparison.html")
    fig.write_image(f"{output_dir}/improved_runtime_comparison.png", scale=3)
    
    # Memory comparison plot - focus on GTM methods
    gtm_data = grouped_perf[grouped_perf['Method_Type'].isin(['Original GTM', 'Hybrid GTM'])]
    
    if not gtm_data.empty and not gtm_data['Total_Memory'].isna().all():
        fig_mem = go.Figure()
        
        # Convert KB to MB for better readability
        gtm_data['Memory_MB'] = gtm_data['Total_Memory'] / 1024
        
        # Sort by memory usage
        gtm_data = gtm_data.sort_values(['Method_Type', 'Memory_MB'])
        
        # Add trace for each GTM method
        for method_type in ['Original GTM', 'Hybrid GTM']:
            type_data = gtm_data[gtm_data['Method_Type'] == method_type]
            
            for model in models:
                model_data = type_data[type_data['Model'] == model]
                
                # Skip if no data
                if model_data.empty:
                    continue
                    
                fig_mem.add_trace(go.Bar(
                    x=model_data['Display_Name'],
                    y=model_data['Memory_MB'],
                    name=f"{model} - {method_type}",
                    marker=dict(
                        color=model_colors.get(model, '#1f77b4'),
                        line=dict(
                            color=type_colors.get(method_type, '#333333'),
                            width=2
                        ),
                        pattern=dict(
                            shape=['/', '\\', 'x'][list(['Original GTM', 'Hybrid GTM']).index(method_type) % 3],
                            solidity=0.7
                        )
                    ),
                    opacity=0.85,
                    hovertemplate="<b>%{x}</b><br>" +
                                  "Model: " + model + "<br>" +
                                  "Type: " + method_type + "<br>" +
                                  "Memory: %{y:.2f} MB (converted from KB)<extra></extra>"  # Clarify units
                ))
        
        # Update layout
        fig_mem.update_layout(
            title=dict(
                text='Memory Usage Comparison - GTM Methods (Memory in MB, converted from KB)',
                font=dict(size=20, family='Arial, sans-serif', color='#333333'),
                x=0.5
            ),
            xaxis=dict(
                title='Method',
                titlefont=dict(size=16),
                tickangle=45,
                tickfont=dict(size=12),
                gridcolor='rgba(230,230,230,0.5)',
                showgrid=False,
                automargin=True
            ),
            yaxis=dict(
                title='Memory Usage (MB, converted from KB)',
                titlefont=dict(size=16),
                tickfont=dict(size=12),
                gridcolor='rgba(230,230,230,0.5)',
                showgrid=True,
                zeroline=True,
                zerolinecolor='rgba(0,0,0,0.2)',
                zerolinewidth=1
            ),
            barmode='group',
            bargap=0.15,
            bargroupgap=0.1,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='rgba(0,0,0,0.2)',
                borderwidth=1,
                font=dict(size=12)
            ),
            plot_bgcolor='rgba(250,250,250,0.9)',
            width=1000,
            height=700,
            margin=dict(l=60, r=60, t=100, b=120)
        )
        
        # Add value annotations
        for i, row in gtm_data.iterrows():
            fig_mem.add_annotation(
                x=row['Display_Name'],
                y=row['Memory_MB'],
                text=f"{row['Memory_MB']:.2f} MB",
                showarrow=False,
                yshift=10,
                font=dict(size=10, color="#333333"),
                bgcolor='rgba(255,255,255,0.7)',
                bordercolor='rgba(0,0,0,0.1)',
                borderwidth=1,
                borderpad=3,
                opacity=0.8
            )
        
        # Save memory comparison
        fig_mem.write_html(f"{output_dir}/improved_memory_comparison.html")
        fig_mem.write_image(f"{output_dir}/improved_memory_comparison.png", scale=3)

def create_normalized_metrics_comparison(df, output_dir, model_colors):
    """Create a plot comparing all normalized metrics (RF, FN, FP) for both models"""
    # Filter for GTM methods
    gtm_methods = [
        'Original GTM (FastTree guide + FastTree subset)',
        'Hybrid GTM 1 (FastTree guide + NJ-LogDet subset)',
        'Hybrid GTM 2 (NJ-LogDet guide + FastTree subset)',
        'Hybrid GTM 3 (NJ-LogDet guide + NJ-LogDet subset)'
    ]
    
    # Use shorter display names
    method_display_names = {
        'Original GTM (FastTree guide + FastTree subset)': 'Original GTM',
        'Hybrid GTM 1 (FastTree guide + NJ-LogDet subset)': 'Hybrid GTM 1',
        'Hybrid GTM 2 (NJ-LogDet guide + FastTree subset)': 'Hybrid GTM 2',
        'Hybrid GTM 3 (NJ-LogDet guide + NJ-LogDet subset)': 'Hybrid GTM 3'
    }
    
    gtm_df = df[df['Method'].isin(gtm_methods)].copy()
    gtm_df['Display_Name'] = gtm_df['Method'].map(method_display_names)
    
    # Normalize FN and FP values
    models = gtm_df['Model'].unique()
    
    # Process and normalize all metrics
    for model in models:
        model_data = gtm_df[gtm_df['Model'] == model]
        
        # Get max values for normalization
        max_fn = model_data['FN_mean'].max()
        max_fp = model_data['FP_mean'].max()
        
        # Normalize values if max > 0
        if max_fn > 0:
            gtm_df.loc[gtm_df['Model'] == model, 'FN_norm'] = gtm_df.loc[gtm_df['Model'] == model, 'FN_mean'] / max_fn
        else:
            gtm_df.loc[gtm_df['Model'] == model, 'FN_norm'] = gtm_df.loc[gtm_df['Model'] == model, 'FN_mean']
            
        if max_fp > 0:
            gtm_df.loc[gtm_df['Model'] == model, 'FP_norm'] = gtm_df.loc[gtm_df['Model'] == model, 'FP_mean'] / max_fp
        else:
            gtm_df.loc[gtm_df['Model'] == model, 'FP_norm'] = gtm_df.loc[gtm_df['Model'] == model, 'FP_mean']
    
    # Prepare data for plotting
    agg_dict = {
        'RF_norm': 'mean',
        'FN_norm': 'mean',
        'FP_norm': 'mean',
        'Display_Name': 'first'
    }
    
    grouped_df = gtm_df.groupby(['Model', 'Method']).agg(agg_dict).reset_index()
    
    # Create a subplot for each model
    fig = make_subplots(
        rows=2, 
        cols=1,
        subplot_titles=[f"{model} - Normalized Metrics Comparison" for model in models],
        vertical_spacing=0.15
    )
    
    # Define metrics and their colors
    metrics = {
        'RF_norm': {'name': 'RF Distance', 'color': 'rgba(31, 119, 180, 0.8)'},   # Blue
        'FN_norm': {'name': 'FN Rate', 'color': 'rgba(255, 127, 14, 0.8)'},       # Orange
        'FP_norm': {'name': 'FP Rate', 'color': 'rgba(44, 160, 44, 0.8)'}         # Green
    }
    
    # Add traces for each model and metric
    for i, model in enumerate(models):
        model_data = grouped_df[grouped_df['Model'] == model].sort_values('Method')
        
        # Create a DataFrame in the right format for plotting
        plot_data = []
        for _, row in model_data.iterrows():
            for metric, meta in metrics.items():
                plot_data.append({
                    'Method': row['Display_Name'],
                    'Metric': meta['name'],
                    'Value': row[metric],
                    'Color': meta['color']
                })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Add one trace per method, with all metrics as a grouped bar
        for j, method in enumerate(plot_df['Method'].unique()):
            method_data = plot_df[plot_df['Method'] == method]
            
            for k, (_, metric_row) in enumerate(method_data.iterrows()):
                fig.add_trace(
                    go.Bar(
                        x=[metric_row['Metric']],
                        y=[metric_row['Value']],
                        name=f"{method} - {metric_row['Metric']}",
                        marker=dict(
                            color=metric_row['Color'],
                            line=dict(color='rgba(0,0,0,0.3)', width=1)
                        ),
                        showlegend=True if i == 0 else False,
                        legendgroup=f"{method} - {metric_row['Metric']}",
                        hovertemplate=f"{method}<br>{metric_row['Metric']}: %{{y:.4f}}<extra></extra>",
                        offsetgroup=method
                    ),
                    row=i+1, 
                    col=1
                )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='Normalized Metrics Comparison Across Models and GTM Methods',
            font=dict(size=22, family='Arial, sans-serif', color='#333333'),
            x=0.5
        ),
        barmode='group',
        bargap=0.15,
        bargroupgap=0.1,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1,
            font=dict(size=10),
            # Group by method
            traceorder='grouped'
        ),
        plot_bgcolor='rgba(250,250,250,0.9)',
        width=1000,
        height=900,
        margin=dict(l=60, r=60, t=100, b=150)
    )
    
    # Update axes
    for i in range(len(models)):
        fig.update_xaxes(
            title=dict(
                text='Metric',
                font=dict(size=14)
            ),
            categoryorder='array',
            categoryarray=['RF Distance', 'FN Rate', 'FP Rate'],
            tickfont=dict(size=12),
            row=i+1, 
            col=1
        )
        
        fig.update_yaxes(
            title=dict(
                text='Normalized Value',
                font=dict(size=14)
            ),
            range=[0, 1.1],
            tickformat='.2f',
            tickfont=dict(size=12),
            gridcolor='rgba(230,230,230,0.8)',
            row=i+1, 
            col=1
        )
    
    # Add value annotations
    for i, model in enumerate(models):
        model_data = grouped_df[grouped_df['Model'] == model]
        
        # Create plot data
        plot_data = []
        for _, row in model_data.iterrows():
            for metric in metrics.keys():
                plot_data.append({
                    'Method': row['Display_Name'],
                    'Metric': metrics[metric]['name'],
                    'Value': row[metric]
                })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Add annotations
        for _, row in plot_df.iterrows():
            # Calculate position (offsetting for grouped bars)
            method_idx = list(plot_df['Method'].unique()).index(row['Method'])
            metric_idx = list(plot_df['Metric'].unique()).index(row['Metric'])
            
            # Get x position based on metric
            if row['Metric'] == 'RF Distance':
                x_pos = 0
            elif row['Metric'] == 'FN Rate':
                x_pos = 1
            else:  # FP Rate
                x_pos = 2
            
            # Add annotation
            fig.add_annotation(
                x=row['Metric'],
                y=row['Value'],
                text=f"{row['Value']:.3f}",
                showarrow=False,
                font=dict(size=9, color='black'),
                bgcolor='rgba(255,255,255,0.7)',
                bordercolor='rgba(0,0,0,0.1)',
                borderwidth=1,
                borderpad=2,
                opacity=0.9,
                yshift=10,
                xshift=(method_idx - 1.5) * 15,  # Offset based on method index
                row=i+1,
                col=1
            )
    
    # Save plot
    fig.write_html(f"{output_dir}/normalized_metrics_comparison.html")
    fig.write_image(f"{output_dir}/normalized_metrics_comparison.png", scale=3)

def create_gtm_comparison_plots(df, output_dir, model_colors):
    """Create plots specifically comparing different GTM approaches"""
    # Filter for only GTM methods
    gtm_methods = [
        'Original GTM (FastTree guide + FastTree subset)',
        'Hybrid GTM 1 (FastTree guide + NJ-LogDet subset)',
        'Hybrid GTM 2 (NJ-LogDet guide + FastTree subset)',
        'Hybrid GTM 3 (NJ-LogDet guide + NJ-LogDet subset)'
    ]
    
    # Use shorter display names for better visualization
    method_display_names = {
        'Original GTM (FastTree guide + FastTree subset)': 'Original GTM',
        'Hybrid GTM 1 (FastTree guide + NJ-LogDet subset)': 'Hybrid GTM 1',
        'Hybrid GTM 2 (NJ-LogDet guide + FastTree subset)': 'Hybrid GTM 2',
        'Hybrid GTM 3 (NJ-LogDet guide + NJ-LogDet subset)': 'Hybrid GTM 3'
    }
    
    gtm_df = df[df['Method'].isin(gtm_methods)].copy()
    
    if gtm_df.empty:
        print("No GTM methods found in the data.")
        return
    
    # Add display name column
    gtm_df['Display_Name'] = gtm_df['Method'].map(method_display_names)
    
    # Get unique models
    models = gtm_df['Model'].unique()
    
    # Prepare aggregated data
    agg_dict = {}
    for col in gtm_df.columns:
        if col not in ['Model', 'Method', 'Display_Name']:
            agg_dict[col] = 'mean' if pd.api.types.is_numeric_dtype(gtm_df[col]) else 'first'
    
    agg_dict['Display_Name'] = 'first'  # Make sure Display_Name is preserved
    grouped_df = gtm_df.groupby(['Model', 'Method']).agg(agg_dict).reset_index()
    
    # 1. Normalized RF Distance Comparison
    fig = go.Figure()
    for model in models:
        model_data = grouped_df[grouped_df['Model'] == model].sort_values('Method')
        fig.add_trace(go.Bar(
            x=model_data['Display_Name'], 
            y=model_data['RF_norm'],
            name=model,
            error_y=dict(
                type='data',
                array=model_data['RF_norm_std'],
                visible=True,
                color='rgba(0,0,0,0.3)',
                thickness=1.5,
                width=4
            ),
            marker=dict(
                color=model_colors.get(model, '#1f77b4'),
                line=dict(color='rgba(0,0,0,0.3)', width=1)
            ),
            hovertemplate='<b>%{x}</b><br>Model: ' + str(model) + 
                          '<br>Normalized RF: %{y:.4f} ± %{error_y.array:.4f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(text='Normalized RF Distance Comparison Among GTM Methods', x=0.5, font=dict(size=18)),
        xaxis=dict(title='GTM Method', tickangle=30, automargin=True),
        yaxis=dict(title='Normalized RF Distance', gridcolor='rgba(230,230,230,0.8)'),
        barmode='group',
        plot_bgcolor='rgba(245,245,245,0.5)',
        width=1000, height=600,
        margin=dict(l=20,r=20,t=60,b=80),
        font=dict(family="Arial, sans-serif", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    
    # Add value annotations
    for i, model in enumerate(models):
        model_data = grouped_df[grouped_df['Model'] == model].sort_values('Method')
        for j, (_, row) in enumerate(model_data.iterrows()):
            fig.add_annotation(
                x=row['Display_Name'],
                y=row['RF_norm'],
                text=f"{row['RF_norm']:.4f}",
                showarrow=False,
                yshift=10 + (i * 15),  # Offset for each model
                font=dict(size=10, color='black'),
                bgcolor='rgba(255,255,255,0.7)',
                bordercolor='rgba(0,0,0,0.1)',
                borderwidth=1,
                borderpad=1
            )
    
    fig.write_html(f"{output_dir}/gtm_rf_comparison.html")
    fig.write_image(f"{output_dir}/gtm_rf_comparison.png", scale=3)
    
    # 2. FN Rate Comparison
    fig = go.Figure()
    for model in models:
        model_data = grouped_df[grouped_df['Model'] == model].sort_values('Method')
        fig.add_trace(go.Bar(
            x=model_data['Display_Name'], 
            y=model_data['FN_mean'],
            name=model,
            error_y=dict(
                type='data',
                array=model_data['FN_std'],
                visible=True,
                color='rgba(0,0,0,0.3)',
                thickness=1.5,
                width=4
            ),
            marker=dict(
                color=model_colors.get(model, '#1f77b4'),
                line=dict(color='rgba(0,0,0,0.3)', width=1)
            ),
            hovertemplate='<b>%{x}</b><br>Model: ' + str(model) + 
                          '<br>FN Rate: %{y:.4f} ± %{error_y.array:.4f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(text='False Negative Rate Comparison Among GTM Methods', x=0.5, font=dict(size=18)),
        xaxis=dict(title='GTM Method', tickangle=30, automargin=True),
        yaxis=dict(title='False Negative Rate', gridcolor='rgba(230,230,230,0.8)'),
        barmode='group',
        plot_bgcolor='rgba(245,245,245,0.5)',
        width=1000, height=600,
        margin=dict(l=20,r=20,t=60,b=80),
        font=dict(family="Arial, sans-serif", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    
    # Add value annotations
    for i, model in enumerate(models):
        model_data = grouped_df[grouped_df['Model'] == model].sort_values('Method')
        for j, (_, row) in enumerate(model_data.iterrows()):
            fig.add_annotation(
                x=row['Display_Name'],
                y=row['FN_mean'],
                text=f"{row['FN_mean']:.4f}",
                showarrow=False,
                yshift=10 + (i * 15),  # Offset for each model
                font=dict(size=10, color='black'),
                bgcolor='rgba(255,255,255,0.7)',
                bordercolor='rgba(0,0,0,0.1)',
                borderwidth=1,
                borderpad=1
            )
    
    fig.write_html(f"{output_dir}/gtm_fn_comparison.html")
    fig.write_image(f"{output_dir}/gtm_fn_comparison.png", scale=3)
    
    # 3. FP Rate Comparison
    fig = go.Figure()
    for model in models:
        model_data = grouped_df[grouped_df['Model'] == model].sort_values('Method')
        fig.add_trace(go.Bar(
            x=model_data['Display_Name'], 
            y=model_data['FP_mean'],
            name=model,
            error_y=dict(
                type='data',
                array=model_data['FP_std'],
                visible=True,
                color='rgba(0,0,0,0.3)',
                thickness=1.5,
                width=4
            ),
            marker=dict(
                color=model_colors.get(model, '#1f77b4'),
                line=dict(color='rgba(0,0,0,0.3)', width=1)
            ),
            hovertemplate='<b>%{x}</b><br>Model: ' + str(model) + 
                          '<br>FP Rate: %{y:.4f} ± %{error_y.array:.4f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(text='False Positive Rate Comparison Among GTM Methods', x=0.5, font=dict(size=18)),
        xaxis=dict(title='GTM Method', tickangle=30, automargin=True),
        yaxis=dict(title='False Positive Rate', gridcolor='rgba(230,230,230,0.8)'),
        barmode='group',
        plot_bgcolor='rgba(245,245,245,0.5)',
        width=1000, height=600,
        margin=dict(l=20,r=20,t=60,b=80),
        font=dict(family="Arial, sans-serif", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    
    # Add value annotations
    for i, model in enumerate(models):
        model_data = grouped_df[grouped_df['Model'] == model].sort_values('Method')
        for j, (_, row) in enumerate(model_data.iterrows()):
            fig.add_annotation(
                x=row['Display_Name'],
                y=row['FP_mean'],
                text=f"{row['FP_mean']:.4f}",
                showarrow=False,
                yshift=10 + (i * 15),  # Offset for each model
                font=dict(size=10, color='black'),
                bgcolor='rgba(255,255,255,0.7)',
                bordercolor='rgba(0,0,0,0.1)',
                borderwidth=1,
                borderpad=1
            )
    
    fig.write_html(f"{output_dir}/gtm_fp_comparison.html")
    fig.write_image(f"{output_dir}/gtm_fp_comparison.png", scale=3)
    
    # 4. Combined matrix view of all GTM methods
    # Process by metric type
    fig = make_subplots(
        rows=len(models), cols=3,
        subplot_titles=[
            f"{model} - Normalized RF Distance" for model in models
        ] + [
            f"{model} - False Negative Rate" for model in models
        ] + [
            f"{model} - False Positive Rate" for model in models
        ],
        shared_xaxes='all',
        vertical_spacing=0.1,
        horizontal_spacing=0.05
    )
    
    # Use custom colors for different metrics
    metric_colors = {
        'RF': 'rgba(31, 119, 180, 0.8)',  # Blue
        'FN': 'rgba(255, 127, 14, 0.8)',  # Orange
        'FP': 'rgba(44, 160, 44, 0.8)',   # Green
    }
    
    # Add traces for each model and metric
    for i, model in enumerate(models):
        md = grouped_df[grouped_df['Model']==model].sort_values('Display_Name')
        
        # RF column
        fig.add_trace(go.Bar(
            x=md['RF_norm'], y=md['Display_Name'], 
            name=f"{model} - RF",
            orientation='h',
            error_x=dict(
                type='data',
                array=md['RF_norm_std'],
                visible=True,
                color='rgba(0,0,0,0.3)',
                thickness=1.5,
                width=4
            ),
            marker=dict(color=metric_colors['RF']),
            hovertemplate='<b>%{y}</b><br>RF: %{x:.4f} ± %{error_x.array:.4f}<extra></extra>',
            showlegend=(i==0)
        ), row=i+1, col=1)
        
        # Add RF values as text
        for j, (_, row) in enumerate(md.iterrows()):
            fig.add_annotation(
                x=row['RF_norm'] + 0.01,
                y=row['Display_Name'],
                text=f"{row['RF_norm']:.4f}",
                showarrow=False,
                font=dict(size=9, color='black'),
                xanchor='left',
                row=i+1, col=1
            )
        
        # FN column
        fig.add_trace(go.Bar(
            x=md['FN_mean'], y=md['Display_Name'], 
            name=f"{model} - FN",
            orientation='h',
            error_x=dict(
                type='data',
                array=md['FN_std'],
                visible=True,
                color='rgba(0,0,0,0.3)',
                thickness=1.5,
                width=4
            ),
            marker=dict(color=metric_colors['FN']),
            hovertemplate='<b>%{y}</b><br>FN: %{x:.4f} ± %{error_x.array:.4f}<extra></extra>',
            showlegend=(i==0)
        ), row=i+1, col=2)
        
        # Add FN values as text
        for j, (_, row) in enumerate(md.iterrows()):
            fig.add_annotation(
                x=row['FN_mean'] + 0.01,
                y=row['Display_Name'],
                text=f"{row['FN_mean']:.4f}",
                showarrow=False,
                font=dict(size=9, color='black'),
                xanchor='left',
                row=i+1, col=2
            )
        
        # FP column
        fig.add_trace(go.Bar(
            x=md['FP_mean'], y=md['Display_Name'], 
            name=f"{model} - FP",
            orientation='h',
            error_x=dict(
                type='data',
                array=md['FP_std'],
                visible=True,
                color='rgba(0,0,0,0.3)',
                thickness=1.5,
                width=4
            ),
            marker=dict(color=metric_colors['FP']),
            hovertemplate='<b>%{y}</b><br>FP: %{x:.4f} ± %{error_x.array:.4f}<extra></extra>',
            showlegend=(i==0)
        ), row=i+1, col=3)
        
        # Add FP values as text
        for j, (_, row) in enumerate(md.iterrows()):
            fig.add_annotation(
                x=row['FP_mean'] + 0.01,
                y=row['Display_Name'],
                text=f"{row['FP_mean']:.4f}",
                showarrow=False,
                font=dict(size=9, color='black'),
                xanchor='left',
                row=i+1, col=3
            )
    
    # Update layout
    fig.update_layout(
        title=dict(text='GTM Methods Comprehensive Comparison Matrix', x=0.5, font=dict(size=20)),
        plot_bgcolor='rgba(245,245,245,0.5)',
        width=1200, 
        height=200 * len(models),
        margin=dict(l=20,r=20,t=80,b=60),
        font=dict(family="Arial, sans-serif", size=12),
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=-0.1, 
            xanchor="center", 
            x=0.5,
            title=dict(text='Metrics')
        )
    )
    
    # Update x-axis titles
    for i in range(len(models)):
        fig.update_xaxes(title_text="Normalized RF", row=i+1, col=1)
        fig.update_xaxes(title_text="FN Rate", row=i+1, col=2)
        fig.update_xaxes(title_text="FP Rate", row=i+1, col=3)
    
    # Update grid and axis lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(230,230,230,0.8)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(230,230,230,0.8)')
    
    fig.write_html(f"{output_dir}/gtm_methods_matrix.html")
    fig.write_image(f"{output_dir}/gtm_methods_matrix.png", scale=3)
    
    # 5. Alternative GTM matrix (all models on one plot, with metrics as columns)
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Normalized RF", "False Negative Rate", "False Positive Rate"),
        shared_yaxes=True
    )
    
    # Get the shortest method names for display
    short_methods = [m.split(' ')[0] + ' ' + m.split(' ')[1] for m in gtm_methods]
    method_short_map = dict(zip(gtm_methods, short_methods))
    
    # Create combined dataframe with all relevant data
    plot_data = []
    for model in models:
        for method in gtm_methods:
            method_data = grouped_df[(grouped_df['Model'] == model) & (grouped_df['Method'] == method)]
            if not method_data.empty:
                plot_data.append({
                    'Model': model,
                    'Method': method,
                    'Short_Method': method_display_names[method],
                    'RF': method_data['RF_norm'].values[0],
                    'RF_std': method_data['RF_norm_std'].values[0],
                    'FN': method_data['FN_mean'].values[0],
                    'FN_std': method_data['FN_std'].values[0],
                    'FP': method_data['FP_mean'].values[0],
                    'FP_std': method_data['FP_std'].values[0]
                })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Add bars for each metric
    for model in models:
        model_df = plot_df[plot_df['Model'] == model]
        
        # RF
        fig.add_trace(go.Bar(
            x=model_df['RF'], y=model_df['Short_Method'],
            name=model,
            orientation='h',
            error_x=dict(
                type='data',
                array=model_df['RF_std'],
                visible=True,
                color='rgba(0,0,0,0.3)',
                thickness=1.5,
                width=4
            ),
            marker=dict(color=model_colors.get(model,'#1f77b4')),
            hovertemplate='<b>%{y}</b><br>Model: '+str(model)+'<br>RF: %{x:.4f} ± %{error_x.array:.4f}<extra></extra>'
        ), row=1, col=1)
        
        # FN
        fig.add_trace(go.Bar(
            x=model_df['FN'], y=model_df['Short_Method'],
            name=model,
            orientation='h',
            error_x=dict(
                type='data',
                array=model_df['FN_std'],
                visible=True,
                color='rgba(0,0,0,0.3)',
                thickness=1.5,
                width=4
            ),
            marker=dict(color=model_colors.get(model,'#1f77b4')),
            hovertemplate='<b>%{y}</b><br>Model: '+str(model)+'<br>FN: %{x:.4f} ± %{error_x.array:.4f}<extra></extra>',
            showlegend=False
        ), row=1, col=2)
        
        # FP
        fig.add_trace(go.Bar(
            x=model_df['FP'], y=model_df['Short_Method'],
            name=model,
            orientation='h',
            error_x=dict(
                type='data',
                array=model_df['FP_std'],
                visible=True,
                color='rgba(0,0,0,0.3)',
                thickness=1.5,
                width=4
            ),
            marker=dict(color=model_colors.get(model,'#1f77b4')),
            hovertemplate='<b>%{y}</b><br>Model: '+str(model)+'<br>FP: %{x:.4f} ± %{error_x.array:.4f}<extra></extra>',
            showlegend=False
        ), row=1, col=3)
    
    # Add value labels
    for model in models:
        model_df = plot_df[plot_df['Model'] == model]
        
        # Add RF values
        for i, row in model_df.iterrows():
            fig.add_annotation(
                x=row['RF'] + max(plot_df['RF'])*0.03,  # Position text right of the bar
                y=row['Short_Method'],
                text=f"{row['RF']:.4f}",
                showarrow=False,
                font=dict(size=9, color='black'),
                xanchor='left',
                yshift=(-8 if model == models[0] else 8),  # Offset based on model
                row=1, col=1
            )
            
        # Add FN values
        for i, row in model_df.iterrows():
            fig.add_annotation(
                x=row['FN'] + max(plot_df['FN'])*0.03,
                y=row['Short_Method'],
                text=f"{row['FN']:.4f}",
                showarrow=False,
                font=dict(size=9, color='black'),
                xanchor='left',
                yshift=(-8 if model == models[0] else 8),
                row=1, col=2
            )
            
        # Add FP values
        for i, row in model_df.iterrows():
            fig.add_annotation(
                x=row['FP'] + max(plot_df['FP'])*0.03,
                y=row['Short_Method'],
                text=f"{row['FP']:.4f}",
                showarrow=False,
                font=dict(size=9, color='black'),
                xanchor='left',
                yshift=(-8 if model == models[0] else 8),
                row=1, col=3
            )
    
    fig.update_layout(
        title=dict(text='GTM Methods Performance Metrics', x=0.5, font=dict(size=20)),
        barmode='group',
        plot_bgcolor='rgba(245,245,245,0.5)',
        width=1200, height=600,
        margin=dict(l=20,r=20,t=80,b=60),
        font=dict(family="Arial, sans-serif", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.65)
    )
    
    # Update x-axis titles
    fig.update_xaxes(title_text="Normalized RF", row=1, col=1)
    fig.update_xaxes(title_text="FN Rate", row=1, col=2)
    fig.update_xaxes(title_text="FP Rate", row=1, col=3)
    
    # Update y-axis titles
    fig.update_yaxes(title_text="GTM Method", row=1, col=1)
    
    # Add grid lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(230,230,230,0.8)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(230,230,230,0.8)')
    
    fig.write_html(f"{output_dir}/gtm_methods_comparison.html")
    fig.write_image(f"{output_dir}/gtm_methods_comparison.png", scale=3)

def create_horizontal_bar_plots(df, output_dir, model_colors):
    """Create horizontal bar plots with methods on y-axis"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Aggregate per Model+Method
    agg_dict = {}
    for col in df.columns:
        if col not in ['Model', 'Method']:
            agg_dict[col] = 'mean' if pd.api.types.is_numeric_dtype(df[col]) else 'first'
    grouped_df = df.groupby(['Model', 'Method']).agg(agg_dict).reset_index()
    
    models  = grouped_df['Model'].unique()
    methods = grouped_df['Method'].unique()
    
    # 1) Per-model FN rate bars
    for model in models:
        model_data = grouped_df[grouped_df['Model']==model].sort_values('FN_mean')
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=model_data['FN_mean'],
            y=model_data['Method'],
            orientation='h',
            error_x=dict(array=model_data['FN_std'],
                         color='rgba(0,0,0,0.5)', thickness=1.5, width=4),
            marker=dict(color=model_colors.get(model,'#1f77b4'),
                        line=dict(color='rgba(0,0,0,0.5)', width=1)),
            hovertemplate='<b>%{y}</b><br>FN Rate: %{x:.4f} ± %{error_x.array:.4f}<extra></extra>'
        ))
        fig.update_layout(
            title=dict(text=f'False Negative Rates for {model}', x=0.5, font=dict(size=18)),
            xaxis=dict(title='False Negative Rate', gridcolor='rgba(230,230,230,0.8)'),
            yaxis=dict(title='Method', automargin=True),
            plot_bgcolor='rgba(245,245,245,0.5)',
            width=900, height=600, margin=dict(l=20,r=20,t=60,b=60),
            font=dict(family="Arial, sans-serif", size=12),
            showlegend=False
        )
        for _, row in model_data.iterrows():
            fig.add_annotation(
                x=row['FN_mean']+0.02, y=row['Method'],
                text=f"{row['FN_mean']:.4f}", showarrow=False,
                font=dict(size=10), xanchor='left'
            )
        fig.write_html(f"{output_dir}/fn_rates_{model}_horizontal.html")
        fig.write_image(f"{output_dir}/fn_rates_{model}_horizontal.png", scale=3)
    
    # 2) Combined FN rate comparison
    fig = go.Figure()
    avg_fn = grouped_df.groupby('Method')['FN_mean'].mean().reset_index().sort_values('FN_mean')
    sorted_methods = avg_fn['Method'].tolist()
    for model in models:
        md = (grouped_df[grouped_df['Model']==model]
              .reindex(columns=['Method','FN_mean','FN_std'])
              .merge(pd.DataFrame({'Method':sorted_methods}),
                     on='Method', how='right').fillna(0))
        fig.add_trace(go.Bar(
            x=md['FN_mean'], y=md['Method'], name=model, orientation='h',
            error_x=dict(array=md['FN_std'], color='rgba(0,0,0,0.3)',
                         thickness=1.5, width=4),
            marker=dict(color=model_colors.get(model,'#1f77b4'),
                        line=dict(color='rgba(0,0,0,0.3)', width=1)),
            hovertemplate='<b>%{y}</b><br>Model: '+str(model)+
                          '<br>FN Rate: %{x:.4f} ± %{error_x.array:.4f}<extra></extra>'
        ))
    fig.update_layout(
        title=dict(text='False Negative Rates Comparison', x=0.5, font=dict(size=18)),
        xaxis=dict(title='False Negative Rate', gridcolor='rgba(230,230,230,0.8)'),
        yaxis=dict(title='Method', automargin=True),
        barmode='group', plot_bgcolor='rgba(245,245,245,0.5)',
        width=1000, height=700, margin=dict(l=20,r=20,t=60,b=60),
        font=dict(family="Arial, sans-serif", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=1.5,
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='rgba(0,0,0,0.1)', borderwidth=1)
    )
    fig.write_html(f"{output_dir}/all_methods_fn_comparison_horizontal.html")
    fig.write_image(f"{output_dir}/all_methods_fn_comparison_horizontal.png", scale=3)
    
    # 3) Combined FP rate comparison
    fig = go.Figure()
    avg_fp = grouped_df.groupby('Method')['FP_mean'].mean().reset_index().sort_values('FP_mean')
    sorted_methods = avg_fp['Method'].tolist()
    for model in models:
        md = (grouped_df[grouped_df['Model']==model]
              .reindex(columns=['Method','FP_mean','FP_std'])
              .merge(pd.DataFrame({'Method':sorted_methods}),
                     on='Method', how='right').fillna(0))
        fig.add_trace(go.Bar(
            x=md['FP_mean'], y=md['Method'], name=model, orientation='h',
            error_x=dict(array=md['FP_std'], color='rgba(0,0,0,0.3)',
                         thickness=1.5, width=4),
            marker=dict(color=model_colors.get(model,'#1f77b4'),
                        line=dict(color='rgba(0,0,0,0.3)', width=1)),
            hovertemplate='<b>%{y}</b><br>Model: '+str(model)+
                          '<br>FP Rate: %{x:.4f} ± %{error_x.array:.4f}<extra></extra>'
        ))
    fig.update_layout(
        title=dict(text='False Positive Rates Comparison', x=0.5, font=dict(size=18)),
        xaxis=dict(title='False Positive Rate', gridcolor='rgba(230,230,230,0.8)'),
        yaxis=dict(title='Method', automargin=True),
        barmode='group', plot_bgcolor='rgba(245,245,245,0.5)',
        width=1000, height=700, margin=dict(l=20,r=20,t=60,b=60),
        font=dict(family="Arial, sans-serif", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=1.5,
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='rgba(0,0,0,0.1)', borderwidth=1)
    )
    fig.write_html(f"{output_dir}/all_methods_fp_comparison_horizontal.html")
    fig.write_image(f"{output_dir}/all_methods_fp_comparison_horizontal.png", scale=3)
    
    # 4) Combined RF distance comparison (normalized)
    fig = go.Figure()
    avg_rf = grouped_df.groupby('Method')['RF_norm'].mean().reset_index().sort_values('RF_norm')
    sorted_methods = avg_rf['Method'].tolist()
    for model in models:
        md = (grouped_df[grouped_df['Model']==model]
              .reindex(columns=['Method','RF_norm','RF_norm_std'])
              .merge(pd.DataFrame({'Method':sorted_methods}),
                     on='Method', how='right').fillna(0))
        fig.add_trace(go.Bar(
            x=md['RF_norm'], y=md['Method'], name=model, orientation='h',
            error_x=dict(array=md['RF_norm_std'], color='rgba(0,0,0,0.3)',
                         thickness=1.5, width=4),
            marker=dict(color=model_colors.get(model,'#1f77b4'),
                        line=dict(color='rgba(0,0,0,0.3)', width=1)),
            hovertemplate='<b>%{y}</b><br>Model: '+str(model)+
                          '<br>Normalized RF Distance: %{x:.4f} ± %{error_x.array:.4f}<extra></extra>'
        ))
    fig.update_layout(
        title=dict(text='Normalized RF Distances Comparison', x=0.5, font=dict(size=18)),
        xaxis=dict(title='Normalized RF Distance', gridcolor='rgba(230,230,230,0.8)'),
        yaxis=dict(title='Method', automargin=True),
        barmode='group', plot_bgcolor='rgba(245,245,245,0.5)',
        width=1000, height=700, margin=dict(l=20,r=20,t=60,b=60),
        font=dict(family="Arial, sans-serif", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=1.5,
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='rgba(0,0,0,0.1)', borderwidth=1)
    )
    fig.write_html(f"{output_dir}/all_methods_rf_comparison_horizontal_normalized.html")
    fig.write_image(f"{output_dir}/all_methods_rf_comparison_horizontal_normalized.png", scale=3)

def create_complexity_scatter_plot(df, output_dir, model_colors):
    """Create interactive scatter plot for performance vs complexity"""
    # Define method complexity
    method_complexity = {
        'FastTree (GTR)': 5, 'FastTree (JC)': 4.5, 'NJ (JC)': 3.5,
        'NJ (LogDet)': 3, 'NJ (p-distance)': 2.5, 'UPGMA (JC)': 3,
        'FastME (BME)': 4,
        'Original GTM (FastTree guide + FastTree subset)': 4.8,
        'Hybrid GTM 1 (FastTree guide + NJ-LogDet subset)': 4,
        'Hybrid GTM 2 (NJ-LogDet guide + FastTree subset)': 4,
        'Hybrid GTM 3 (NJ-LogDet guide + NJ-LogDet subset)': 2.8
    }
    df['Complexity'] = df['Method'].map(method_complexity).fillna(3)
    
    # Aggregate
    agg_dict = {
        'FN_mean': 'mean',
        'FP_mean': 'mean',
        'RF_mean': 'mean',
        'RF_norm': 'mean',
        'Complexity': 'first'
    }
    grouped_df = df.groupby(['Model', 'Method']).agg(agg_dict).reset_index()
    
    # Size by normalized RF
    size_scale = lambda rn: 25 * (1 - rn) + 10
    
    # Plot 1: Complexity on X, FN on Y
    fig = go.Figure()
    for _, row in grouped_df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['Complexity']], y=[row['FN_mean']],
            mode='markers',
            name=f"{row['Model']} – {row['Method']}",
            marker=dict(
                symbol='circle',
                size=size_scale(row['RF_norm']),
                color=model_colors.get(row['Model'],'#1f77b4'),
                line=dict(color='black', width=1)
            ),
            text=[row['Method']],
            hovertemplate=(
                '<b>%{text}</b><br>'
                'Model: '+str(row['Model'])+'<br>'
                'Complexity: %{x:.1f}<br>'
                'FN Rate: %{y:.4f}<br>'
                'RF (norm): '+f"{row['RF_norm']:.2f}"+'<extra></extra>'
            )
        ))
    fig.update_layout(
        title=dict(text='Performance vs. Computational Complexity', x=0.5, font=dict(size=20)),
        xaxis=dict(title='Computational Complexity', range=[2,5.5], gridcolor='rgba(230,230,230,0.8)'),
        yaxis=dict(title='False Negative Rate', gridcolor='rgba(230,230,230,0.8)'),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        plot_bgcolor='rgba(245,245,245,0.5)',
        width=1000, height=800,
        margin=dict(l=20,r=20,t=80,b=60),
        font=dict(family="Arial, sans-serif", size=12),
        hovermode='closest'
    )
    fig.write_html(f"{output_dir}/performance_vs_complexity_interactive.html")
    fig.write_image(f"{output_dir}/performance_vs_complexity.png", scale=3)
    
    # Plot 2: Complexity on Y, FN on X
    fig2 = go.Figure()
    for _, row in grouped_df.iterrows():
        fig2.add_trace(go.Scatter(
            y=[row['Complexity']], x=[row['FN_mean']],
            mode='markers',
            name=f"{row['Model']} – {row['Method']}",
            marker=dict(
                symbol='circle',
                size=size_scale(row['RF_norm']),
                color=model_colors.get(row['Model'],'#1f77b4'),
                line=dict(color='black', width=1)
            ),
            text=[row['Method']],
            hovertemplate=(
                '<b>%{text}</b><br>'
                'Model: '+str(row['Model'])+'<br>'
                'Complexity: %{y:.1f}<br>'
                'FN Rate: %{x:.4f}<br>'
                'RF (norm): '+f"{row['RF_norm']:.2f}"+'<extra></extra>'
            )
        ))
    fig2.update_layout(
        title=dict(text='Performance vs. Computational Complexity', x=0.5, font=dict(size=20)),
        yaxis=dict(title='Computational Complexity', range=[2,5.5], gridcolor='rgba(230,230,230,0.8)'),
        xaxis=dict(title='False Negative Rate', gridcolor='rgba(230,230,230,0.8)'),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        plot_bgcolor='rgba(245,245,245,0.5)',
        width=1000, height=800,
        margin=dict(l=20,r=20,t=80,b=60),
        font=dict(family="Arial, sans-serif", size=12),
        hovermode='closest'
    )
    fig2.write_html(f"{output_dir}/complexity_vs_performance_interactive.html")
    fig2.write_image(f"{output_dir}/complexity_vs_performance.png", scale=3)

def create_comparison_matrix(df, output_dir, model_colors):
    """Create a comparison matrix of methods and metrics"""
    try:
        # 1) Method-wise averages
        agg_dict = {'RF_mean':'mean','FN_mean':'mean','FP_mean':'mean'}
        grouped_df = df.groupby('Method').agg(agg_dict).reset_index()
        grouped_df = grouped_df.sort_values('FN_mean').fillna(0)
        
        # Normalize RF for heatmap
        max_rf = grouped_df['RF_mean'].max()
        if max_rf>0:
            normalized_rf = grouped_df['RF_mean']/max_rf
        else:
            normalized_rf = grouped_df['RF_mean']
        
        fig = make_subplots(rows=1, cols=3,
                            subplot_titles=("RF Distance","False Negative Rate","False Positive Rate"),
                            shared_yaxes=True)
        # RF heatmap
        fig.add_trace(go.Heatmap(
            z=normalized_rf.values.reshape(1,-1),
            y=['RF Distance'], x=grouped_df['Method'],
            colorscale='YlOrRd', showscale=False,
            text=grouped_df['RF_mean'].round(1).values.reshape(1,-1),
            texttemplate='%{text}', hovertemplate='<b>%{x}</b><br>RF: %{text}<extra></extra>',
            zmin=0, zmax=1
        ), row=1, col=1)
        # FN heatmap
        fig.add_trace(go.Heatmap(
            z=grouped_df['FN_mean'].values.reshape(1,-1),
            y=['FN Rate'], x=grouped_df['Method'],
            colorscale='YlOrRd', showscale=False,
            text=grouped_df['FN_mean'].round(4).values.reshape(1,-1),
            texttemplate='%{text}', hovertemplate='<b>%{x}</b><br>FN: %{text}<extra></extra>',
            zmin=0, zmax=1
        ), row=1, col=2)
        # FP heatmap
        fig.add_trace(go.Heatmap(
            z=grouped_df['FP_mean'].values.reshape(1,-1),
            y=['FP Rate'], x=grouped_df['Method'],
            colorscale='YlOrRd', showscale=True,
            text=grouped_df['FP_mean'].round(4).values.reshape(1,-1),
            texttemplate='%{text}', hovertemplate='<b>%{x}</b><br>FP: %{text}<extra></extra>',
            zmin=0, zmax=1
        ), row=1, col=3)
        fig.update_layout(
            title=dict(text='Method Performance Comparison Matrix', x=0.5, font=dict(size=20)),
            width=1200, height=300, margin=dict(l=20,r=20,t=80,b=150),
            font=dict(family="Arial, sans-serif", size=12),
            plot_bgcolor='rgba(245,245,245,0.5)'
        )
        fig.update_xaxes(tickangle=45, automargin=True)
        fig.write_html(f"{output_dir}/method_comparison_matrix.html")
        fig.write_image(f"{output_dir}/method_comparison_matrix.png", scale=3)
        
        # 2) Model-specific heatmaps
        agg_dict = {'RF_mean':'mean','FN_mean':'mean','FP_mean':'mean'}
        model_grouped = df.groupby(['Model','Method']).agg(agg_dict).reset_index()
        models = model_grouped['Model'].unique()
        methods = grouped_df['Method'].tolist()
        
        fig = make_subplots(
            rows=len(models), cols=3,
            subplot_titles=[f"{m} - RF" for m in models] +
                           [f"{m} - FN" for m in models] +
                           [f"{m} - FP" for m in models],
            shared_xaxes=True, vertical_spacing=0.05
        )
        for i, mdl in enumerate(models):
            md = model_grouped[model_grouped['Model']==mdl]
            md = pd.DataFrame({'Method':methods}).merge(md, on='Method', how='left').fillna(0)
            # normalize per-model RF
            mmax = md['RF_mean'].max()
            nrf = md['RF_mean']/mmax if mmax>0 else md['RF_mean']
            # RF
            fig.add_trace(go.Heatmap(
                z=nrf.values.reshape(1,-1),
                y=[f"{mdl} RF"], x=methods,
                colorscale='YlOrRd', showscale=(i==0),
                text=md['RF_mean'].round(1).values.reshape(1,-1),
                texttemplate='%{text}', hovertemplate='<b>%{x}</b><br>RF: %{text}<extra></extra>',
                zmin=0, zmax=1
            ), row=i+1, col=1)
            # FN
            fig.add_trace(go.Heatmap(
                z=md['FN_mean'].values.reshape(1,-1),
                y=[f"{mdl} FN"], x=methods,
                colorscale='YlOrRd', showscale=(i==0),
                text=md['FN_mean'].round(4).values.reshape(1,-1),
                texttemplate='%{text}', hovertemplate='<b>%{x}</b><br>FN: %{text}<extra></extra>',
                zmin=0, zmax=1
            ), row=i+1, col=2)
            # FP
            fig.add_trace(go.Heatmap(
                z=md['FP_mean'].values.reshape(1,-1),
                y=[f"{mdl} FP"], x=methods,
                colorscale='YlOrRd', showscale=(i==0),
                text=md['FP_mean'].round(4).values.reshape(1,-1),
                texttemplate='%{text}', hovertemplate='<b>%{x}</b><br>FP: %{text}<extra></extra>',
                zmin=0, zmax=1
            ), row=i+1, col=3)
        fig.update_layout(
            title=dict(text='Detailed Method Performance by Model', x=0.5, font=dict(size=20)),
            width=1200, height=300*len(models),
            margin=dict(l=20,r=20,t=80,b=150),
            font=dict(family="Arial, sans-serif", size=12),
            plot_bgcolor='rgba(245,245,245,0.5)'
        )
        fig.update_xaxes(tickangle=45, automargin=True)
        # Hide upper x-labels
        for r in range(1, len(models)):
            for c in range(1,4):
                fig.update_xaxes(showticklabels=False, row=r, col=c)
        fig.write_html(f"{output_dir}/detailed_method_comparison_matrix.html")
        fig.write_image(f"{output_dir}/detailed_method_comparison_matrix.png", scale=3)
        
        # 3) Comprehensive dashboard
        dashboard = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Performance vs. Complexity',
                'Robinson-Foulds Distances',
                'False Negative Rates',
                'False Positive Rates',
                'Method Comparison Matrix',
                ''
            ),
            specs=[
                [{"type":"scatter"},{"type":"bar"}],
                [{"type":"bar"},{"type":"bar"}],
                [{"type":"heatmap","colspan":2},{}]
            ],
            vertical_spacing=0.1, horizontal_spacing=0.05
        )
        # Complexity vs FN
        for _, row in grouped_df.iterrows():
            dashboard.add_trace(go.Scatter(
                x=[row['Complexity']], y=[row['FN_mean']],
                mode='markers+text', name=row['Model'],
                marker=dict(size=15,
                            color=model_colors.get(row['Model'],'#1f77b4'),
                            line=dict(color='black', width=1)),
                text=[row['Method'].split(' ')[0]],
                textposition="top center",
                hovertemplate='<b>%{text}</b><br>Complexity: %{x:.1f}<br>FN Rate: %{y:.4f}<extra></extra>'
            ), row=1, col=1)
        # RF bars
        for mdl in models:
            md = grouped_df[grouped_df['Model']==mdl]
            dashboard.add_trace(go.Bar(
                x=md['Method'], y=md['RF_mean'], name=mdl,
                marker=dict(color=model_colors.get(mdl,'#1f77b4')),
                hovertemplate='<b>%{x}</b><br>RF: %{y:.1f}<extra></extra>'
            ), row=1, col=2)
        # FN bars
        for mdl in models:
            md = grouped_df[grouped_df['Model']==mdl]
            dashboard.add_trace(go.Bar(
                x=md['Method'], y=md['FN_mean'], name=mdl,
                marker=dict(color=model_colors.get(mdl,'#1f77b4')),
                hovertemplate='<b>%{x}</b><br>FN: %{y:.4f}<extra></extra>',
                showlegend=False
            ), row=2, col=1)
        # FP bars
        for mdl in models:
            md = grouped_df[grouped_df['Model']==mdl]
            dashboard.add_trace(go.Bar(
                x=md['Method'], y=md['FP_mean'], name=mdl,
                marker=dict(color=model_colors.get(mdl,'#1f77b4')),
                hovertemplate='<b>%{x}</b><br>FP: %{y:.4f}<extra></extra>',
                showlegend=False
            ), row=2, col=2)
        # Matrix heatmap
        dashboard.add_trace(go.Heatmap(
            z=np.vstack([
                normalized_rf.values,
                grouped_df['FN_mean'].values,
                grouped_df['FP_mean'].values
            ]),
            y=grouped_df['Method'],
            x=['RF (norm)','FN Rate','FP Rate'],
            colorscale='YlOrRd', showscale=True,
            text=np.vstack([
                grouped_df['RF_mean'].round(1).values,
                grouped_df['FN_mean'].round(4).values,
                grouped_df['FP_mean'].round(4).values
            ]),
            texttemplate='%{text}',
            hovertemplate='<b>%{y}</b><br>%{x}: %{text}<extra></extra>'
        ), row=3, col=1)
        dashboard.update_layout(
            title=dict(text='Comprehensive Dashboard', x=0.5, font=dict(size=24)),
            width=1600, height=1200,
            margin=dict(l=20,r=20,t=100,b=20),
            barmode='group', hovermode='closest',
            plot_bgcolor='rgba(245,245,245,0.5)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="center", x=0.5)
        )
        dashboard.update_xaxes(tickangle=45, automargin=True)
        # Axis titles
        dashboard.update_xaxes(title_text="Complexity", row=1, col=1)
        dashboard.update_yaxes(title_text="FN Rate", row=1, col=1)
        dashboard.update_xaxes(title_text="Method", row=1, col=2)
        dashboard.update_yaxes(title_text="RF Distance", row=1, col=2)
        dashboard.update_xaxes(title_text="Method", row=2, col=1)
        dashboard.update_yaxes(title_text="FN Rate", row=2, col=1)
        dashboard.update_xaxes(title_text="Method", row=2, col=2)
        dashboard.update_yaxes(title_text="FP Rate", row=2, col=2)
        dashboard.update_xaxes(title_text="Metric", row=3, col=1)
        dashboard.update_yaxes(title_text="Method", row=3, col=1)
        dashboard.write_html(f"{output_dir}/comprehensive_dashboard.html")
        dashboard.write_image(f"{output_dir}/comprehensive_dashboard.png", scale=3)

    except Exception as e:
        print(f"Error in creating comparison matrix: {e}")
        # Simplified fallback...
        alt_fig = go.Figure()
        for method in df['Method'].unique():
            mdf = df[df['Method']==method]
            avg = mdf[['FN_mean','FP_mean']].mean()
            alt_fig.add_trace(go.Bar(
                y=[method], x=[avg['FN_mean']], name='FN',
                orientation='h', marker=dict(color='red'),
                hovertemplate='<b>%{y}</b><br>FN: %{x:.4f}<extra></extra>'
            ))
            alt_fig.add_trace(go.Bar(
                y=[method], x=[avg['FP_mean']], name='FP',
                orientation='h', marker=dict(color='blue'),
                hovertemplate='<b>%{y}</b><br>FP: %{x:.4f}<extra></extra>'
            ))
        alt_fig.update_layout(
            title='Simplified Comparison', barmode='group',
            width=1000, height=600,
            xaxis=dict(title='Rate'),
            yaxis=dict(title='Method'),
            legend=dict(orientation="h", y=1.1, x=0.5)
        )
        alt_fig.write_html(f"{output_dir}/simplified_comparison.html")
        alt_fig.write_image(f"{output_dir}/simplified_comparison.png", scale=3)

if __name__ == "__main__":
    main()



# import argparse
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# import plotly.express as px
# from plotly.subplots import make_subplots
# import os

# def parse_args():
#     """Parse command line arguments"""
#     parser = argparse.ArgumentParser(description="Create publication-quality comparison plots")
#     parser.add_argument("--baseline-summary", required=True, help="Baseline summary CSV file")
#     parser.add_argument("--gtm-summary", required=True, help="GTM summary CSV file")
#     parser.add_argument("--output-dir", required=True, help="Output directory for plots")
#     return parser.parse_args()

# def main():
#     """Main function to create beautiful interactive plots"""
#     args = parse_args()
    
#     # Define model colors
#     model_colors = {
#         '1000M1': '#1f77b4',  # Blue
#         '1000M4': '#ff7f0e',  # Orange
#     }

#     # Read summary files
#     print(f"Reading baseline summary: {args.baseline_summary}")
#     baseline_df = pd.read_csv(args.baseline_summary)
#     print(f"Reading GTM summary: {args.gtm_summary}")
#     gtm_df = pd.read_csv(args.gtm_summary)

#     # --- DROP 'Unnamed: 0' if present ---
#     baseline_df = baseline_df.loc[:, ~baseline_df.columns.str.startswith('Unnamed')]
#     gtm_df      = gtm_df.loc[:,      ~gtm_df.columns.str.startswith('Unnamed')]
    
#     # Print column information for debugging
#     print("Baseline DataFrame columns:", baseline_df.columns.tolist())
#     print("GTM DataFrame columns:", gtm_df.columns.tolist())
    
#     # Map column names to expected format
#     column_mapping = {
#         'RF': 'RF_mean',
#         'RF.1': 'RF_std',
#         'FN': 'FN_mean',
#         'FN.1': 'FN_std',
#         'FP': 'FP_mean',
#         'FP.1': 'FP_std'
#     }
    
#     # Rename columns in both DataFrames
#     baseline_df = baseline_df.rename(columns=column_mapping)
#     gtm_df = gtm_df.rename(columns=column_mapping)
    
#     # Ensure all numeric columns are properly converted to float
#     numeric_columns = ['RF_mean', 'RF_std', 'FN_mean', 'FN_std', 'FP_mean', 'FP_std']
    
#     for col in numeric_columns:
#         if col in baseline_df.columns:
#             # Convert to string first to handle any weird values
#             baseline_df[col] = baseline_df[col].astype(str)
#             # Remove any non-numeric characters
#             baseline_df[col] = baseline_df[col].str.replace(r'[^\d.]', '', regex=True)
#             # Convert to float, coercing errors to NaN
#             baseline_df[col] = pd.to_numeric(baseline_df[col], errors='coerce')
        
#         if col in gtm_df.columns:
#             # Convert to string first to handle any weird values
#             gtm_df[col] = gtm_df[col].astype(str)
#             # Remove any non-numeric characters
#             gtm_df[col] = gtm_df[col].str.replace(r'[^\d.]', '', regex=True)
#             # Convert to float, coercing errors to NaN
#             gtm_df[col] = pd.to_numeric(gtm_df[col], errors='coerce')
    
#     # Combine dataframes
#     combined_df = pd.concat([baseline_df, gtm_df], ignore_index=True)
    
#     # Fill NaN values with zeros
#     combined_df = combined_df.fillna(0)
    
#     # Save combined summary
#     combined_df.to_csv(f"{args.output_dir}/all_methods_summary.csv", index=False)
    
#     # Create comparison plots
#     create_horizontal_bar_plots(combined_df, args.output_dir, model_colors)
    
#     # Create scatter plot for performance vs complexity
#     create_complexity_scatter_plot(combined_df, args.output_dir, model_colors)
    
#     # Create comparison matrix
#     create_comparison_matrix(combined_df, args.output_dir, model_colors)
    
#     print("All plots created successfully!")

# def create_horizontal_bar_plots(df, output_dir, model_colors):
#     """Create horizontal bar plots with methods on y-axis"""
#     # Ensure output directory exists
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Group by Model and Method to get clean data for plotting
#     # Use 'first' aggregation for string/object columns to avoid errors
#     agg_dict = {}
#     for col in df.columns:
#         if col not in ['Model', 'Method']:
#             if pd.api.types.is_numeric_dtype(df[col]):
#                 agg_dict[col] = 'mean'
#             else:
#                 agg_dict[col] = 'first'
    
#     grouped_df = df.groupby(['Model', 'Method']).agg(agg_dict).reset_index()
    
#     # Get unique models and methods
#     models = grouped_df['Model'].unique()
#     methods = grouped_df['Method'].unique()
    
#     # Create horizontal bar plots for each model - FN rates
#     for model in models:
#         model_data = grouped_df[grouped_df['Model'] == model].sort_values('FN_mean')
        
#         fig = go.Figure()
        
#         fig.add_trace(go.Bar(
#             x=model_data['FN_mean'],
#             y=model_data['Method'],
#             orientation='h',
#             error_x=dict(
#                 type='data',
#                 array=model_data['FN_std'],
#                 visible=True,
#                 color='rgba(0,0,0,0.5)',
#                 thickness=1.5,
#                 width=4
#             ),
#             marker=dict(
#                 color=model_colors.get(model, '#1f77b4'),
#                 line=dict(color='rgba(0,0,0,0.5)', width=1)
#             ),
#             hovertemplate='<b>%{y}</b><br>FN Rate: %{x:.4f} ± %{error_x.array:.4f}<extra></extra>'
#         ))
        
#         fig.update_layout(
#             title=dict(
#                 text=f'False Negative Rates for {model} Model',
#                 font=dict(size=18),
#                 x=0.5
#             ),
#             xaxis=dict(
#                 title='False Negative Rate',
#                 title_font=dict(size=14),
#                 gridcolor='rgba(230,230,230,0.8)',
#                 zeroline=True,
#                 zerolinecolor='rgba(0,0,0,0.2)',
#                 zerolinewidth=1
#             ),
#             yaxis=dict(
#                 title='Method',
#                 title_font=dict(size=14),
#                 automargin=True
#             ),
#             plot_bgcolor='rgba(245,245,245,0.5)',
#             width=900,
#             height=600,
#             margin=dict(l=20, r=20, t=60, b=60),
#             font=dict(family="Arial, sans-serif", size=12),
#             showlegend=False,
#             hovermode='closest'
#         )
        
#         # Add annotations for the values
#         for i, row in model_data.iterrows():
#             fig.add_annotation(
#                 x=row['FN_mean'] + 0.02,
#                 y=row['Method'],
#                 text=f"{row['FN_mean']:.4f}",
#                 showarrow=False,
#                 font=dict(size=10),
#                 xanchor='left'
#             )
        
#         fig.write_html(f"{output_dir}/fn_rates_{model}_horizontal.html")
#         # Also save as static image for publications
#         fig.write_image(f"{output_dir}/fn_rates_{model}_horizontal.png", scale=3)
    
#     # Create combined horizontal bar plot for all methods and models - FN rates
#     fig = go.Figure()
    
#     # Sort methods by overall performance
#     avg_fn_by_method = grouped_df.groupby('Method')['FN_mean'].mean().reset_index()
#     avg_fn_by_method = avg_fn_by_method.sort_values('FN_mean')
#     sorted_methods = avg_fn_by_method['Method'].tolist()
    
#     for model in models:
#         model_data = grouped_df[grouped_df['Model'] == model]
        
#         # Create a method-indexed df for consistent ordering
#         model_data_ordered = pd.DataFrame({'Method': sorted_methods})
#         model_data_ordered = model_data_ordered.merge(model_data, on='Method', how='left')
#         model_data_ordered = model_data_ordered.fillna(0)
        
#         fig.add_trace(go.Bar(
#             x=model_data_ordered['FN_mean'],
#             y=model_data_ordered['Method'],
#             name=model,
#             orientation='h',
#             error_x=dict(
#                 type='data',
#                 array=model_data_ordered['FN_std'],
#                 visible=True,
#                 color='rgba(0,0,0,0.3)',
#                 thickness=1.5,
#                 width=4
#             ),
#             marker=dict(
#                 color=model_colors.get(model, '#1f77b4'),
#                 line=dict(color='rgba(0,0,0,0.3)', width=1)
#             ),
#             hovertemplate='<b>%{y}</b><br>Model: ' + str(model) + '<br>FN Rate: %{x:.4f} ± %{error_x.array:.4f}<extra></extra>',
#             offsetgroup=model
#         ))
    
#     fig.update_layout(
#         title=dict(
#             text='False Negative Rates Comparison Across Methods and Models',
#             font=dict(size=18),
#             x=0.5
#         ),
#         xaxis=dict(
#             title='False Negative Rate',
#             title_font=dict(size=14),
#             gridcolor='rgba(230,230,230,0.8)',
#             zeroline=True,
#             zerolinecolor='rgba(0,0,0,0.2)',
#             zerolinewidth=1
#         ),
#         yaxis=dict(
#             title='Method',
#             title_font=dict(size=14),
#             automargin=True
#         ),
#         barmode='group',
#         plot_bgcolor='rgba(245,245,245,0.5)',
#         width=1000,
#         height=700,
#         margin=dict(l=20, r=20, t=60, b=60),
#         font=dict(family="Arial, sans-serif", size=12),
#         legend=dict(
#             orientation="h",
#             yanchor="bottom",
#             y=1.02,
#             xanchor="center",
#             x=0.5,
#             bgcolor='rgba(255,255,255,0.8)',
#             bordercolor='rgba(0,0,0,0.1)',
#             borderwidth=1
#         ),
#         hovermode='closest'
#     )
    
#     fig.write_html(f"{output_dir}/all_methods_fn_comparison_horizontal.html")
#     fig.write_image(f"{output_dir}/all_methods_fn_comparison_horizontal.png", scale=3)
    
#     # Similar plots for FP rates
#     fig = go.Figure()
    
#     # Sort methods by overall performance
#     avg_fp_by_method = grouped_df.groupby('Method')['FP_mean'].mean().reset_index()
#     avg_fp_by_method = avg_fp_by_method.sort_values('FP_mean')
#     sorted_methods = avg_fp_by_method['Method'].tolist()
    
#     for model in models:
#         model_data = grouped_df[grouped_df['Model'] == model]
        
#         # Create a method-indexed df for consistent ordering
#         model_data_ordered = pd.DataFrame({'Method': sorted_methods})
#         model_data_ordered = model_data_ordered.merge(model_data, on='Method', how='left')
#         model_data_ordered = model_data_ordered.fillna(0)
        
#         fig.add_trace(go.Bar(
#             x=model_data_ordered['FP_mean'],
#             y=model_data_ordered['Method'],
#             name=model,
#             orientation='h',
#             error_x=dict(
#                 type='data',
#                 array=model_data_ordered['FP_std'],
#                 visible=True,
#                 color='rgba(0,0,0,0.3)',
#                 thickness=1.5,
#                 width=4
#             ),
#             marker=dict(
#                 color=model_colors.get(model, '#1f77b4'),
#                 line=dict(color='rgba(0,0,0,0.3)', width=1)
#             ),
#             hovertemplate='<b>%{y}</b><br>Model: ' + str(model) + '<br>FP Rate: %{x:.4f} ± %{error_x.array:.4f}<extra></extra>',
#             offsetgroup=model
#         ))
    
#     fig.update_layout(
#         title=dict(
#             text='False Positive Rates Comparison Across Methods and Models',
#             font=dict(size=18),
#             x=0.5
#         ),
#         xaxis=dict(
#             title='False Positive Rate',
#             title_font=dict(size=14),
#             gridcolor='rgba(230,230,230,0.8)',
#             zeroline=True,
#             zerolinecolor='rgba(0,0,0,0.2)',
#             zerolinewidth=1
#         ),
#         yaxis=dict(
#             title='Method',
#             title_font=dict(size=14),
#             automargin=True
#         ),
#         barmode='group',
#         plot_bgcolor='rgba(245,245,245,0.5)',
#         width=1000,
#         height=700,
#         margin=dict(l=20, r=20, t=60, b=60),
#         font=dict(family="Arial, sans-serif", size=12),
#         legend=dict(
#             orientation="h",
#             yanchor="bottom",
#             y=1.02,
#             xanchor="center",
#             x=0.5,
#             bgcolor='rgba(255,255,255,0.8)',
#             bordercolor='rgba(0,0,0,0.1)',
#             borderwidth=1
#         ),
#         hovermode='closest'
#     )
    
#     fig.write_html(f"{output_dir}/all_methods_fp_comparison_horizontal.html")
#     fig.write_image(f"{output_dir}/all_methods_fp_comparison_horizontal.png", scale=3)
    
#     # RF distances
#     fig = go.Figure()
    
#     # Sort methods by overall performance
#     avg_rf_by_method = grouped_df.groupby('Method')['RF_mean'].mean().reset_index()
#     avg_rf_by_method = avg_rf_by_method.sort_values('RF_mean')
#     sorted_methods = avg_rf_by_method['Method'].tolist()
    
#     for model in models:
#         model_data = grouped_df[grouped_df['Model'] == model]
        
#         # Create a method-indexed df for consistent ordering
#         model_data_ordered = pd.DataFrame({'Method': sorted_methods})
#         model_data_ordered = model_data_ordered.merge(model_data, on='Method', how='left')
#         model_data_ordered = model_data_ordered.fillna(0)
        
#         fig.add_trace(go.Bar(
#             x=model_data_ordered['RF_mean'],
#             y=model_data_ordered['Method'],
#             name=model,
#             orientation='h',
#             error_x=dict(
#                 type='data',
#                 array=model_data_ordered['RF_std'],
#                 visible=True,
#                 color='rgba(0,0,0,0.3)',
#                 thickness=1.5,
#                 width=4
#             ),
#             marker=dict(
#                 color=model_colors.get(model, '#1f77b4'),
#                 line=dict(color='rgba(0,0,0,0.3)', width=1)
#             ),
#             hovertemplate='<b>%{y}</b><br>Model: ' + str(model) + '<br>RF Distance: %{x:.1f} ± %{error_x.array:.1f}<extra></extra>',
#             offsetgroup=model
#         ))
    
#     fig.update_layout(
#         title=dict(
#             text='Robinson-Foulds Distances Comparison Across Methods and Models',
#             font=dict(size=18),
#             x=0.5
#         ),
#         xaxis=dict(
#             title='Robinson-Foulds Distance',
#             title_font=dict(size=14),
#             gridcolor='rgba(230,230,230,0.8)',
#             zeroline=True,
#             zerolinecolor='rgba(0,0,0,0.2)',
#             zerolinewidth=1
#         ),
#         yaxis=dict(
#             title='Method',
#             title_font=dict(size=14),
#             automargin=True
#         ),
#         barmode='group',
#         plot_bgcolor='rgba(245,245,245,0.5)',
#         width=1000,
#         height=700,
#         margin=dict(l=20, r=20, t=60, b=60),
#         font=dict(family="Arial, sans-serif", size=12),
#         legend=dict(
#             orientation="h",
#             yanchor="bottom",
#             y=1.02,
#             xanchor="center",
#             x=0.5,
#             bgcolor='rgba(255,255,255,0.8)',
#             bordercolor='rgba(0,0,0,0.1)',
#             borderwidth=1
#         ),
#         hovermode='closest'
#     )
    
#     fig.write_html(f"{output_dir}/all_methods_rf_comparison_horizontal.html")
#     fig.write_image(f"{output_dir}/all_methods_rf_comparison_horizontal.png", scale=3)

# def create_complexity_scatter_plot(df, output_dir, model_colors):
#     """Create interactive scatter plot for performance vs complexity"""
#     # Define method complexity (approximate runtime complexity)
#     method_complexity = {
#         'FastTree (GTR)': 5,
#         'FastTree (JC)': 4.5,
#         'NJ (JC)': 3.5,
#         'NJ (LogDet)': 3,
#         'NJ (p-distance)': 2.5,
#         'UPGMA (JC)': 3,
#         'FastME (BME)': 4,
#         'Original GTM (FastTree guide + FastTree subset)': 4.8,
#         'Hybrid GTM 1 (FastTree guide + NJ-LogDet subset)': 4,
#         'Hybrid GTM 2 (NJ-LogDet guide + FastTree subset)': 4,
#         'Hybrid GTM 3 (NJ-LogDet guide + NJ-LogDet subset)': 2.8
#     }
    
#     # Add complexity to dataframe with a safe default
#     df['Complexity'] = df['Method'].apply(lambda x: method_complexity.get(x, 3))
    
#     # Group by Model and Method with safe aggregation
#     agg_dict = {
#         'FN_mean': 'mean', 
#         'FP_mean': 'mean',
#         'RF_mean': 'mean',
#         'Complexity': 'first'
#     }
    
#     grouped_df = df.groupby(['Model', 'Method']).agg(agg_dict).reset_index()
    
#     # Set method symbols
#     method_symbols = {
#         'FastTree (GTR)': 'circle',
#         'FastTree (JC)': 'square',
#         'NJ (JC)': 'diamond',
#         'Original GTM (FastTree guide + FastTree subset)': 'star',
#         'Hybrid GTM 1 (FastTree guide + NJ-LogDet subset)': 'pentagon',
#         'Hybrid GTM 2 (NJ-LogDet guide + FastTree subset)': 'hexagon',
#         'Hybrid GTM 3 (NJ-LogDet guide + NJ-LogDet subset)': 'cross'
#     }
    
#     # Size based on RF distance (smaller is better)
#     max_rf = grouped_df['RF_mean'].max()
#     if max_rf > 0:
#         size_scale = lambda rf: 25 * (1 - (rf / max_rf)) + 10
#     else:
#         size_scale = lambda rf: 20  # Default size
    
#     # Create the scatter plot
#     fig = go.Figure()
    
#     # Add traces for each method and model
#     for model in grouped_df['Model'].unique():
#         for method in grouped_df['Method'].unique():
#             data = grouped_df[(grouped_df['Model'] == model) & (grouped_df['Method'] == method)]
#             if len(data) == 0:
#                 continue
            
#             symbol = method_symbols.get(method, 'circle')
#             color = model_colors.get(model, '#1f77b4')
            
#             # Handle missing or NaN values
#             fn_mean = data['FN_mean'].values[0] if not pd.isna(data['FN_mean'].values[0]) else 0
#             fp_mean = data['FP_mean'].values[0] if not pd.isna(data['FP_mean'].values[0]) else 0
#             rf_mean = data['RF_mean'].values[0] if not pd.isna(data['RF_mean'].values[0]) else 0
#             complexity = data['Complexity'].values[0] if not pd.isna(data['Complexity'].values[0]) else 3
            
#             fig.add_trace(go.Scatter(
#                 x=complexity,
#                 y=fn_mean,
#                 mode='markers',
#                 name=f"{model} - {method}",
#                 marker=dict(
#                     symbol=symbol,
#                     size=size_scale(rf_mean),
#                     color=color,
#                     line=dict(color='black', width=1)
#                 ),
#                 text=method,
#                 hovertemplate=(
#                     '<b>%{text}</b><br>' +
#                     'Model: ' + str(model) + '<br>' +
#                     'Complexity: %{x:.1f}<br>' +
#                     'FN Rate: %{y:.4f}<br>' +
#                     'FP Rate: ' + str(fp_mean) + '<br>' +
#                     'RF Distance: ' + str(rf_mean) +
#                     '<extra></extra>'
#                 )
#             ))
    
#     # Improve layout
#     fig.update_layout(
#         title=dict(
#             text='Performance vs. Computational Complexity',
#             font=dict(size=20),
#             x=0.5
#         ),
#         xaxis=dict(
#             title='Computational Complexity (Relative Scale)',
#             title_font=dict(size=16),
#             gridcolor='rgba(230,230,230,0.8)',
#             zeroline=True,
#             zerolinecolor='rgba(0,0,0,0.2)',
#             zerolinewidth=1,
#             range=[2, 5.5]
#         ),
#         yaxis=dict(
#             title='False Negative Rate',
#             title_font=dict(size=16),
#             gridcolor='rgba(230,230,230,0.8)',
#             zeroline=True,
#             zerolinecolor='rgba(0,0,0,0.2)',
#             zerolinewidth=1
#         ),
#         plot_bgcolor='rgba(245,245,245,0.5)',
#         width=1000,
#         height=800,
#         margin=dict(l=20, r=20, t=80, b=60),
#         font=dict(family="Arial, sans-serif", size=12),
#         legend=dict(
#             orientation="h",
#             yanchor="bottom",
#             y=-0.2,
#             xanchor="center",
#             x=0.5,
#             font=dict(size=10)
#         ),
#         hovermode='closest',
#         showlegend=True
#     )
    
#     # Add text annotations for each point
#     for model in grouped_df['Model'].unique():
#         for method in grouped_df['Method'].unique():
#             data = grouped_df[(grouped_df['Model'] == model) & (grouped_df['Method'] == method)]
#             if len(data) == 0:
#                 continue
                
#             # Shorten method name for annotation
#             short_method = method.split(' ')[0] if isinstance(method, str) else "Unknown"
            
#             # Handle missing values
#             x_val = data['Complexity'].values[0] if not pd.isna(data['Complexity'].values[0]) else 3
#             y_val = data['FN_mean'].values[0] if not pd.isna(data['FN_mean'].values[0]) else 0
            
#             fig.add_annotation(
#                 x=x_val + 0.05,
#                 y=y_val,
#                 text=short_method,
#                 showarrow=False,
#                 font=dict(size=10, color='black'),
#                 bgcolor='rgba(255,255,255,0.7)',
#                 bordercolor='rgba(0,0,0,0.1)',
#                 borderwidth=1,
#                 borderpad=2,
#                 opacity=0.8,
#                 xanchor='left'
#             )
    
#     # Save the plot
#     fig.write_html(f"{output_dir}/performance_vs_complexity_interactive.html")
#     fig.write_image(f"{output_dir}/performance_vs_complexity.png", scale=3)
    
#     # Create another version with complexity on Y-axis
#     fig2 = go.Figure()
    
#     # Add traces for each method and model
#     for model in grouped_df['Model'].unique():
#         for method in grouped_df['Method'].unique():
#             data = grouped_df[(grouped_df['Model'] == model) & (grouped_df['Method'] == method)]
#             if len(data) == 0:
#                 continue
                
#             symbol = method_symbols.get(method, 'circle')
#             color = model_colors.get(model, '#1f77b4')
            
#             # Handle missing or NaN values
#             fn_mean = data['FN_mean'].values[0] if not pd.isna(data['FN_mean'].values[0]) else 0
#             fp_mean = data['FP_mean'].values[0] if not pd.isna(data['FP_mean'].values[0]) else 0
#             rf_mean = data['RF_mean'].values[0] if not pd.isna(data['RF_mean'].values[0]) else 0
#             complexity = data['Complexity'].values[0] if not pd.isna(data['Complexity'].values[0]) else 3
            
#             fig2.add_trace(go.Scatter(
#                 y=complexity,
#                 x=fn_mean,
#                 mode='markers',
#                 name=f"{model} - {method}",
#                 marker=dict(
#                     symbol=symbol,
#                     size=size_scale(rf_mean),
#                     color=color,
#                     line=dict(color='black', width=1)
#                 ),
#                 text=method,
#                 hovertemplate=(
#                     '<b>%{text}</b><br>' +
#                     'Model: ' + str(model) + '<br>' +
#                     'Complexity: %{y:.1f}<br>' +
#                     'FN Rate: %{x:.4f}<br>' +
#                     'FP Rate: ' + str(fp_mean) + '<br>' +
#                     'RF Distance: ' + str(rf_mean) +
#                     '<extra></extra>'
#                 )
#             ))
    
#     # Improve layout
#     fig2.update_layout(
#         title=dict(
#             text='Performance vs. Computational Complexity',
#             font=dict(size=20),
#             x=0.5
#         ),
#         yaxis=dict(
#             title='Computational Complexity (Relative Scale)',
#             title_font=dict(size=16),
#             gridcolor='rgba(230,230,230,0.8)',
#             zeroline=True,
#             zerolinecolor='rgba(0,0,0,0.2)',
#             zerolinewidth=1,
#             range=[2, 5.5]
#         ),
#         xaxis=dict(
#             title='False Negative Rate',
#             title_font=dict(size=16),
#             gridcolor='rgba(230,230,230,0.8)',
#             zeroline=True,
#             zerolinecolor='rgba(0,0,0,0.2)',
#             zerolinewidth=1
#         ),
#         plot_bgcolor='rgba(245,245,245,0.5)',
#         width=1000,
#         height=800,
#         margin=dict(l=20, r=20, t=80, b=60),
#         font=dict(family="Arial, sans-serif", size=12),
#         legend=dict(
#             orientation="h",
#             yanchor="bottom",
#             y=-0.2,
#             xanchor="center",
#             x=0.5,
#             font=dict(size=10)
#         ),
#         hovermode='closest',
#         showlegend=True
#     )
    
#     # Add text annotations for each point
#     for model in grouped_df['Model'].unique():
#         for method in grouped_df['Method'].unique():
#             data = grouped_df[(grouped_df['Model'] == model) & (grouped_df['Method'] == method)]
#             if len(data) == 0:
#                 continue
                
#             # Shorten method name for annotation
#             short_method = method.split(' ')[0] if isinstance(method, str) else "Unknown"
            
#             # Handle missing values
#             x_val = data['FN_mean'].values[0] if not pd.isna(data['FN_mean'].values[0]) else 0
#             y_val = data['Complexity'].values[0] if not pd.isna(data['Complexity'].values[0]) else 3
            
#             fig2.add_annotation(
#                 y=y_val,
#                 x=x_val + 0.01,
#                 text=short_method,
#                 showarrow=False,
#                 font=dict(size=10, color='black'),
#                 bgcolor='rgba(255,255,255,0.7)',
#                 bordercolor='rgba(0,0,0,0.1)',
#                 borderwidth=1,
#                 borderpad=2,
#                 opacity=0.8,
#                 xanchor='left'
#             )
    
#     # Save the plot
#     fig2.write_html(f"{output_dir}/complexity_vs_performance_interactive.html")
#     fig2.write_image(f"{output_dir}/complexity_vs_performance.png", scale=3)

# def create_comparison_matrix(df, output_dir, model_colors):
#     """Create a comparison matrix of methods and metrics"""
#     # Safe grouping with error handling
#     try:
#         # Group by Method to get averages across all models
#         agg_dict = {
#             'RF_mean': 'mean',
#             'FN_mean': 'mean',
#             'FP_mean': 'mean'
#         }
        
#         grouped_df = df.groupby('Method').agg(agg_dict).reset_index()
        
#         # Sort by performance (FN rate)
#         grouped_df = grouped_df.sort_values('FN_mean')
        
#         # Create a comparison matrix as a heatmap
#         fig = make_subplots(
#             rows=1, cols=3,
#             subplot_titles=("RF Distance", "False Negative Rate", "False Positive Rate"),
#             shared_yaxes=True
#         )
        
#         # Handle NaN values
#         grouped_df = grouped_df.fillna(0)
        
#         # Normalize values for better visualization
#         max_rf = grouped_df['RF_mean'].max()
#         if max_rf > 0:
#             normalized_rf = grouped_df['RF_mean'] / max_rf
#         else:
#             normalized_rf = grouped_df['RF_mean']
        
        
#         # Add heatmap traces for each metric
#         fig.add_trace(
#             go.Heatmap(
#                 z=normalized_rf.values.reshape(-1, 1).T,
#                 y=['RF Distance'],
#                 x=grouped_df['Method'],
#                 colorscale='YlOrRd',
#                 showscale=False,
#                 text=grouped_df['RF_mean'].round(1).values.reshape(-1, 1).T,
#                 hovertemplate='<b>%{x}</b><br>RF Distance: %{text}<extra></extra>',
#                 texttemplate='%{text}',
#                 zmin=0,
#                 zmax=1
#             ),
#             row=1, col=1
#         )

#         fig.add_trace(
#             go.Heatmap(
#                 z=grouped_df['FN_mean'].values.reshape(-1, 1).T,
#                 y=['FN Rate'],
#                 x=grouped_df['Method'],
#                 colorscale='YlOrRd',
#                 showscale=False,
#                 text=grouped_df['FN_mean'].round(4).values.reshape(-1, 1).T,
#                 hovertemplate='<b>%{x}</b><br>FN Rate: %{text}<extra></extra>',
#                 texttemplate='%{text}',
#                 zmin=0,
#                 zmax=1
#             ),
#             row=1, col=2
#         )

#         fig.add_trace(
#             go.Heatmap(
#                 z=grouped_df['FP_mean'].values.reshape(-1, 1).T,
#                 y=['FP Rate'],
#                 x=grouped_df['Method'],
#                 colorscale='YlOrRd',
#                 showscale=True,
#                 text=grouped_df['FP_mean'].round(4).values.reshape(-1, 1).T,
#                 hovertemplate='<b>%{x}</b><br>FP Rate: %{text}<extra></extra>',
#                 texttemplate='%{text}',
#                 zmin=0,
#                 zmax=1
#             ),
#             row=1, col=3
#         )

#         # Update layout
#         fig.update_layout(
#             title=dict(
#                 text='Method Performance Comparison Matrix',
#                 font=dict(size=20),
#                 x=0.5
#             ),
#             height=300,
#             width=1200,
#             margin=dict(l=20, r=20, t=80, b=150),
#             font=dict(family="Arial, sans-serif", size=12),
#             coloraxis=dict(colorscale='YlOrRd'),
#             plot_bgcolor='rgba(245,245,245,0.5)'
#         )

#         # Update xaxis properties
#         fig.update_xaxes(
#             tickangle=45,
#             tickfont=dict(size=10),
#             tickmode='array',
#             tickvals=grouped_df['Method'],
#             ticktext=grouped_df['Method'],
#             automargin=True
#         )

#         # Save the plot
#         fig.write_html(f"{output_dir}/method_comparison_matrix.html")
#         fig.write_image(f"{output_dir}/method_comparison_matrix.png", scale=3)

#         # Model-specific heatmaps
#         # Group by Model and Method
#         agg_dict = {
#             'RF_mean': 'mean',
#             'FN_mean': 'mean',
#             'FP_mean': 'mean'
#         }

#         model_grouped = df.groupby(['Model', 'Method']).agg(agg_dict).reset_index()

#         # Get unique models and methods
#         models = model_grouped['Model'].unique()
#         methods = grouped_df['Method'].tolist()  # Use sorted methods

#         # Create a larger comparison matrix with all models
#         fig = make_subplots(
#             rows=len(models), 
#             cols=3,
#             subplot_titles=[
#                 f"{model} - RF Distance" for model in models] + 
#                 [f"{model} - FN Rate" for model in models] + 
#                 [f"{model} - FP Rate" for model in models
#             ],
#             shared_xaxes=True,
#             vertical_spacing=0.05
#         )

#         # Add heatmap traces for each model and metric
#         for i, model in enumerate(models):
#             model_data = model_grouped[model_grouped['Model'] == model]
            
#             # Create method-indexed data for consistent ordering
#             method_data = pd.DataFrame({'Method': methods})
#             method_data = method_data.merge(model_data, on='Method', how='left')
#             method_data = method_data.fillna(0)
            
#             # Normalize RF values for better visualization
#             max_rf_model = method_data['RF_mean'].max()
#             if max_rf_model > 0:
#                 normalized_rf_model = method_data['RF_mean'] / max_rf_model
#             else:
#                 normalized_rf_model = method_data['RF_mean']
            
#             # Add RF distance heatmap
#             fig.add_trace(
#                 go.Heatmap(
#                     z=normalized_rf_model.values.reshape(-1, 1).T,
#                     y=[f"{model} RF"],
#                     x=method_data['Method'],
#                     colorscale='YlOrRd',
#                     showscale=(i==0),
#                     text=method_data['RF_mean'].round(1).values.reshape(-1, 1).T,
#                     hovertemplate='<b>%{x}</b><br>RF Distance: %{text}<extra></extra>',
#                     texttemplate='%{text}',
#                     zmin=0,
#                     zmax=1
#                 ),
#                 row=i+1, col=1
#             )
            
#             # Add FN rate heatmap
#             fig.add_trace(
#                 go.Heatmap(
#                     z=method_data['FN_mean'].values.reshape(-1, 1).T,
#                     y=[f"{model} FN"],
#                     x=method_data['Method'],
#                     colorscale='YlOrRd',
#                     showscale=(i==0),
#                     text=method_data['FN_mean'].round(4).values.reshape(-1, 1).T,
#                     hovertemplate='<b>%{x}</b><br>FN Rate: %{text}<extra></extra>',
#                     texttemplate='%{text}',
#                     zmin=0,
#                     zmax=1
#                 ),
#                 row=i+1, col=2
#             )
            
#             # Add FP rate heatmap
#             fig.add_trace(
#                 go.Heatmap(
#                     z=method_data['FP_mean'].values.reshape(-1, 1).T,
#                     y=[f"{model} FP"],
#                     x=method_data['Method'],
#                     colorscale='YlOrRd',
#                     showscale=(i==0),
#                     text=method_data['FP_mean'].round(4).values.reshape(-1, 1).T,
#                     hovertemplate='<b>%{x}</b><br>FP Rate: %{text}<extra></extra>',
#                     texttemplate='%{text}',
#                     zmin=0,
#                     zmax=1
#                 ),
#                 row=i+1, col=3
#             )

#         # Update layout
#         fig.update_layout(
#             title=dict(
#                 text='Detailed Method Performance Comparison by Model',
#                 font=dict(size=20),
#                 x=0.5
#             ),
#             height=300 * len(models),
#             width=1200,
#             margin=dict(l=20, r=20, t=80, b=150),
#             font=dict(family="Arial, sans-serif", size=12),
#             coloraxis=dict(colorscale='YlOrRd'),
#             plot_bgcolor='rgba(245,245,245,0.5)'
#         )

#         # Update xaxis properties
#         fig.update_xaxes(
#             tickangle=45,
#             tickfont=dict(size=10),
#             tickmode='array',
#             tickvals=methods,
#             ticktext=methods,
#             automargin=True
#         )

#         # Only show x-axis labels for the bottom row
#         for i in range(len(models)-1):
#             for j in range(3):
#                 fig.update_xaxes(showticklabels=False, row=i+1, col=j+1)

#         # Save the plot
#         fig.write_html(f"{output_dir}/detailed_method_comparison_matrix.html")
#         fig.write_image(f"{output_dir}/detailed_method_comparison_matrix.png", scale=3)

#         # Create a comprehensive dashboard with all visualizations
#         dashboard = make_subplots(
#             rows=3, 
#             cols=2,
#             subplot_titles=(
#                 'Performance vs. Complexity', 
#                 'Robinson-Foulds Distances', 
#                 'False Negative Rates', 
#                 'False Positive Rates',
#                 'Method Comparison Matrix',
#                 ''
#             ),
#             specs=[
#                 [{"type": "scatter"}, {"type": "bar"}],
#                 [{"type": "bar"}, {"type": "bar"}],
#                 [{"type": "heatmap", "colspan": 2}, {}]
#             ],
#             vertical_spacing=0.1,
#             horizontal_spacing=0.05
#         )

#         # Add complexity plot
#         for model in grouped_df['Model'].unique():
#             model_data = grouped_df[grouped_df['Model'] == model]
            
#             dashboard.add_trace(
#                 go.Scatter(
#                     x=model_data['Complexity'],
#                     y=model_data['FN_mean'],
#                     mode='markers+text',
#                     name=model,
#                     marker=dict(
#                         size=15,
#                         color=model_colors.get(model, '#1f77b4'),
#                         line=dict(color='black', width=1)
#                     ),
#                     text=model_data['Method'].apply(lambda x: x.split(' ')[0] if isinstance(x, str) else "Unknown"),
#                     textposition="top center",
#                     hovertemplate='<b>%{text}</b><br>Complexity: %{x:.1f}<br>FN Rate: %{y:.4f}<extra></extra>'
#                 ),
#                 row=1, col=1
#             )

#         # Add RF distance bars
#         for i, model in enumerate(models):
#             model_data = grouped_df[grouped_df['Model'] == model]
            
#             dashboard.add_trace(
#                 go.Bar(
#                     x=model_data['Method'],
#                     y=model_data['RF_mean'],
#                     name=model,
#                     marker=dict(color=model_colors.get(model, '#1f77b4')),
#                     hovertemplate='<b>%{x}</b><br>Model: ' + str(model) + '<br>RF Distance: %{y:.1f}<extra></extra>'
#                 ),
#                 row=1, col=2
#             )

#         # Add FN rate bars
#         for i, model in enumerate(models):
#             model_data = grouped_df[grouped_df['Model'] == model]
            
#             dashboard.add_trace(
#                 go.Bar(
#                     x=model_data['Method'],
#                     y=model_data['FN_mean'],
#                     name=model,
#                     marker=dict(color=model_colors.get(model, '#1f77b4')),
#                     hovertemplate='<b>%{x}</b><br>Model: ' + str(model) + '<br>FN Rate: %{y:.4f}<extra></extra>',
#                     showlegend=False
#                 ),
#                 row=2, col=1
#             )

#         # Add FP rate bars
#         for i, model in enumerate(models):
#             model_data = grouped_df[grouped_df['Model'] == model]
            
#             dashboard.add_trace(
#                 go.Bar(
#                     x=model_data['Method'],
#                     y=model_data['FP_mean'],
#                     name=model,
#                     marker=dict(color=model_colors.get(model, '#1f77b4')),
#                     hovertemplate='<b>%{x}</b><br>Model: ' + str(model) + '<br>FP Rate: %{y:.4f}<extra></extra>',
#                     showlegend=False
#                 ),
#                 row=2, col=2
#             )

#         # Add comparison matrix heatmap
#         dashboard.add_trace(
#             go.Heatmap(
#                 z=np.column_stack((
#                     normalized_rf.values,
#                     grouped_df['FN_mean'].values,
#                     grouped_df['FP_mean'].values
#                 )),
#                 y=grouped_df['Method'],
#                 x=['RF (normalized)', 'FN Rate', 'FP Rate'],
#                 colorscale='YlOrRd',
#                 showscale=True,
#                 text=np.column_stack((
#                     grouped_df['RF_mean'].round(1).values,
#                     grouped_df['FN_mean'].round(4).values,
#                     grouped_df['FP_mean'].round(4).values
#                 )),
#                 hovertemplate='<b>%{y}</b><br>%{x}: %{text}<extra></extra>',
#                 texttemplate='%{text}'
#             ),
#             row=3, col=1
#         )

#         # Update layout
#         dashboard.update_layout(
#             title=dict(
#                 text='Comprehensive Method Performance Dashboard',
#                 font=dict(size=24),
#                 x=0.5
#             ),
#             height=1200,
#             width=1600,
#             margin=dict(l=20, r=20, t=100, b=20),
#             font=dict(family="Arial, sans-serif", size=12),
#             legend=dict(
#                 orientation="h",
#                 yanchor="bottom",
#                 y=1.02,
#                 xanchor="center",
#                 x=0.5
#             ),
#             barmode='group',
#             hovermode='closest',
#             plot_bgcolor='rgba(245,245,245,0.5)'
#         )

#         # Update xaxis properties
#         dashboard.update_xaxes(tickangle=45, automargin=True)

#         # Add axis titles
#         dashboard.update_xaxes(title_text="Computational Complexity", row=1, col=1)
#         dashboard.update_yaxes(title_text="False Negative Rate", row=1, col=1)

#         dashboard.update_xaxes(title_text="Method", row=1, col=2)
#         dashboard.update_yaxes(title_text="Robinson-Foulds Distance", row=1, col=2)

#         dashboard.update_xaxes(title_text="Method", row=2, col=1)
#         dashboard.update_yaxes(title_text="False Negative Rate", row=2, col=1)

#         dashboard.update_xaxes(title_text="Method", row=2, col=2)
#         dashboard.update_yaxes(title_text="False Positive Rate", row=2, col=2)

#         dashboard.update_xaxes(title_text="Metric", row=3, col=1)
#         dashboard.update_yaxes(title_text="Method", row=3, col=1)

#         # Save the dashboard
#         dashboard.write_html(f"{output_dir}/comprehensive_dashboard.html")
#         dashboard.write_image(f"{output_dir}/comprehensive_dashboard.png", scale=3)

#     except Exception as e:
#         print(f"Error in creating comparison matrix: {str(e)}")
#         # Create a simple alternative comparison plot
#         try:
#             alt_fig = go.Figure()
            
#             # Add bars for each metric and method
#             for method in df['Method'].unique():
#                 method_data = df[df['Method'] == method]
#                 method_avg = method_data[['RF_mean', 'FN_mean', 'FP_mean']].mean()
                
#                 alt_fig.add_trace(go.Bar(
#                     y=[method],
#                     x=[method_avg['FN_mean']],
#                     name='FN Rate',
#                     orientation='h',
#                     marker=dict(color='red'),
#                     hovertemplate='<b>%{y}</b><br>FN Rate: %{x:.4f}<extra></extra>'
#                 ))
                
#                 alt_fig.add_trace(go.Bar(
#                     y=[method],
#                     x=[method_avg['FP_mean']],
#                     name='FP Rate',
#                     orientation='h',
#                     marker=dict(color='blue'),
#                     hovertemplate='<b>%{y}</b><br>FP Rate: %{x:.4f}<extra></extra>'
#                 ))
            
#             alt_fig.update_layout(
#                 title='Method Performance Comparison (Simplified)',
#                 barmode='group',
#                 height=600,
#                 width=1000,
#                 yaxis=dict(title='Method'),
#                 xaxis=dict(title='Rate'),
#                 font=dict(family="Arial, sans-serif", size=12),
#                 legend=dict(orientation="h", y=1.1, x=0.5)
#             )
            
#             alt_fig.write_html(f"{output_dir}/simplified_comparison.html")
#             alt_fig.write_image(f"{output_dir}/simplified_comparison.png", scale=3)
            
#         except Exception as e2:
#             print(f"Error creating simplified alternative: {str(e2)}")

# if __name__ == "__main__":
#     main()