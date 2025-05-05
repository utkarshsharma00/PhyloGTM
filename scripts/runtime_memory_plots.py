import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker

# Set the style for better-looking plots
plt.style.use('ggplot')

# Runtime data with dependencies properly included
runtime_data = {
    'Method': [
        'NJ (JC)',
        'FastME (BME)',
        'Hybrid GTM 3',
        'NJ (p-distance)',
        'NJ (LogDet)',
        'Hybrid GTM 1',
        'FastTree (JC)',
        'Hybrid GTM 2',
        'FastTree (GTR)',
        'Original GTM'
    ],
    'Model Condition 1': [
        308.44,
        287.60,
        1153.33,  # 352.60 + 800.73 for NJ-LogDet dependency
        999.71,
        800.73,
        1441.17,  # 357.21 + 1083.95 for FastTree GTR dependency
        887.73,
        4279.06,  # 3478.33 + 800.73 for NJ-LogDet dependency
        1083.95,
        4984.52   # Sum of all Original GTM steps
    ],
    'Model Condition 4': [
        216.09,
        196.17,
        929.34,   # 352.60 + 576.74 for NJ-LogDet dependency
        216.17,
        576.74,
        1017.13,  # 357.21 + 659.92 for FastTree GTR dependency
        513.58,
        4055.07,  # 3478.33 + 576.74 for NJ-LogDet dependency
        659.92,
        4984.52   # Same as Model 1 since no Model 4 data in original
    ],
    'Description': [
        'Neighbor Joining with JC distances (fastest baseline method)',
        'FastME with Balanced Minimum Evolution criterion',
        'NJ-LogDet guide tree + NJ-LogDet subset trees',
        'Neighbor Joining with p-distances',
        'Neighbor Joining with LogDet distances',
        'FastTree guide tree + NJ-LogDet subset trees',
        'FastTree with JC model',
        'NJ-LogDet guide tree + FastTree subset trees',
        'FastTree with GTR model (slowest baseline method)',
        'FastTree guide tree + FastTree subset trees (slowest method overall)'
    ]
}

# Memory data with dependencies (max memory usage) in MB
memory_data = {
    'Method': [
        'NJ (JC)',
        'FastME (BME)',
        'Hybrid GTM 3',
        'NJ (p-distance)',
        'NJ (LogDet)',
        'Hybrid GTM 1',
        'FastTree (JC)',
        'Hybrid GTM 2',
        'FastTree (GTR)',
        'Original GTM'
    ],
    'Model Condition 1': [
        1872/1024,
        1904/1024,
        1936/1024,  # Max of Hybrid GTM 3 steps and NJ-LogDet
        1904/1024,
        1904/1024,
        1952/1024,  # Max of Hybrid GTM 1 steps and FastTree GTR
        1904/1024,
        1952/1024,  # Max of Hybrid GTM 2 steps and NJ-LogDet
        1904/1024,
        1952/1024   # Max of Original GTM steps
    ],
    'Model Condition 4': [
        1952/1024,
        1952/1024,
        1952/1024,  # Max of Hybrid GTM 3 steps and NJ-LogDet
        1952/1024,
        1952/1024,
        1952/1024,  # Max of Hybrid GTM 1 steps and FastTree GTR
        1952/1024,
        1952/1024,  # Max of Hybrid GTM 2 steps and NJ-LogDet
        1952/1024,
        1952/1024   # Same as Model 1
    ]
}

# Create dataframes
runtime_df = pd.DataFrame(runtime_data)
memory_df = pd.DataFrame(memory_data)

# Calculate average runtime for sorting
runtime_df['Average'] = (runtime_df['Model Condition 1'] + runtime_df['Model Condition 4']) / 2
memory_df['Average'] = (memory_df['Model Condition 1'] + memory_df['Model Condition 4']) / 2

# Sort by average runtime (fastest first)
runtime_df = runtime_df.sort_values('Average')
memory_df = memory_df.loc[runtime_df.index]  # Keep same order as runtime for consistency

# Define colors based on method types
def get_method_color(method_name):
    if 'Hybrid GTM' in method_name:
        return '#8884d8'  # Purple for hybrid methods
    elif 'Original GTM' in method_name:
        return '#4C78A8'  # Blue for original GTM
    elif 'FastTree' in method_name:
        return '#E45756'  # Red for FastTree methods
    elif 'NJ' in method_name:
        return '#72B7B2'  # Teal for NJ methods
    else:
        return '#F28E2B'  # Orange for other methods

# Function to format time for better readability
def format_time(seconds):
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    if minutes > 0:
        return f"{minutes}m {remaining_seconds}s"
    else:
        return f"{remaining_seconds}s"

# Helper to add value labels on bars
def add_value_labels(ax, spacing=5, format_func=None):
    for rect in ax.patches:
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2
        
        # If formatting function is provided, use it
        if format_func:
            label = format_func(y_value)
        else:
            label = f"{y_value:.1f}"
            
        # Vertical alignment depends on height of bar
        va = 'bottom' if y_value < 0 else 'top'
        
        # Create annotation
        ax.annotate(
            label,
            (x_value, y_value),
            xytext=(0, spacing),
            textcoords="offset points",
            ha='center',
            va=va,
            fontsize=8,
            rotation=0
        )

#-------------------------------------------------------------------------
# 1. Runtime Performance Visualization
#-------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 10))

# Plot data
bar_width = 0.35
x = np.arange(len(runtime_df))

# Calculate average for bar coloring
avg_runtime = (runtime_df['Model Condition 1'] + runtime_df['Model Condition 4']) / 2
colors = [get_method_color(method) for method in runtime_df['Method']]

# Plot bars
bars1 = ax.bar(x - bar_width/2, runtime_df['Model Condition 1'], bar_width, 
              color='#1f77b4', label='1000M1')  # Blue
bars2 = ax.bar(x + bar_width/2, runtime_df['Model Condition 4'], bar_width, 
              color='#ff7f0e', label='1000M4')  # Orange

# Add grid
ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.7)

# Configure axes
ax.set_xticks(x)
ax.set_xticklabels(runtime_df['Method'], rotation=45, ha='right')
ax.set_ylabel('Runtime (seconds)', fontsize=12)
ax.set_title('Runtime Comparison', fontsize=16, pad=20)

# Format y-axis with comma separators for thousands
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

# Add value labels to runtime bars
for i, value in enumerate(runtime_df['Model Condition 1']):
    ax.text(i - bar_width/2, value + 50, f"{int(value)}", ha='center', va='bottom', rotation=0, fontsize=9)
    
for i, value in enumerate(runtime_df['Model Condition 4']):
    ax.text(i + bar_width/2, value + 50, f"{int(value)}", ha='center', va='bottom', rotation=0, fontsize=9)

# Move legend to the left side
legend1 = ax.legend(handles=[bars1, bars2], labels=['1000M1', '1000M4'],
                   loc='upper left', title="Model Conditions")
ax.add_artist(legend1)

# Calculate speedup compared to Original GTM
original_avg = runtime_df[runtime_df['Method'] == 'Original GTM']['Average'].values[0]

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig('runtime_comparison.png', dpi=300, bbox_inches='tight')

#-------------------------------------------------------------------------
# 2. Memory Usage Visualization
#-------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 10))

# Plot data
bar_width = 0.35
x = np.arange(len(memory_df))

# Calculate average for bar coloring
avg_memory = (memory_df['Model Condition 1'] + memory_df['Model Condition 4']) / 2
colors = [get_method_color(method) for method in memory_df['Method']]

# Plot bars
# For more distinct colors instead of transparency
bars1 = ax.bar(x - bar_width/2, memory_df['Model Condition 1'], bar_width, 
              color='#1f77b4', label='1000M1')  # Blue
bars2 = ax.bar(x + bar_width/2, memory_df['Model Condition 4'], bar_width, 
              color='#ff7f0e', label='1000M4')  # Orange

# Add grid
ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.7)

# Configure axes
ax.set_xticks(x)
ax.set_xticklabels(memory_df['Method'], rotation=45, ha='right')
ax.set_ylabel('Memory Usage (GB)', fontsize=12)
ax.set_title('Memory Usage Comparison', fontsize=16, pad=20)

# Format y-axis as GB (input is in MB)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x:.2f}"))

# Add value labels to memory bars
for i, value in enumerate(memory_df['Model Condition 1']):
    ax.text(i - bar_width/2, value + 0.05, f"{value:.2f}", ha='center', va='bottom', rotation=0, fontsize=9)
    
for i, value in enumerate(memory_df['Model Condition 4']):
    ax.text(i + bar_width/2, value + 0.05, f"{value:.2f}", ha='center', va='bottom', rotation=0, fontsize=9)

# Move legend to the left side
legend1 = ax.legend(handles=[bars1, bars2], labels=['1000M1', '1000M4'],
                   loc='lower left', title="Model Conditions")
ax.add_artist(legend1)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig('memory_comparison.png', dpi=300, bbox_inches='tight')

# Display key performance insights as a summary table
print("\nPerformance Summary (ordered from fastest to slowest):")
print("=" * 80)
print(f"{'Method':<30} {'Avg Runtime':<15} {'Speedup vs Original':<20} {'Avg Memory (GB)':<15}")
print("-" * 80)

for i, method in enumerate(runtime_df['Method']):
    avg_runtime = runtime_df['Average'].iloc[i]
    avg_memory = memory_df['Average'].iloc[i]
    speedup = original_avg / avg_runtime
    
    runtime_str = format_time(avg_runtime)
    print(f"{method:<30} {runtime_str:<15} {speedup:.1f}x{' faster':13} {avg_memory:.2f}")

print("=" * 80)