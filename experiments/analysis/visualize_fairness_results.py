import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Read the fairness analysis results
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "output")
results_path = os.path.join(output_dir, "fairness_analysis_results.csv")
fairness_df = pd.read_csv(results_path)

def plot_fairness_gaps(df, metric_type, save_dir):
    """
    Create a bar plot of fairness gaps for a specific metric
    """
    plt.figure(figsize=(12, 6))
    
    # Create subplot for each subgroup type
    for i, subgroup in enumerate(df['subgroup'].unique()):
        subgroup_data = df[df['subgroup'] == subgroup]
        
        plt.subplot(1, len(df['subgroup'].unique()), i+1)
        
        # Create bar plot
        bars = plt.bar(range(len(subgroup_data)), 
                      subgroup_data[f'fairness_gap_{metric_type}'],
                      color='skyblue')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        # Customize plot
        plt.title(f'{subgroup} - {metric_type.capitalize()} Gap')
        plt.xticks(range(len(subgroup_data)), 
                  [f'{row.value1}\nvs\n{row.value2}' for _, row in subgroup_data.iterrows()],
                  rotation=45)
        plt.ylabel('Gap')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'fairness_gaps_{metric_type}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_confidence_intervals(df, metric_type, save_dir):
    """
    Create a forest plot of confidence intervals for a specific metric
    """
    plt.figure(figsize=(12, 6))
    
    # Extract confidence intervals
    ci_column = f'{metric_type}_confidence_interval'
    df['lower'] = df[ci_column].apply(lambda x: eval(x)[0] if pd.notna(x) else None)
    df['upper'] = df[ci_column].apply(lambda x: eval(x)[1] if pd.notna(x) else None)
    
    # Plot for each subgroup
    for i, subgroup in enumerate(df['subgroup'].unique()):
        subgroup_data = df[df['subgroup'] == subgroup]
        
        plt.subplot(1, len(df['subgroup'].unique()), i+1)
        
        # Create error bars
        y_pos = range(len(subgroup_data))
        plt.errorbar(
            x=(subgroup_data['lower'] + subgroup_data['upper'])/2,
            y=y_pos,
            xerr=[(subgroup_data['upper'] - subgroup_data['lower'])/2],
            fmt='o',
            capsize=5,
            capthick=2,
            label=subgroup
        )
        
        # Customize plot
        plt.title(f'{subgroup} - {metric_type.capitalize()} CI')
        plt.yticks(y_pos, [f'{row.value1}\nvs\n{row.value2}' for _, row in subgroup_data.iterrows()])
        plt.xlabel(f'{metric_type.capitalize()} Score')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'confidence_intervals_{metric_type}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_group_sizes(df, save_dir):
    """
    Create a comparison plot of group sizes
    """
    plt.figure(figsize=(12, 6))
    
    for i, subgroup in enumerate(df['subgroup'].unique()):
        subgroup_data = df[df['subgroup'] == subgroup]
        
        plt.subplot(1, len(df['subgroup'].unique()), i+1)
        
        # Create grouped bar plot
        x = range(len(subgroup_data))
        width = 0.35
        
        plt.bar(x, subgroup_data['size1'], width, label='Group 1', color='skyblue')
        plt.bar([i + width for i in x], subgroup_data['size2'], width, label='Group 2', color='lightcoral')
        
        # Customize plot
        plt.title(f'{subgroup} - Group Sizes')
        plt.xticks([i + width/2 for i in x], 
                  [f'{row.value1}\nvs\n{row.value2}' for _, row in subgroup_data.iterrows()],
                  rotation=45)
        plt.ylabel('Size')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'group_sizes.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_table(df, save_dir):
    """
    Create a summary table of all metrics
    """
    summary = []
    for subgroup in df['subgroup'].unique():
        subgroup_data = df[df['subgroup'] == subgroup]
        
        for _, row in subgroup_data.iterrows():
            summary.append({
                'Subgroup': subgroup,
                'Comparison': f'{row.value1} vs {row.value2}',
                'Group Sizes': f'{row.size1} vs {row.size2}',
                'Accuracy Gap': f'{row.fairness_gap_accuracy:.3f}',
                'Sensitivity Gap': f'{row.fairness_gap_sensitivity:.3f}',
                'Specificity Gap': f'{row.fairness_gap_specificity:.3f}'
            })
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(save_dir, 'fairness_summary.csv'), index=False)

def main():
    # Create visualization directory
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Generate all plots
    metrics = ['accuracy', 'sensitivity', 'specificity']
    for metric in metrics:
        plot_fairness_gaps(fairness_df, metric, vis_dir)
        plot_confidence_intervals(fairness_df, metric, vis_dir)
    
    plot_group_sizes(fairness_df, vis_dir)
    create_summary_table(fairness_df, vis_dir)
    
    print("Visualizations have been created in:", vis_dir)

if __name__ == "__main__":
    main()