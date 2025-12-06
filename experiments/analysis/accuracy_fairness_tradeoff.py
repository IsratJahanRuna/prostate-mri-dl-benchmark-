import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Define the base directory where the model predictions are stored
base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "input")

# Load tumor-related data
tumor_data = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "patient_tumor_analysis_results.csv"))

# Define models and datasets
models = ['densenet', 'resnet', 'efficientnet']
datasets = ['T2', 'ADC', 'DWI']

# Function to define subgroups by combining values from two tumor readers
def define_subgroups(row):
    # Lesion Zones: Combine Lesion_Zones_1 and Lesion_Zones_2
    if row['Lesion_Zones_1'] != row['Lesion_Zones_2']:
        row['Lesion_Zones'] = 'Mixed'  # Use 'Mixed' if values differ
    else:
        row['Lesion_Zones'] = row['Lesion_Zones_1']  # Use the same label if values are the same
    
    # Tumor Size: Combine Tumor_Size_1 and Tumor_Size_2
    if row['Tumor_Size_1'] != row['Tumor_Size_2']:
        row['Tumor_Size'] = 'Mixed'  # Use 'Mixed' if values differ
    else:
        row['Tumor_Size'] = row['Tumor_Size_1']  # Use the same label if values are the same
    
    # Lesion Count: Combine Lesion_Count_1 and Lesion_Count_2
    if row['Lesion_Count_1'] != row['Lesion_Count_2']:
        row['Lesion_Count'] = 'Mixed'  # Use 'Mixed' if values differ
    else:
        row['Lesion_Count'] = row['Lesion_Count_1']  # Use the same count if values are the same
    
    return row

# Initialize an empty DataFrame for storing the merged data
merged_data = pd.DataFrame()

# Load each prediction file and merge with tumor data
for model_name in models:
    for dataset_name in datasets:
        # Construct the file path for the prediction file
        file_path = os.path.join(base_dir, model_name, dataset_name, "test_patient_preds.csv")
        
        if os.path.exists(file_path):
            preds = pd.read_csv(file_path)
            preds['model'] = model_name
            preds['dataset'] = dataset_name
            
            # Merge predictions with tumor data
            merged_data = pd.concat([merged_data, pd.merge(preds, tumor_data, left_on='patient_id', right_on='Patient_ID', how='left')])

# Apply the function to define subgroups
merged_data = merged_data.apply(define_subgroups, axis=1)

# Define thresholds (0.1 to 0.9)
thresholds = np.linspace(0.1, 0.9, 9)

# Create a DataFrame to store the results for each threshold
threshold_results = []

# Loop over the thresholds
for thr in thresholds:
    # Apply threshold to the predicted probabilities
    merged_data['y_hat'] = (merged_data['p'] >= thr).astype(int)
    
    # Calculate metrics for all tumor size pairs
    size_pairs = [
        ('Large', 'Medium'),
        ('Large', 'Small'),
        ('Medium', 'Small')
    ]
    
    pair_results = {}
    for size1, size2 in size_pairs:
        subgroup1 = merged_data[merged_data['Tumor_Size'] == size1]
        subgroup2 = merged_data[merged_data['Tumor_Size'] == size2]
        
        # Skip if either subgroup is empty
        if len(subgroup1) == 0 or len(subgroup2) == 0:
            print(f"Warning: Empty subgroup at threshold {thr}. {size1}: {len(subgroup1)}, {size2}: {len(subgroup2)}")
            continue
            
        # Calculate metrics for this pair
        accuracy1 = (subgroup1['y_true'] == subgroup1['y_hat']).mean()
        accuracy2 = (subgroup2['y_true'] == subgroup2['y_hat']).mean()
        
        # Safe calculation of sensitivity with error handling
        try:
            sensitivity1 = (subgroup1['y_true'] & subgroup1['y_hat']).sum() / max(subgroup1['y_true'].sum(), 1)
            sensitivity2 = (subgroup2['y_true'] & subgroup2['y_hat']).sum() / max(subgroup2['y_true'].sum(), 1)
        except:
            sensitivity1 = sensitivity2 = float('nan')
            
        # Safe calculation of specificity
        try:
            tn1 = ((~subgroup1['y_true']) & (~subgroup1['y_hat'])).sum()
            fp1 = ((~subgroup1['y_true']) & subgroup1['y_hat']).sum()
            tn2 = ((~subgroup2['y_true']) & (~subgroup2['y_hat'])).sum()
            fp2 = ((~subgroup2['y_true']) & subgroup2['y_hat']).sum()
            
            specificity1 = tn1 / max(tn1 + fp1, 1)
            specificity2 = tn2 / max(tn2 + fp2, 1)
        except:
            specificity1 = specificity2 = float('nan')
            
        pair_results[f"{size1}_{size2}"] = {
            'size1': size1,
            'size2': size2,
            'accuracy1': accuracy1,
            'accuracy2': accuracy2,
            'sensitivity1': sensitivity1,
            'sensitivity2': sensitivity2,
            'specificity1': specificity1,
            'specificity2': specificity2,
            'accuracy_gap': accuracy1 - accuracy2,
            'sensitivity_gap': sensitivity1 - sensitivity2,
            'specificity_gap': specificity1 - specificity2,
            'n1': len(subgroup1),
            'n2': len(subgroup2)
        }
    
    # Calculate accuracy, sensitivity, and specificity for each subgroup
    accuracy1 = (subgroup1['y_true'] == subgroup1['y_hat']).mean()
    accuracy2 = (subgroup2['y_true'] == subgroup2['y_hat']).mean()
    sensitivity1 = (subgroup1['y_true'] & subgroup1['y_hat']).sum() / (subgroup1['y_true']).sum()
    sensitivity2 = (subgroup2['y_true'] & subgroup2['y_hat']).sum() / (subgroup2['y_true']).sum()
    specificity1 = ((~subgroup1['y_true']) & (~subgroup1['y_hat'])).sum() / ((~subgroup1['y_true']) | (~subgroup1['y_hat'])).sum()
    specificity2 = ((~subgroup2['y_true']) & (~subgroup2['y_hat'])).sum() / ((~subgroup2['y_true']) | (~subgroup2['y_hat'])).sum()

    # Calculate fairness gap between subgroups (accuracy gap as an example)
    accuracy_gap = accuracy1 - accuracy2
    sensitivity_gap = sensitivity1 - sensitivity2
    specificity_gap = specificity1 - specificity2
    
    # Store results
    # Store results for each pair
    result = {'Threshold': thr}
    for pair_key, pair_data in pair_results.items():
        size1, size2 = pair_data['size1'], pair_data['size2']
        result.update({
            f'Accuracy_{size1}': pair_data['accuracy1'],
            f'Accuracy_{size2}': pair_data['accuracy2'],
            f'Accuracy_Gap_{size1}_{size2}': pair_data['accuracy_gap'],
            f'Sensitivity_{size1}': pair_data['sensitivity1'],
            f'Sensitivity_{size2}': pair_data['sensitivity2'],
            f'Sensitivity_Gap_{size1}_{size2}': pair_data['sensitivity_gap'],
            f'Specificity_{size1}': pair_data['specificity1'],
            f'Specificity_{size2}': pair_data['specificity2'],
            f'Specificity_Gap_{size1}_{size2}': pair_data['specificity_gap'],
            f'N_{size1}': pair_data['n1'],
            f'N_{size2}': pair_data['n2']
        })
    threshold_results.append(result)

# Convert the results into a DataFrame
threshold_df = pd.DataFrame(threshold_results)

# Display the threshold results
print(threshold_df)

# Create visualizations
def create_visualizations(df, save_dir):
    # Plot 1: Accuracy Gaps
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    for pair in ['Large_Medium', 'Large_Small', 'Medium_Small']:
        plt.plot(df['Threshold'], df[f'Accuracy_Gap_{pair}'], 
                marker='o', label=f'{pair.replace("_", " vs ")}')
    plt.xlabel('Classification Threshold')
    plt.ylabel('Accuracy Gap')
    plt.title('Accuracy Gaps vs Threshold')
    plt.legend()
    plt.grid(True)

    # Plot 2: Sensitivity Gaps
    plt.subplot(2, 2, 2)
    for pair in ['Large_Medium', 'Large_Small', 'Medium_Small']:
        plt.plot(df['Threshold'], df[f'Sensitivity_Gap_{pair}'],
                marker='o', label=f'{pair.replace("_", " vs ")}')
    plt.xlabel('Classification Threshold')
    plt.ylabel('Sensitivity Gap')
    plt.title('Sensitivity Gaps vs Threshold')
    plt.legend()
    plt.grid(True)

    # Plot 3: Specificity Gaps
    plt.subplot(2, 2, 3)
    for pair in ['Large_Medium', 'Large_Small', 'Medium_Small']:
        plt.plot(df['Threshold'], df[f'Specificity_Gap_{pair}'],
                marker='o', label=f'{pair.replace("_", " vs ")}')
    plt.xlabel('Classification Threshold')
    plt.ylabel('Specificity Gap')
    plt.title('Specificity Gaps vs Threshold')
    plt.legend()
    plt.grid(True)

    # Plot 4: Accuracy Trade-off
    plt.subplot(2, 2, 4)
    colors = ['blue', 'red', 'green']
    markers = ['o', 's', '^']
    for i, size in enumerate(['Large', 'Medium', 'Small']):
        plt.scatter(df[f'Accuracy_Gap_Large_Small'], df[f'Accuracy_{size}'],
                   label=size, color=colors[i], marker=markers[i])
    plt.xlabel('Accuracy Gap (Large vs Small)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy-Fairness Trade-off')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fairness_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Create detailed performance plots
    plt.figure(figsize=(15, 5))
    
    # Plot accuracy for each group
    plt.subplot(1, 3, 1)
    for size in ['Large', 'Medium', 'Small']:
        plt.plot(df['Threshold'], df[f'Accuracy_{size}'],
                marker='o', label=size)
    plt.xlabel('Classification Threshold')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Tumor Size')
    plt.legend()
    plt.grid(True)

    # Plot sensitivity for each group
    plt.subplot(1, 3, 2)
    for size in ['Large', 'Medium', 'Small']:
        plt.plot(df['Threshold'], df[f'Sensitivity_{size}'],
                marker='o', label=size)
    plt.xlabel('Classification Threshold')
    plt.ylabel('Sensitivity')
    plt.title('Sensitivity by Tumor Size')
    plt.legend()
    plt.grid(True)

    # Plot specificity for each group
    plt.subplot(1, 3, 3)
    for size in ['Large', 'Medium', 'Small']:
        plt.plot(df['Threshold'], df[f'Specificity_{size}'],
                marker='o', label=size)
    plt.xlabel('Classification Threshold')
    plt.ylabel('Specificity')
    plt.title('Specificity by Tumor Size')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'performance_by_size.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Function to find optimal thresholds based on different criteria
def analyze_optimal_thresholds(df):
    results = []
    
    # 1. Minimize maximum fairness gap
    max_gaps = df.apply(lambda row: max(abs(row['Accuracy_Gap_Large_Medium']),
                                      abs(row['Accuracy_Gap_Large_Small']),
                                      abs(row['Accuracy_Gap_Medium_Small'])), axis=1)
    opt_thresh_min_gap = df.loc[max_gaps.idxmin(), 'Threshold']
    
    # 2. Maximize minimum accuracy while maintaining fairness
    # (fairness constraint: max gap < 0.1)
    fair_thresholds = df[max_gaps < 0.1]
    if not fair_thresholds.empty:
        min_accuracies = fair_thresholds.apply(lambda row: min(row['Accuracy_Large'],
                                                             row['Accuracy_Medium'],
                                                             row['Accuracy_Small']), axis=1)
        opt_thresh_fair = fair_thresholds.loc[min_accuracies.idxmax(), 'Threshold']
    else:
        opt_thresh_fair = None
    
    # 3. Find Pareto optimal points (accuracy vs fairness)
    accuracies = df.apply(lambda row: (row['Accuracy_Large'] + 
                                     row['Accuracy_Medium'] + 
                                     row['Accuracy_Small'])/3, axis=1)
    pareto_optimal = []
    for i, row in df.iterrows():
        dominated = False
        for j, other_row in df.iterrows():
            if i != j:
                if (accuracies[j] >= accuracies[i] and max_gaps[j] <= max_gaps[i] and
                    (accuracies[j] > accuracies[i] or max_gaps[j] < max_gaps[i])):
                    dominated = True
                    break
        if not dominated:
            pareto_optimal.append(row['Threshold'])
    
    return {
        'min_gap_threshold': opt_thresh_min_gap,
        'fair_max_acc_threshold': opt_thresh_fair,
        'pareto_optimal_thresholds': pareto_optimal
    }

# Function to analyze group size effects
def analyze_group_size_effects(df):
    size_effects = []
    
    for thresh in df['Threshold'].unique():
        thresh_data = df[df['Threshold'] == thresh].iloc[0]
        
        # Calculate relative performance difference vs relative size difference
        for pair in [('Large', 'Medium'), ('Large', 'Small'), ('Medium', 'Small')]:
            size1, size2 = pair
            
            acc_diff = abs(thresh_data[f'Accuracy_{size1}'] - thresh_data[f'Accuracy_{size2}'])
            size_diff = abs(thresh_data[f'N_{size1}'] - thresh_data[f'N_{size2}'])
            rel_size_diff = size_diff / max(thresh_data[f'N_{size1}'], thresh_data[f'N_{size2}'])
            
            size_effects.append({
                'threshold': thresh,
                'pair': f'{size1}_vs_{size2}',
                'accuracy_diff': acc_diff,
                'relative_size_diff': rel_size_diff,
                'n1': thresh_data[f'N_{size1}'],
                'n2': thresh_data[f'N_{size2}']
            })
    
    return pd.DataFrame(size_effects)

# Create output directory
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "output")
os.makedirs(output_dir, exist_ok=True)

# Analyze optimal thresholds
optimal_thresholds = analyze_optimal_thresholds(threshold_df)
print("\nOptimal Threshold Analysis:")
print(f"1. Threshold minimizing maximum fairness gap: {optimal_thresholds['min_gap_threshold']}")
print(f"2. Threshold maximizing accuracy with fairness constraint: {optimal_thresholds['fair_max_acc_threshold']}")
print(f"3. Pareto optimal thresholds: {optimal_thresholds['pareto_optimal_thresholds']}")

# Analyze group size effects
size_effects_df = analyze_group_size_effects(threshold_df)

# Create enhanced visualizations
def create_enhanced_visualizations(df, size_effects_df, optimal_thresholds, save_dir):
    # Original visualizations
    create_visualizations(df, save_dir)
    
    # Create Pareto frontier plot
    plt.figure(figsize=(10, 6))
    accuracies = df.apply(lambda row: (row['Accuracy_Large'] + 
                                     row['Accuracy_Medium'] + 
                                     row['Accuracy_Small'])/3, axis=1)
    max_gaps = df.apply(lambda row: max(abs(row['Accuracy_Gap_Large_Medium']),
                                      abs(row['Accuracy_Gap_Large_Small']),
                                      abs(row['Accuracy_Gap_Medium_Small'])), axis=1)
    
    # Plot all points
    plt.scatter(max_gaps, accuracies, c='blue', alpha=0.5)
    
    # Highlight Pareto optimal points
    pareto_mask = df['Threshold'].isin(optimal_thresholds['pareto_optimal_thresholds'])
    plt.scatter(max_gaps[pareto_mask], accuracies[pareto_mask], 
                c='red', label='Pareto optimal', zorder=5)
    
    # Add threshold labels
    for thresh in df['Threshold']:
        idx = df['Threshold'] == thresh
        plt.annotate(f'{thresh}', (max_gaps[idx].iloc[0], accuracies[idx].iloc[0]),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Maximum Fairness Gap')
    plt.ylabel('Average Accuracy')
    plt.title('Pareto Frontier of Accuracy vs Fairness')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'pareto_frontier.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create group size effect plot
    plt.figure(figsize=(12, 6))
    for pair in ['Large_vs_Medium', 'Large_vs_Small', 'Medium_vs_Small']:
        pair_data = size_effects_df[size_effects_df['pair'] == pair]
        plt.scatter(pair_data['relative_size_diff'], pair_data['accuracy_diff'],
                   label=pair.replace('_', ' '), alpha=0.7)
        
        # Add trend line
        z = np.polyfit(pair_data['relative_size_diff'], pair_data['accuracy_diff'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(pair_data['relative_size_diff'].min(),
                            pair_data['relative_size_diff'].max(), 100)
        plt.plot(x_trend, p(x_trend), '--', alpha=0.5)
    
    plt.xlabel('Relative Difference in Group Size')
    plt.ylabel('Absolute Difference in Accuracy')
    plt.title('Impact of Group Size Differences on Performance Disparity')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'size_effects.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Create enhanced visualizations
create_enhanced_visualizations(threshold_df, size_effects_df, optimal_thresholds, output_dir)

# Save detailed analysis results
analysis_results = {
    'optimal_thresholds': optimal_thresholds,
    'threshold_recommendations': {
        'balanced': {
            'threshold': optimal_thresholds['fair_max_acc_threshold'],
            'explanation': 'Best balance between accuracy and fairness'
        },
        'fairness_focused': {
            'threshold': optimal_thresholds['min_gap_threshold'],
            'explanation': 'Minimizes performance disparities between groups'
        },
        'pareto_optimal': {
            'thresholds': optimal_thresholds['pareto_optimal_thresholds'],
            'explanation': 'These thresholds represent different optimal trade-offs between accuracy and fairness'
        }
    }
}

# Save analysis results
import json
with open(os.path.join(output_dir, 'threshold_analysis.json'), 'w') as f:
    json.dump(analysis_results, f, indent=2)

# Save size effects analysis
size_effects_df.to_csv(os.path.join(output_dir, 'size_effects_analysis.csv'), index=False)
plt.legend()
plt.grid(True)
plt.show()

# Save the results to a CSV in the output directory
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "output")
os.makedirs(output_dir, exist_ok=True)
threshold_df.to_csv(os.path.join(output_dir, "accuracy_fairness_tradeoff.csv"), index=False)

# Save the plot
plt.savefig(os.path.join(output_dir, "accuracy_fairness_tradeoff.png"), dpi=300, bbox_inches='tight')
plt.close()
