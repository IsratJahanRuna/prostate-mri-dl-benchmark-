import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.utils import resample
import os
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.fairness_metrics import specificity_score

# Define the base directory where all the model folders are located
base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "input")

# Step 1: Load and Merge the Data

# Function to load predictions and merge with tumor data
def load_and_merge_data(model_name, dataset_name):
    # Construct the file path
    file_path = os.path.join(base_dir, model_name, dataset_name, "test_patient_preds.csv")
    
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist!")
        return pd.DataFrame()  # Return empty DataFrame if file doesn't exist

    preds = pd.read_csv(file_path)
    preds['model'] = model_name
    preds['dataset'] = dataset_name
    # Convert patient_id columns to integers for proper merging
    preds['patient_id'] = preds['patient_id'].astype(int)
    tumor_data['Patient_ID'] = tumor_data['Patient_ID'].astype(int)
    # Merge with tumor data
    merged_data = pd.merge(preds, tumor_data, left_on='patient_id', right_on='Patient_ID', how='left')
    return merged_data

# Load the tumor data
tumor_data = pd.read_csv("patient_tumor_analysis_results.csv")

# List of models and datasets
models = ['densenet', 'resnet', 'efficientnet']
datasets = ['T2', 'ADC', 'DWI']

# Merge all prediction files into one DataFrame
merged_data = pd.DataFrame()
for model_name in models:
    for dataset_name in datasets:
        merged_data = pd.concat([merged_data, load_and_merge_data(model_name, dataset_name)])

# Step 2: Define Subgroups

def define_subgroups(row):
    # Lesion Zones: Combine Lesion_Zones_1 and Lesion_Zones_2
    if row['Lesion_Zones_1'] != row['Lesion_Zones_2']:
        row['Lesion_Zones'] = 'Mixed'
    else:
        row['Lesion_Zones'] = row['Lesion_Zones_1']
    
    # Tumor Size: Combine Tumor_Size_1 and Tumor_Size_2
    if row['Tumor_Size_1'] != row['Tumor_Size_2']:
        row['Tumor_Size'] = 'Mixed'
    else:
        row['Tumor_Size'] = row['Tumor_Size_1']
    
    # Lesion Count: Combine Lesion_Count_1 and Lesion_Count_2
    if row['Lesion_Count_1'] != row['Lesion_Count_2']:
        row['Lesion_Count'] = 'Mixed'
    else:
        row['Lesion_Count'] = row['Lesion_Count_1']
    
    return row

# Apply the function to define subgroups
merged_data = merged_data.apply(define_subgroups, axis=1)

# Step 3: Calculate Metrics for Each Subgroup

def calculate_metrics(subgroup_data):
    if len(subgroup_data) == 0:
        return 0, 0, 0  # Return zeros for empty subgroups
    
    # Drop any rows with missing values
    subgroup_data = subgroup_data.dropna(subset=['y_true', 'y_hat'])
    
    if len(subgroup_data) == 0:
        return 0, 0, 0  # Return zeros if no valid data after dropping NA
    
    y_true = subgroup_data['y_true']
    y_pred = subgroup_data['y_hat']
    
    # Calculate metrics with proper handling of edge cases
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred, zero_division=0, labels=[0, 1])
    
    # Calculate confusion matrix with explicit labels
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return accuracy, sensitivity, specificity

# Step 4: Compute Fairness Gap

def compute_fairness_gap(metrics_subgroup1, metrics_subgroup2):
    accuracy_gap = metrics_subgroup1[0] - metrics_subgroup2[0]
    sensitivity_gap = metrics_subgroup1[1] - metrics_subgroup2[1]
    specificity_gap = metrics_subgroup1[2] - metrics_subgroup2[2]
    
    return accuracy_gap, sensitivity_gap, specificity_gap

# Step 5: Bootstrap Confidence Intervals (CIs)

def bootstrap_confidence_intervals(data, metric_func, n_iterations=1000, ci=95):
    stats = []
    for _ in range(n_iterations):
        sample = resample(data, replace=True, n_samples=len(data))
        stats.append(metric_func(sample))
    lower = np.percentile(stats, (100-ci)/2)
    upper = np.percentile(stats, 100-(100-ci)/2)
    return lower, upper

# Analyze subgroups for fairness
subgroup_names = ['Lesion_Zones', 'Tumor_Size', 'Lesion_Count']
fairness_results = []

for subgroup in subgroup_names:
    # Get unique values in the subgroup
    unique_values = sorted([str(x) for x in merged_data[subgroup].unique()])  # Convert all values to strings
    print(f"\nAnalyzing {subgroup} - Unique values found: {unique_values}")
    
    # For each pair of values in the subgroup
    for i, val1 in enumerate(unique_values):
        for val2 in unique_values[i+1:]:  # Only compare with values after val1
            try:
                # Convert values to original type for comparison
                orig_val1 = merged_data[subgroup][merged_data[subgroup].astype(str) == val1].iloc[0]
                orig_val2 = merged_data[subgroup][merged_data[subgroup].astype(str) == val2].iloc[0]
                
                subgroup1 = merged_data[merged_data[subgroup].astype(str) == val1]
                subgroup2 = merged_data[merged_data[subgroup].astype(str) == val2]
                
                # Skip if either subgroup is too small
                if len(subgroup1) < 5 or len(subgroup2) < 5:
                    print(f"Skipping comparison of {val1} vs {val2} - insufficient data")
                    continue
                
                print(f"Comparing {val1} vs {val2}")
                print(f"Group sizes: {len(subgroup1)} vs {len(subgroup2)}")
                
                # Compute metrics for each subgroup
                metrics_subgroup1 = calculate_metrics(subgroup1)
                metrics_subgroup2 = calculate_metrics(subgroup2)
                
                # Compute fairness gap
                fairness_gap = compute_fairness_gap(metrics_subgroup1, metrics_subgroup2)
                
                try:
                    # Bootstrap CIs for metrics with full label set
                    def safe_accuracy(x):
                        return accuracy_score(x['y_true'], x['y_hat'])
                    
                    def safe_recall(x):
                        return recall_score(x['y_true'], x['y_hat'], zero_division=0, labels=[0, 1])
                    
                    def safe_specificity(x):
                        return specificity_score(x['y_true'], x['y_hat'])
                    
                    lower_acc, upper_acc = bootstrap_confidence_intervals(subgroup1, safe_accuracy)
                    lower_sens, upper_sens = bootstrap_confidence_intervals(subgroup1, safe_recall)
                    lower_spec, upper_spec = bootstrap_confidence_intervals(subgroup1, safe_specificity)
                except Exception as e:
                    print(f"Warning: Could not compute confidence intervals for {subgroup} {val1}: {str(e)}")
                    lower_acc = upper_acc = lower_sens = upper_sens = lower_spec = upper_spec = None
                
                fairness_results.append({
                    'subgroup': subgroup,
                    'value1': str(val1),
                    'value2': str(val2),
                    'size1': len(subgroup1),
                    'size2': len(subgroup2),
                    'fairness_gap_accuracy': fairness_gap[0],
                    'fairness_gap_sensitivity': fairness_gap[1],
                    'fairness_gap_specificity': fairness_gap[2],
                    'accuracy_confidence_interval': (lower_acc, upper_acc),
                    'sensitivity_confidence_interval': (lower_sens, upper_sens),
                    'specificity_confidence_interval': (lower_spec, upper_spec)
                })
            except Exception as e:
                print(f"Warning: Error comparing {val1} vs {val2}: {str(e)}")

# Output the results as a DataFrame
fairness_df = pd.DataFrame(fairness_results)
print(fairness_df)

# Save the results to a CSV file
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "output")
os.makedirs(output_dir, exist_ok=True)
fairness_df.to_csv(os.path.join(output_dir, "fairness_analysis_results.csv"), index=False)
