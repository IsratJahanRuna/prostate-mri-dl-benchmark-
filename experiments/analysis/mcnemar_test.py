import pandas as pd
import argparse
import os
from statsmodels.stats.contingency_tables import mcnemar

# Function to perform McNemar's test
def mcnemar_test(model1_preds, model2_preds):
    # Contingency table
    table = pd.crosstab(model1_preds, model2_preds)
    result = mcnemar(table, exact=True)
    return result

def main():
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description="McNemar's Test for statistical significance between two models")

    # Input and output folder structure arguments
    parser.add_argument('--input_folder', required=True, type=str, help="Input folder path inside 'input'")
    parser.add_argument('--model1', required=True, type=str, choices=['resnet', 'densenet', 'efficientnet'], help="First model folder")
    parser.add_argument('--model2', required=True, type=str, choices=['resnet', 'densenet', 'efficientnet'], help="Second model folder")
    parser.add_argument('--dataset', required=True, type=str, choices=['ADC', 'DWI', 'T2'], help="Dataset folder (ADC, DWI, T2)")
    parser.add_argument('--output_folder', required=True, type=str, help="Output folder path inside 'output'")

    # Parse arguments
    args = parser.parse_args()

    # Construct input and output file paths based on the folder structure
    model1_preds_file = os.path.join(args.input_folder, args.model1, args.dataset, 'test_patient_preds.csv')
    model2_preds_file = os.path.join(args.input_folder, args.model2, args.dataset, 'test_patient_preds.csv')
    output_file = os.path.join(args.output_folder, args.model1 + "_vs_" + args.model2, args.dataset, 'mcnemar_results.txt')

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load the prediction data from both models
    model1_preds = pd.read_csv(model1_preds_file)['y_hat']
    model2_preds = pd.read_csv(model2_preds_file)['y_hat']

    # Perform McNemar's Test
    result = mcnemar_test(model1_preds, model2_preds)

    # Prepare the results as a string
    output = f"McNemar's Test p-value: {result.pvalue}"

    # Save the output to the specified output file
    with open(output_file, 'w') as f:
        f.write(output)

    print(f"Results saved to {output_file}")
    print(output)

if __name__ == '__main__':
    main()
