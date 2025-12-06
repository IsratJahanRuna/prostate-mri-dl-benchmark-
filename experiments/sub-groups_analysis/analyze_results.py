import pandas as pd

# Read the CSV file
df = pd.read_csv('patient_tumor_analysis_results.csv')

# Get basic statistics
total_patients = len(df)
patients_with_tumors = len(df[df['Total_Tumor_Area_1'] > 0])

# Print summary
print(f"Total number of patients analyzed: {total_patients}")
print(f"Number of patients with detected tumors: {patients_with_tumors}")
print(f"Percentage of patients with tumors: {(patients_with_tumors/total_patients)*100:.1f}%")

print("\nTumor size distribution (Reader 1):")
print(df['Tumor_Size_1'].value_counts())

print("\nLesion zones distribution (Reader 1):")
print(df['Lesion_Zones_1'].value_counts())

print("\nAgreement between readers:")
agreement = (df['Total_Tumor_Area_1'] > 0) == (df['Total_Tumor_Area_2'] > 0)
print(f"Reader agreement on tumor presence: {(agreement.mean())*100:.1f}%")

# For cases with tumors, compare areas
tumor_cases = df[df['Total_Tumor_Area_1'] > 0]
if len(tumor_cases) > 0:
    print("\nFor cases with tumors:")
    print("Average tumor area (Reader 1):", tumor_cases['Total_Tumor_Area_1'].mean())
    print("Average tumor area (Reader 2):", tumor_cases['Total_Tumor_Area_2'].mean())