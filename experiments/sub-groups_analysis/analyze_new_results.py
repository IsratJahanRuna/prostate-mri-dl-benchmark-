import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('patient_tumor_analysis_results.csv')

# Get basic statistics
total_patients = len(df)

print(f"Total number of patients analyzed: {total_patients}")
print("\nTumor size distribution (Reader 1):")
print(df['Tumor_Size_1'].value_counts())
print("\nTumor size distribution (Reader 2):")
print(df['Tumor_Size_2'].value_counts())

print("\nLesion count statistics:")
print("Reader 1 average lesions per patient:", df['Lesion_Count_1'].mean())
print("Reader 2 average lesions per patient:", df['Lesion_Count_2'].mean())

print("\nLesion class distribution (Reader 1):")
print(df['Lesion_Class_1'].value_counts())
print("\nLesion class distribution (Reader 2):")
print(df['Lesion_Class_2'].value_counts())

print("\nTumor area statistics:")
print("Reader 1:")
print(f"Mean area: {df['Total_Tumor_Area_1'].mean():.1f}")
print(f"Median area: {df['Total_Tumor_Area_1'].median():.1f}")
print(f"Min area: {df['Total_Tumor_Area_1'].min():.1f}")
print(f"Max area: {df['Total_Tumor_Area_1'].max():.1f}")

print("\nReader 2:")
print(f"Mean area: {df['Total_Tumor_Area_2'].mean():.1f}")
print(f"Median area: {df['Total_Tumor_Area_2'].median():.1f}")
print(f"Min area: {df['Total_Tumor_Area_2'].min():.1f}")
print(f"Max area: {df['Total_Tumor_Area_2'].max():.1f}")

# Calculate agreement between readers on tumor size classification
size_agreement = (df['Tumor_Size_1'] == df['Tumor_Size_2']).mean() * 100
print(f"\nReader agreement on tumor size classification: {size_agreement:.1f}%")