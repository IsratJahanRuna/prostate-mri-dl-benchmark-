import os
import nibabel as nib
import numpy as np
from skimage.measure import label, regionprops
import pandas as pd

# Function to process tumor annotation and calculate size, lesion count, and zone classification
def process_tumor_file(file_path):
    print(f"\nProcessing file: {file_path}")
    # Load the tumor annotation NIfTI file
    tumor_data = nib.load(file_path)
    tumor_image = tumor_data.get_fdata()
    
    print(f"Image shape: {tumor_image.shape}")
    print(f"Value range: [{tumor_image.min()}, {tumor_image.max()}]")

    # Threshold the tumor annotation to create a binary mask (tumor vs non-tumor) across all slices
    tumor_mask = tumor_image > 0.5  # Threshold value can be adjusted
    print(f"Checking all {tumor_image.shape[2]} slices")
    print(f"Number of pixels above threshold: {np.sum(tumor_mask)}")

    # Label connected components (lesions)
    labeled_mask = label(tumor_mask)

    # Calculate the area of each lesion (connected region)
    regions = regionprops(labeled_mask)
    total_tumor_area = sum([region.area for region in regions])

    # Classify tumor size based on area
    if total_tumor_area < 1000:
        tumor_size_class = "Small"
    elif 1000 <= total_tumor_area <= 5000:
        tumor_size_class = "Medium"
    else:
        tumor_size_class = "Large"

    # Count the number of lesions
    lesion_count = len(regions)

    # Classify lesion count as few or many
    if lesion_count < 3:
        lesion_class = "Few lesions"
    else:
        lesion_class = "Many lesions"

    # Classify lesion zones (PZ, CG, Mixed) based on position in the image
    lesion_zones = []
    for region in regions:
        # Classify based on position: top half = PZ, bottom half = CG, mixed if in both
        min_row, min_col = region.bbox[0], region.bbox[1]
        max_row, max_col = region.bbox[2], region.bbox[3]
        
        # Assume image height (rows) is 270, divide into two halves for PZ and CG
        if max_row < 135:  # Top half (Peripheral Zone)
            lesion_zones.append('PZ')
        elif min_row >= 135:  # Bottom half (Central Gland)
            lesion_zones.append('CG')
        else:  # Mixed
            lesion_zones.append('Mixed')

    # Return the results (Tumor Size, Lesion Count, Lesion Class, Tumor Area, Lesion Zones)
    return tumor_size_class, lesion_count, lesion_class, total_tumor_area, lesion_zones

# Path to the directory containing all patient data
base_dir = r'D:\thesis\I_will_work_with\prostate158_train\prostate158_train\train'

# Prepare a list to store results for all patients
results = []

# Iterate through patient folders (020 to 158)
print("Starting patient analysis...")
for patient_id in range(20, 159):  # Patient IDs from 020 to 158
    patient_folder = f"{patient_id:03d}"
    patient_path = os.path.join(base_dir, patient_folder)
    
    # Print diagnostic information
    print(f"\nChecking patient {patient_folder}...")
    print(f"Looking in path: {patient_path}")

    # Check if tumor annotation files exist for this patient
    tumor_file1 = os.path.join(patient_path, 'adc_tumor_reader1.nii.gz')
    tumor_file2 = os.path.join(patient_path, 'adc_tumor_reader2.nii.gz')

    # Print file existence status
    print(f"Reader 1 file exists: {os.path.exists(tumor_file1)}")
    print(f"Reader 2 file exists: {os.path.exists(tumor_file2)}")

    # If both files exist, process them
    if os.path.exists(tumor_file1) and os.path.exists(tumor_file2):
        results1 = process_tumor_file(tumor_file1)
        results2 = process_tumor_file(tumor_file2)

        # Append the results to the list (using patient ID)
        results.append({
            'Patient_ID': patient_id,
            'Tumor_Size_1': results1[0],
            'Lesion_Count_1': results1[1],
            'Lesion_Class_1': results1[2],
            'Total_Tumor_Area_1': results1[3],
            'Lesion_Zones_1': ', '.join(results1[4]),
            'Tumor_Size_2': results2[0],
            'Lesion_Count_2': results2[1],
            'Lesion_Class_2': results2[2],
            'Total_Tumor_Area_2': results2[3],
            'Lesion_Zones_2': ', '.join(results2[4]),
        })

# Convert the results to a DataFrame
df_results = pd.DataFrame(results)

# Save the results to a CSV file
output_csv = r'C:\Users\User\OneDrive\Documents\Thesis_Analysis_3_to_6\patient_tumor_analysis_results.csv'
df_results.to_csv(output_csv, index=False)

print(f"Results saved to {output_csv}")
