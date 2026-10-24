
from pathlib import Path
import nibabel as nib
import numpy as np

patient_dir = Path(r"D:\dataset\Patient-1")

files = sorted(patient_dir.glob("*.nii"))

if not files:
    print("No .nii files found. Check the dataset path.")
else:
    for file in files:
        print(f"\n{'=' * 50}")
        print("File:", file.name)

        image = nib.load(str(file))
        data = image.get_fdata()

        print("Shape:", image.shape)
        print("Voxel spacing (mm):",
              nib.affines.voxel_sizes(image.affine))
        print("Data type:", image.get_data_dtype())
        print("Value range:",
              float(np.min(data)), "to", float(np.max(data)))

        if "lesionseg" in file.name.lower():
            values = np.unique(data)
            print("Unique mask values (first 20):", values[:20])
            print("Number of unique values:", len(values))

print("\nInspection complete.")

