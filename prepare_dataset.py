
from pathlib import Path
import nibabel as nib
import numpy as np

DATASET_DIR = Path(r"D:\dataset")

usable = 0
missing = []
shape_mismatches = []
invalid_data = []
total_slices = 0
total_lesion_voxels = 0

for patient_dir in sorted(
    DATASET_DIR.glob("Patient-*"),
    key=lambda p: int(p.name.split("-")[1])
):
    patient_id = patient_dir.name.split("-")[1]

    flair_path = patient_dir / f"{patient_id}-Flair.nii"
    mask_path = patient_dir / f"{patient_id}-LesionSeg-Flair.nii"

    if not flair_path.exists() or not mask_path.exists():
        missing.append(patient_dir.name)
        continue

    flair_img = nib.load(str(flair_path))
    mask_img = nib.load(str(mask_path))

    if flair_img.shape != mask_img.shape:
        shape_mismatches.append(patient_dir.name)
        continue

    flair = flair_img.get_fdata(dtype=np.float32)
    mask = mask_img.get_fdata(dtype=np.float32)

    if not np.isfinite(flair).all() or not np.isfinite(mask).all():
        invalid_data.append(patient_dir.name)
        continue

    lesion_voxels = int(np.count_nonzero(mask > 0))

    total_lesion_voxels += lesion_voxels
    total_slices += flair.shape[2]
    usable += 1

    print(
        f"{patient_dir.name}: "
        f"shape={flair.shape}, "
        f"lesion_voxels={lesion_voxels}"
    )

print("\n--- DATASET VALIDATION SUMMARY ---")
print("Usable patients:", usable)
print("Missing files:", missing)
print("Shape mismatches:", shape_mismatches)
print("Invalid data:", invalid_data)
print("Total slices:", total_slices)
print("Positive mask voxels:", total_lesion_voxels)
