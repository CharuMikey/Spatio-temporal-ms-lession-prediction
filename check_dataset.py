
from pathlib import Path
import nibabel as nib
import numpy as np

root = Path(r"D:\dataset")

patient_dirs = sorted(
    root.glob("Patient-*"),
    key=lambda p: int(p.name.split("-")[1])
)

print("Patient folders found:", len(patient_dirs))

usable = 0
missing = []
total_slices = 0
positive_slices = 0
total_lesion_voxels = 0

for patient in patient_dirs:
    flair_path = patient / f"{patient.name.split('-')[1]}-Flair.nii"
    mask_path = patient / (
        f"{patient.name.split('-')[1]}-LesionSeg-Flair.nii"
    )

    if not flair_path.exists() or not mask_path.exists():
        missing.append(patient.name)
        continue

    flair_img = nib.load(str(flair_path))
    mask_img = nib.load(str(mask_path))

    if flair_img.shape != mask_img.shape:
        print(f"Shape mismatch: {patient.name}")
        continue

    mask = np.asarray(mask_img.dataobj)
    lesion_slices = np.any(mask > 0, axis=(0, 1))

    usable += 1
    total_slices += mask.shape[2]
    positive_slices += int(np.sum(lesion_slices))
    total_lesion_voxels += int(np.sum(mask > 0))

    print(
        f"{patient.name}: shape={mask.shape}, "
        f"lesion-positive slices={np.sum(lesion_slices)}"
    )

print("\n--- SUMMARY ---")
print("Usable patients:", usable)
print("Missing FLAIR or mask:", missing)
print("Total slices:", total_slices)
print("Lesion-positive slices:", positive_slices)
print("Total positive mask voxels:", total_lesion_voxels)
