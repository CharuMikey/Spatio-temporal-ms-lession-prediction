
from pathlib import Path

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

patient_dir = Path(r"D:\dataset\Patient-1")

flair_path = patient_dir / "1-Flair.nii"
mask_path = patient_dir / "1-LesionSeg-Flair.nii"

# Load the MRI and its corresponding segmentation
flair_img = nib.load(str(flair_path))
mask_img = nib.load(str(mask_path))

flair = flair_img.get_fdata()
mask = mask_img.get_fdata()

print("FLAIR shape:", flair.shape)
print("Mask shape:", mask.shape)
print("Unique mask values:", np.unique(mask)[:20])

if flair.shape != mask.shape:
    raise ValueError(
        "FLAIR and mask dimensions differ. "
        "They need alignment before overlaying."
    )

# Select the middle slice
slice_idx = flair.shape[2] // 2

mri_slice = flair[:, :, slice_idx]
mask_slice = mask[:, :, slice_idx]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(mri_slice.T, cmap="gray", origin="lower")
axes[0].set_title(f"FLAIR MRI - Slice {slice_idx}")
axes[0].axis("off")

axes[1].imshow(mri_slice.T, cmap="gray", origin="lower")
axes[1].imshow(
    np.ma.masked_where(mask_slice.T == 0, mask_slice.T),
    cmap="autumn",
    alpha=0.65,
    origin="lower"
)
axes[1].set_title("FLAIR with Lesion Mask Overlay")
axes[1].axis("off")

plt.tight_layout()
plt.show()
