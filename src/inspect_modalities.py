
from pathlib import Path
import nibabel as nib
import numpy as np

ROOT = Path(r"D:\dataset")
modalities = ["Flair", "T1", "T2"]

problems = []
patients_checked = 0

for folder in sorted(
    ROOT.glob("Patient-*"),
    key=lambda p: int(p.name.split("-")[1])
):
    pid = folder.name.split("-")[1]
    images = {}

    for modality in modalities:
        image_path = folder / f"{pid}-{modality}.nii"
        mask_path = folder / f"{pid}-LesionSeg-{modality}.nii"

        if not image_path.exists():
            problems.append(f"{folder.name}: missing {image_path.name}")
            continue

        img = nib.load(str(image_path))
        images[modality] = img

        print(
            f"{folder.name} {modality}: "
            f"shape={img.shape}, "
            f"spacing={np.round(nib.affines.voxel_sizes(img.affine), 3)}, "
            f"orientation={nib.aff2axcodes(img.affine)}"
        )

        if mask_path.exists():
            mask_img = nib.load(str(mask_path))
            same_shape = img.shape == mask_img.shape
            same_affine = np.allclose(
                img.affine, mask_img.affine, atol=1e-3
            )

            mask_data = np.asarray(mask_img.dataobj)
            mask_values = np.unique(mask_data)

            print(
                f"  Mask: shape_match={same_shape}, "
                f"affine_match={same_affine}, "
                f"values={mask_values[:10]}"
            )

            if not same_shape or not same_affine:
                problems.append(
                    f"{folder.name} {modality}: "
                    f"mask alignment metadata differs"
                )
        else:
            problems.append(
                f"{folder.name}: missing {mask_path.name}"
            )

    if len(images) == 3:
        patients_checked += 1

        reference = images["Flair"]
        for modality in ["T1", "T2"]:
            img = images[modality]
            if (
                img.shape != reference.shape
                or not np.allclose(
                    img.affine, reference.affine, atol=1e-3
                )
            ):
                print(
                    f"  NOTE: {modality} and FLAIR "
                    "differ in shape or spatial metadata"
                )

print("\n--- FINAL SUMMARY ---")
print("Patients with all three modalities:", patients_checked)
print("Issues requiring review:", len(problems))

for problem in problems:
    print("-", problem)

if not problems:
    print("All required files and mask metadata checks passed.")

