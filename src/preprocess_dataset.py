
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from PIL import Image
from tqdm import tqdm


# ---------------- CONFIGURATION ----------------

DATASET_DIR = Path(r"D:\dataset")
OUTPUT_DIR = Path(r"D:\MS diagnosis\processed_data")

IMAGE_SIZE = 224
MODALITIES = ["Flair", "T1", "T2"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------- PREPROCESSING FUNCTIONS ----------------

def normalize_volume(volume):
    """Normalize MRI intensities using robust percentile clipping."""

    volume = np.asarray(volume, dtype=np.float32)
    volume = np.nan_to_num(
        volume, nan=0.0, posinf=0.0, neginf=0.0
    )

    nonzero = volume[volume != 0]

    if nonzero.size == 0:
        raise ValueError("MRI volume contains only zeros.")

    low, high = np.percentile(nonzero, [1, 99])

    if high <= low:
        raise ValueError("MRI intensity range is invalid.")

    volume = np.clip(volume, low, high)
    volume = (volume - low) / (high - low)

    return volume


def resize_image(slice_2d):
    """Resize an MRI slice using bilinear interpolation."""

    image = Image.fromarray(
        (slice_2d * 255).astype(np.uint8)
    )

    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR
    )

    return np.asarray(image, dtype=np.uint8)


def resize_mask(slice_2d):
    """Resize a binary mask without creating new label values."""

    mask = Image.fromarray(
        (slice_2d > 0).astype(np.uint8) * 255
    )

    mask = mask.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.NEAREST
    )

    return (np.asarray(mask) > 0).astype(np.uint8)


# ---------------- MAIN PIPELINE ----------------

def preprocess_patient(patient_dir):
    patient_id = patient_dir.name.split("-")[1]

    flair_path = patient_dir / f"{patient_id}-Flair.nii"
    mask_path = patient_dir / f"{patient_id}-LesionSeg-Flair.nii"

    if not flair_path.exists() or not mask_path.exists():
        raise FileNotFoundError(
            f"Missing FLAIR image or lesion mask for {patient_dir.name}"
        )

    flair_img = nib.load(str(flair_path))
    mask_img = nib.load(str(mask_path))

    # The FLAIR image defines our reference voxel grid.
    reference = (flair_img.shape, flair_img.affine)

    # Resample the FLAIR lesion mask if its grid differs.
    if (
        mask_img.shape != flair_img.shape
        or not np.allclose(mask_img.affine, flair_img.affine, atol=1e-3)
    ):
        mask_img = resample_from_to(
            mask_img, reference, order=0
        )

    mask_volume = np.asarray(
        mask_img.dataobj, dtype=np.float32
    )
    mask_volume = (mask_volume > 0).astype(np.uint8)

    image_volumes = []

    for modality in MODALITIES:
        image_path = patient_dir / f"{patient_id}-{modality}.nii"

        if not image_path.exists():
            raise FileNotFoundError(
                f"Missing {modality} for {patient_dir.name}"
            )

        image = nib.load(str(image_path))

        # Put each modality on the FLAIR voxel grid.
        # This resamples the grid; it does NOT register anatomy.
        if (
            image.shape != flair_img.shape
            or not np.allclose(image.affine, flair_img.affine, atol=1e-3)
        ):
            image = resample_from_to(
                image, reference, order=1
            )

        volume = normalize_volume(
            np.asarray(image.dataobj, dtype=np.float32)
        )
        image_volumes.append(volume)

    # Stack modalities as three input channels.
    combined = np.stack(image_volumes, axis=-1)

    # NIfTI arrays are (X, Y, Z, channels).
    # Prepare ordered slices along the third axis.
    num_slices = combined.shape[2]

    images_out = np.zeros(
        (num_slices, IMAGE_SIZE, IMAGE_SIZE, 3),
        dtype=np.uint8
    )

    masks_out = np.zeros(
        (num_slices, IMAGE_SIZE, IMAGE_SIZE),
        dtype=np.uint8
    )

    for z in range(num_slices):
        for channel in range(3):
            images_out[z, :, :, channel] = resize_image(
                combined[:, :, z, channel]
            )

        masks_out[z] = resize_mask(mask_volume[:, :, z])

    output_path = OUTPUT_DIR / f"Patient-{patient_id}.npz"

    np.savez_compressed(
        output_path,
        images=images_out,
        masks=masks_out,
        patient_id=patient_id
    )

    positive_slices = int(
        np.any(masks_out > 0, axis=(1, 2)).sum()
    )

    return num_slices, positive_slices, output_path


def main():
    patients = sorted(
        DATASET_DIR.glob("Patient-*"),
        key=lambda p: int(p.name.split("-")[1])
    )

    if not patients:
        raise RuntimeError(
            f"No patient folders found in {DATASET_DIR}"
        )

    successful = 0
    failed = []

    total_slices = 0
    total_positive_slices = 0

    print(f"Patients found: {len(patients)}")
    print(f"Output folder: {OUTPUT_DIR}\n")

    for patient_dir in tqdm(patients, desc="Preprocessing"):
        try:
            slices, positive, output_path = preprocess_patient(
                patient_dir
            )

            successful += 1
            total_slices += slices
            total_positive_slices += positive

            print(
                f"\n{patient_dir.name}: "
                f"{slices} slices, "
                f"{positive} lesion-positive slices"
            )

        except Exception as error:
            failed.append((patient_dir.name, str(error)))
            print(f"\nERROR - {patient_dir.name}: {error}")

    print("\n========== PREPROCESSING SUMMARY ==========")
    print(f"Successfully processed: {successful}/{len(patients)}")
    print(f"Total slices: {total_slices}")
    print(f"Lesion-positive slices: {total_positive_slices}")
    print(f"Failed patients: {len(failed)}")

    if failed:
        for patient, error in failed:
            print(f"- {patient}: {error}")
    else:
        print("All patients processed successfully.")

    print(f"Saved files to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
