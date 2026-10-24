
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# ---------------- CONFIGURATION ----------------

DATA_DIR = Path(r"D:\MS diagnosis\processed_data")

SEQUENCE_LENGTH = 5
BATCH_SIZE = 4
RANDOM_SEED = 42


# ---------------- DATASET ----------------

class MRIDataset(Dataset):

    def __init__(self, patient_files, sequence_length=5):
        self.samples = []
        self.sequence_length = sequence_length

        for file_path in patient_files:

            with np.load(file_path) as data:
                images = data["images"]
                masks = data["masks"]
                patient_id = str(data["patient_id"].item())

            # Verify image and mask dimensions.
            if len(images) != len(masks):
                raise ValueError(
                    f"Image/mask slice count mismatch: {file_path.name}"
                )

            # Label each slice according to its lesion mask.
            slice_labels = np.any(
                masks > 0, axis=(1, 2)
            ).astype(np.uint8)

            # Create consecutive, ordered slice sequences.
            for start in range(
                len(images) - sequence_length + 1
            ):
                end = start + sequence_length

                sequence = images[start:end]

                # Sequence label = 1 if any slice has a lesion.
                sequence_label = float(
                    np.any(slice_labels[start:end] > 0)
                )

                self.samples.append(
                    (
                        sequence.copy(),
                        sequence_label,
                        patient_id,
                        start
                    )
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        images, label, patient_id, start_slice = (
            self.samples[index]
        )

        # [time, height, width, channels]
        # becomes [time, channels, height, width].
        images = torch.from_numpy(
            images.copy()
        ).permute(0, 3, 1, 2).float() / 255.0

        label = torch.tensor(
            label, dtype=torch.float32
        )

        return {
            "images": images,
            "labels": label,
            "patient_id": patient_id,
            "start_slice": start_slice
        }


# ---------------- PATIENT-LEVEL SPLIT ----------------

def main():

    files = sorted(
        DATA_DIR.glob("Patient-*.npz"),
        key=lambda p: int(p.stem.split("-")[1])
    )

    if len(files) < 3:
        raise RuntimeError(
            f"Expected patient files in {DATA_DIR}; "
            f"found {len(files)}"
        )

    # 80% development, 20% testing.
    train_val_files, test_files = train_test_split(
        files,
        test_size=0.20,
        random_state=RANDOM_SEED
    )

    # Split the development set into training and validation.
    train_files, val_files = train_test_split(
        train_val_files,
        test_size=0.20,
        random_state=RANDOM_SEED
    )

    print("========== PATIENT SPLIT ==========")
    print(f"Training patients:   {len(train_files)}")
    print(f"Validation patients: {len(val_files)}")
    print(f"Testing patients:    {len(test_files)}")

    # Verify patient-level separation.
    train_ids = {p.stem for p in train_files}
    val_ids = {p.stem for p in val_files}
    test_ids = {p.stem for p in test_files}

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    # Build datasets only after splitting patients.
    train_dataset = MRIDataset(
        train_files, SEQUENCE_LENGTH
    )

    val_dataset = MRIDataset(
        val_files, SEQUENCE_LENGTH
    )

    test_dataset = MRIDataset(
        test_files, SEQUENCE_LENGTH
    )

    print("\n========== SEQUENCE SUMMARY ==========")
    print(f"Training sequences:   {len(train_dataset)}")
    print(f"Validation sequences: {len(val_dataset)}")
    print(f"Testing sequences:    {len(test_dataset)}")

    if len(train_dataset) == 0:
        raise RuntimeError("No training sequences generated.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # Inspect a batch.
    batch = next(iter(train_loader))

    print("\n========== BATCH CHECK ==========")
    print(
        "Image tensor shape:",
        tuple(batch["images"].shape)
    )

    print(
        "Label tensor shape:",
        tuple(batch["labels"].shape)
    )

    print(
        "Example patient IDs:",
        batch["patient_id"][:2]
    )

    print(
        "Example sequence labels:",
        batch["labels"][:4].tolist()
    )

    print("\nDataset loader is working.")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    main()

