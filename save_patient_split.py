
from pathlib import Path
import csv
import json

from sklearn.model_selection import train_test_split


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(r"D:\MS diagnosis")
DATA_DIR = PROJECT_DIR / "processed_data"
OUTPUT_DIR = PROJECT_DIR / "evaluation_results"

RANDOM_SEED = 42

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_DIR / "patient_split.csv"
JSON_PATH = OUTPUT_DIR / "patient_split_summary.json"


# ============================================================
# 2. LOAD PATIENT FILES IN THE SAME ORDER AS train.py
# ============================================================

patient_files = sorted(
    DATA_DIR.glob("Patient-*.npz"),
    key=lambda path: int(path.stem.split("-")[1]),
)

if len(patient_files) < 3:
    raise RuntimeError(
        f"Expected at least 3 patient files in {DATA_DIR}, "
        f"but found {len(patient_files)}."
    )

patient_ids = [path.stem for path in patient_files]

if len(patient_ids) != len(set(patient_ids)):
    raise RuntimeError("Duplicate patient IDs were found.")


# ============================================================
# 3. RECONSTRUCT THE ORIGINAL TRAINING SPLIT
# ============================================================

# This matches the first split in train.py.
train_val_files, test_files = train_test_split(
    patient_files,
    test_size=0.20,
    random_state=RANDOM_SEED,
)

# This matches the second split in train.py.
train_files, val_files = train_test_split(
    train_val_files,
    test_size=0.20,
    random_state=RANDOM_SEED,
)


# ============================================================
# 4. VERIFY THAT PATIENTS DO NOT CROSS SPLITS
# ============================================================

train_ids = {path.stem for path in train_files}
val_ids = {path.stem for path in val_files}
test_ids = {path.stem for path in test_files}

assert train_ids.isdisjoint(val_ids), (
    "Patient leakage detected between training and validation."
)

assert train_ids.isdisjoint(test_ids), (
    "Patient leakage detected between training and testing."
)

assert val_ids.isdisjoint(test_ids), (
    "Patient leakage detected between validation and testing."
)

assert (
    len(train_ids | val_ids | test_ids) == len(patient_files)
), "Some patients are missing from the split."

assert (
    len(train_ids) + len(val_ids) + len(test_ids)
    == len(patient_files)
), "Patient counts do not match the dataset."


# ============================================================
# 5. SAVE PATIENT ASSIGNMENTS TO CSV
# ============================================================

split_assignments = {}

for patient_file in train_files:
    split_assignments[patient_file.stem] = "train"

for patient_file in val_files:
    split_assignments[patient_file.stem] = "validation"

for patient_file in test_files:
    split_assignments[patient_file.stem] = "test"

with open(
    CSV_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as csv_file:

    writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "patient_id",
            "split",
            "random_seed",
            "data_file",
        ],
    )

    writer.writeheader()

    for patient_file in patient_files:
        writer.writerow(
            {
                "patient_id": patient_file.stem,
                "split": split_assignments[patient_file.stem],
                "random_seed": RANDOM_SEED,
                "data_file": patient_file.name,
            }
        )


# ============================================================
# 6. SAVE SPLIT SUMMARY
# ============================================================

summary = {
    "project": (
        "Spatio-Temporal Attention Mechanism "
        "for Multiple Sclerosis Diagnosis"
    ),
    "random_seed": RANDOM_SEED,
    "split_method": "Patient-level train_test_split",
    "dataset_directory": str(DATA_DIR),
    "total_patients": len(patient_files),
    "training_patients": len(train_ids),
    "validation_patients": len(val_ids),
    "test_patients": len(test_ids),
    "training_patient_ids": sorted(
        train_ids,
        key=lambda value: int(value.split("-")[1]),
    ),
    "validation_patient_ids": sorted(
        val_ids,
        key=lambda value: int(value.split("-")[1]),
    ),
    "test_patient_ids": sorted(
        test_ids,
        key=lambda value: int(value.split("-")[1]),
    ),
    "patient_overlap_check": "PASSED",
    "note": (
        "Split reconstructed from the deterministic procedure "
        "in train.py. It is reproducible under the same file "
        "ordering and compatible split implementation, but the "
        "checkpoint alone does not prove these were its original "
        "training patients."
    ),
}

with open(
    JSON_PATH,
    "w",
    encoding="utf-8",
) as json_file:
    json.dump(
        summary,
        json_file,
        indent=4,
    )


# ============================================================
# 7. PRINT VERIFICATION RESULTS
# ============================================================

print("\n========== PATIENT SPLIT RECORD ==========")
print(f"Total patients:      {len(patient_files)}")
print(f"Training patients:   {len(train_ids)}")
print(f"Validation patients: {len(val_ids)}")
print(f"Test patients:       {len(test_ids)}")

print("\nPatient overlap checks: PASSED")

print("\nTraining patient IDs:")
print(", ".join(sorted(
    train_ids,
    key=lambda value: int(value.split("-")[1]),
)))

print("\nValidation patient IDs:")
print(", ".join(sorted(
    val_ids,
    key=lambda value: int(value.split("-")[1]),
)))

print("\nTest patient IDs:")
print(", ".join(sorted(
    test_ids,
    key=lambda value: int(value.split("-")[1]),
)))

print("\nSaved files:")
print(CSV_PATH)
print(JSON_PATH)

print("\nPatient split record created successfully.")

