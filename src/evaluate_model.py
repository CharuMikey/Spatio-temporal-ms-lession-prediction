
import os
import csv
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)
import matplotlib.pyplot as plt

from src.model import SpatioTemporalMSModel
from src.dataset_loader import MRIDataset


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = r"D:\MS diagnosis"
DATA_DIR = os.path.join(PROJECT_DIR, "processed_data")
CHECKPOINT_PATH = os.path.join(
    PROJECT_DIR, "checkpoints", "best_model.pth"
)
OUTPUT_DIR = os.path.join(
    PROJECT_DIR, "evaluation_results"
)

SEQUENCE_LENGTH = 5
BATCH_SIZE = 4
RANDOM_STATE = 42
THRESHOLD = 0.5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD PATIENT FILES AND CREATE PATIENT-LEVEL SPLIT
# ============================================================

patient_files = sorted(
    [
        os.path.join(DATA_DIR, filename)
        for filename in os.listdir(DATA_DIR)
        if filename.startswith("Patient-")
        and filename.endswith(".npz")
    ],
    key=lambda path: int(
        os.path.basename(path)
        .replace("Patient-", "")
        .replace(".npz", "")
    ),
)

if len(patient_files) != 60:
    print(
        f"Warning: expected 60 patient files, "
        f"but found {len(patient_files)}."
    )

if len(patient_files) < 5:
    raise RuntimeError(
        f"Not enough patient files found in {DATA_DIR}"
    )

# First reserve 20% of patients for testing.
train_val_files, test_files = train_test_split(
    patient_files,
    test_size=0.20,
    random_state=RANDOM_STATE,
    shuffle=True,
)

# Split the remaining patients into training and validation.
train_files, val_files = train_test_split(
    train_val_files,
    test_size=10 / 48,
    random_state=RANDOM_STATE,
    shuffle=True,
)

print("Patient-level split:")
print(f"Training patients:   {len(train_files)}")
print(f"Validation patients: {len(val_files)}")
print(f"Test patients:       {len(test_files)}")


# ============================================================
# 3. CREATE TEST DATASET AND LOADER
# ============================================================

test_dataset = MRIDataset(
    test_files,
    sequence_length=SEQUENCE_LENGTH,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

print(f"Test sequences: {len(test_dataset)}")


# ============================================================
# 4. LOAD TRAINED MODEL
# ============================================================

model = SpatioTemporalMSModel().to(DEVICE)

if not os.path.isfile(CHECKPOINT_PATH):
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT_PATH}"
    )

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False,
)

# Support checkpoints saved as either a raw state dictionary
# or a dictionary containing the model state.
if isinstance(checkpoint, dict):
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)
model.eval()

print(f"Model loaded on: {DEVICE}")


# ============================================================
# 5. EVALUATE THE TEST SET
# ============================================================

all_labels = []
all_probabilities = []
all_predictions = []
all_patient_ids = []
all_start_slices = []

with torch.no_grad():

    for batch in test_loader:

        # The dataset returns a dictionary, not a tuple.
        if not isinstance(batch, dict):
            raise TypeError(
                "Expected each batch to be a dictionary, "
                f"but received {type(batch).__name__}."
            )

        required_keys = [
            "images",
            "labels",
            "patient_id",
            "start_slice",
        ]

        missing_keys = [
            key for key in required_keys
            if key not in batch
        ]

        if missing_keys:
            raise KeyError(
                f"Missing dataset keys: {missing_keys}"
            )

        images = batch["images"].to(DEVICE)
        labels = torch.as_tensor(batch["labels"])

        patient_ids = batch["patient_id"]
        start_slices = batch["start_slice"]

        # Forward pass: one prediction per input sequence.
        logits, attention_weights = model(images)
        probabilities = torch.sigmoid(logits).reshape(-1)

        # Convert labels into one binary target per sequence.
        # For labels shaped (batch_size, sequence_length),
        # a sequence is positive if any slice has a lesion.
        if labels.ndim == 2:
            sequence_labels = (
                labels > 0
            ).any(dim=1).long()
        elif labels.ndim == 1:
            sequence_labels = (
                labels > 0
            ).long()
        else:
            raise ValueError(
                f"Unexpected label shape: {tuple(labels.shape)}"
            )

        sequence_labels = sequence_labels.reshape(-1)

        if len(sequence_labels) != len(probabilities):
            raise RuntimeError(
                "Number of labels does not match number "
                f"of predictions: {len(sequence_labels)} "
                f"labels vs {len(probabilities)} predictions. "
                f"Original label shape: {tuple(labels.shape)}"
            )

        predictions = (
            probabilities >= THRESHOLD
        ).long()

        all_labels.extend(
            sequence_labels.cpu().numpy().tolist()
        )
        all_probabilities.extend(
            probabilities.cpu().numpy().tolist()
        )
        all_predictions.extend(
            predictions.cpu().numpy().tolist()
        )

        # Preserve patient IDs and sequence start slices.
        if isinstance(patient_ids, (list, tuple)):
            all_patient_ids.extend(
                [str(pid) for pid in patient_ids]
            )
        else:
            all_patient_ids.extend(
                [str(pid) for pid in patient_ids]
            )

        if torch.is_tensor(start_slices):
            all_start_slices.extend(
                start_slices.cpu().reshape(-1).tolist()
            )
        elif isinstance(start_slices, (list, tuple)):
            all_start_slices.extend(
                list(start_slices)
            )
        else:
            all_start_slices.extend(
                [start_slices]
            )

        if len(all_labels) <= BATCH_SIZE:
            print("Image batch shape:", tuple(images.shape))
            print("Label batch shape:", tuple(labels.shape))
            print(
                "Prediction batch shape:",
                tuple(probabilities.shape),
            )


# ============================================================
# 6. CALCULATE METRICS
# ============================================================

y_true = np.asarray(all_labels, dtype=int)
y_prob = np.asarray(all_probabilities, dtype=float)
y_pred = np.asarray(all_predictions, dtype=int)

if len(y_true) == 0:
    raise RuntimeError("No test predictions were generated.")

if not (
    len(y_true)
    == len(y_prob)
    == len(y_pred)
    == len(all_patient_ids)
    == len(all_start_slices)
):
    raise RuntimeError(
        "Prediction metadata lengths do not match. "
        "Check the dataset's batch format."
    )

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true, y_pred, zero_division=0
)

sensitivity = recall_score(
    y_true, y_pred, zero_division=0
)

f1 = f1_score(
    y_true, y_pred, zero_division=0
)

cm = confusion_matrix(
    y_true, y_pred, labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0.0
)

roc_auc = (
    roc_auc_score(y_true, y_prob)
    if len(np.unique(y_true)) == 2
    else float("nan")
)

pr_auc = (
    average_precision_score(y_true, y_prob)
    if len(np.unique(y_true)) == 2
    else float("nan")
)


# ============================================================
# 7. PRINT RESULTS
# ============================================================

print("\n" + "=" * 55)
print("TEST SET EVALUATION RESULTS")
print("=" * 55)

print(f"Test sequences: {len(y_true)}")
print(f"Accuracy:       {accuracy:.4f}")
print(f"Precision:      {precision:.4f}")
print(f"Sensitivity:    {sensitivity:.4f}")
print(f"Specificity:    {specificity:.4f}")
print(f"F1-score:       {f1:.4f}")
print(f"ROC-AUC:        {roc_auc:.4f}")
print(f"PR-AUC:         {pr_auc:.4f}")

print("\nConfusion Matrix:")
print("                 Predicted")
print("                 Negative  Positive")
print(f"Actual Negative  {tn:8d}  {fp:8d}")
print(f"Actual Positive  {fn:8d}  {tp:8d}")


# ============================================================
# 8. SAVE INDIVIDUAL SEQUENCE PREDICTIONS
# ============================================================

predictions_df = pd.DataFrame({
    "patient_id": all_patient_ids,
    "start_slice": all_start_slices,
    "actual_label": y_true,
    "predicted_probability": y_prob,
    "predicted_label": y_pred,
})

predictions_path = os.path.join(
    OUTPUT_DIR, "test_predictions.csv"
)

predictions_df.to_csv(
    predictions_path,
    index=False,
)


# ============================================================
# 9. SAVE METRICS TO A TEXT FILE
# ============================================================

metrics_path = os.path.join(
    OUTPUT_DIR, "evaluation_metrics.txt"
)

with open(metrics_path, "w", encoding="utf-8") as file:
    file.write("TEST SET EVALUATION RESULTS\n")
    file.write("=" * 45 + "\n")
    file.write(f"Device: {DEVICE}\n")
    file.write(f"Test patients: {len(test_files)}\n")
    file.write(f"Test sequences: {len(y_true)}\n")
    file.write(f"Threshold: {THRESHOLD}\n\n")

    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Precision: {precision:.4f}\n")
    file.write(f"Sensitivity: {sensitivity:.4f}\n")
    file.write(f"Specificity: {specificity:.4f}\n")
    file.write(f"F1-score: {f1:.4f}\n")
    file.write(f"ROC-AUC: {roc_auc:.4f}\n")
    file.write(f"PR-AUC: {pr_auc:.4f}\n\n")

    file.write("Confusion Matrix:\n")
    file.write(f"TN: {tn}\n")
    file.write(f"FP: {fp}\n")
    file.write(f"FN: {fn}\n")
    file.write(f"TP: {tp}\n")


# ============================================================
# 10. SAVE ROC CURVE
# ============================================================

if len(np.unique(y_true)) == 2:

    fpr, tpr, _ = roc_curve(
        y_true, y_prob
    )

    plt.figure(figsize=(7, 6))
    plt.plot(
        fpr,
        tpr,
        label=f"ROC-AUC = {roc_auc:.4f}",
    )
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier",
    )
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Test Set")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "roc_curve.png"),
        dpi=300,
    )
    plt.close()


# ============================================================
# 11. SAVE PRECISION-RECALL CURVE
# ============================================================

if len(np.unique(y_true)) == 2:

    pr_precision, pr_recall, _ = (
        precision_recall_curve(y_true, y_prob)
    )

    plt.figure(figsize=(7, 6))
    plt.plot(
        pr_recall,
        pr_precision,
        label=f"PR-AUC (AP) = {pr_auc:.4f}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - Test Set")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "precision_recall_curve.png",
        ),
        dpi=300,
    )
    plt.close()


# ============================================================
# 12. SAVE PER-PATIENT SUMMARY
# ============================================================

patient_summary = (
    predictions_df
    .groupby("patient_id", as_index=False)
    .agg(
        total_sequences=("actual_label", "size"),
        actual_positive_sequences=("actual_label", "sum"),
        predicted_positive_sequences=("predicted_label", "sum"),
        mean_predicted_probability=(
            "predicted_probability", "mean"
        ),
    )
)

patient_summary_path = os.path.join(
    OUTPUT_DIR, "per_patient_summary.csv"
)

patient_summary.to_csv(
    patient_summary_path,
    index=False,
)


# ============================================================
# 13. FINISH
# ============================================================

print("\nEvaluation completed.")
print("Results saved in:", OUTPUT_DIR)
print("Predictions:", predictions_path)
print("Metrics:", metrics_path)
print("Per-patient summary:", patient_summary_path)

print(
    "\nImportant: these are sequence-level lesion-presence metrics. "
    
)
