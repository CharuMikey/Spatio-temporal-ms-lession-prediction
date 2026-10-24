
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.model import SpatioTemporalMSModel
from src.dataset_loader import MRIDataset


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(r"D:\MS diagnosis")
DATA_DIR = PROJECT_DIR / "processed_data"
CHECKPOINT_PATH = PROJECT_DIR / "checkpoints" / "best_model.pth"

OUTPUT_DIR = PROJECT_DIR / "xgboost_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCE_LENGTH = 5
BATCH_SIZE = 4
RANDOM_STATE = 42

DEVICE = torch.device("cpu")

print("=" * 65)
print("XGBOOST CLASSIFIER ON DEEP SEQUENCE FEATURES")
print("=" * 65)
print("Project directory:", PROJECT_DIR)
print("Device:", DEVICE)


# ============================================================
# 2. LOAD PATIENT FILES
# ============================================================

patient_files = sorted(
    DATA_DIR.glob("Patient-*.npz"),
    key=lambda path: int(path.stem.split("-")[1]),
)

if len(patient_files) != 60:
    print(
        f"Warning: expected 60 patient files, "
        f"but found {len(patient_files)}."
    )

if len(patient_files) < 5:
    raise RuntimeError(
        f"Not enough processed patient files in {DATA_DIR}"
    )

if not CHECKPOINT_PATH.is_file():
    raise FileNotFoundError(
        f"Trained model checkpoint not found: {CHECKPOINT_PATH}"
    )


# ============================================================
# 3. RECREATE THE SAME PATIENT-LEVEL SPLIT AS train.py
# ============================================================

train_val_files, test_files = train_test_split(
    patient_files,
    test_size=0.20,
    random_state=RANDOM_STATE,
    shuffle=True,
)

train_files, val_files = train_test_split(
    train_val_files,
    test_size=0.20,
    random_state=RANDOM_STATE,
    shuffle=True,
)

train_ids = {p.stem for p in train_files}
val_ids = {p.stem for p in val_files}
test_ids = {p.stem for p in test_files}

assert train_ids.isdisjoint(val_ids)
assert train_ids.isdisjoint(test_ids)
assert val_ids.isdisjoint(test_ids)

print("\nPatient-level split:")
print("Training patients:", len(train_files))
print("Validation patients:", len(val_files))
print("Test patients:", len(test_files))

split_rows = []

for split_name, files in [
    ("train", train_files),
    ("validation", val_files),
    ("test", test_files),
]:
    for patient_file in files:
        split_rows.append(
            {
                "patient_id": patient_file.stem,
                "split": split_name,
            }
        )

pd.DataFrame(split_rows).sort_values(
    "patient_id"
).to_csv(
    OUTPUT_DIR / "patient_split_used.csv",
    index=False,
)


# ============================================================
# 4. LOAD THE EXISTING DEEP LEARNING CHECKPOINT
# ============================================================

model = SpatioTemporalMSModel().to(DEVICE)

try:
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )
except TypeError:
    # Compatibility with PyTorch versions that do not support
    # the weights_only argument.
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

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

print("\nDeep learning checkpoint loaded successfully.")


# ============================================================
# 5. CAPTURE TEMPORAL-ATTENTION CONTEXT FEATURES
# ============================================================

# The existing model returns:
#   logits, attention_weights
#
# The input to its classifier is the attention-weighted
# context vector. We capture that vector without modifying
# model.py or retraining the neural network.

captured = {}


def capture_context(module, module_inputs, module_output):
    if not isinstance(module_output, (tuple, list)):
        raise RuntimeError(
            "Unexpected TemporalAttention output format."
        )

    captured["context"] = (
        module_output[0].detach().cpu().numpy().astype(np.float32)
    )


hook_handle = model.temporal_attention.register_forward_hook(
    capture_context
)


def batch_values_to_list(values):
    if torch.is_tensor(values):
        return values.detach().cpu().reshape(-1).tolist()

    if isinstance(values, np.ndarray):
        return values.reshape(-1).tolist()

    if isinstance(values, (list, tuple)):
        return list(values)

    return [values]


def extract_sequence_labels(labels):
    labels = torch.as_tensor(labels)

    if labels.ndim == 2:
        # A sequence is positive if at least one of its
        # slices has a positive lesion-mask label.
        return (labels > 0).any(dim=1).long().reshape(-1)

    if labels.ndim == 1:
        # Labels are already sequence-level.
        return (labels > 0).long().reshape(-1)

    raise ValueError(
        f"Unexpected label tensor shape: {tuple(labels.shape)}"
    )


def extract_features(files, split_name):
    dataset = MRIDataset(
        files,
        sequence_length=SEQUENCE_LENGTH,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    feature_batches = []
    label_batches = []
    patient_ids = []
    start_slices = []

    print(
        f"\nExtracting {split_name} features "
        f"from {len(dataset)} sequences..."
    )

    with torch.no_grad():
        for batch in loader:
            if not isinstance(batch, dict):
                raise TypeError(
                    "Expected MRIDataset to return a dictionary."
                )

            required_keys = [
                "images",
                "labels",
                "patient_id",
                "start_slice",
            ]

            missing = [
                key for key in required_keys if key not in batch
            ]

            if missing:
                raise KeyError(
                    f"Missing keys in dataset batch: {missing}"
                )

            images = batch["images"].to(DEVICE)
            labels = extract_sequence_labels(batch["labels"])

            captured.pop("context", None)

            # Runs EfficientNet-B0, BiLSTM, temporal attention,
            # and the original classifier.
            # The hook captures the attention context vector.
            model(images)

            if "context" not in captured:
                raise RuntimeError(
                    "Failed to capture temporal-attention features."
                )

            features = captured["context"]

            if features.shape[0] != len(labels):
                raise RuntimeError(
                    "Feature and label counts do not match."
                )

            feature_batches.append(features)
            label_batches.append(labels.numpy())

            batch_patient_ids = batch_values_to_list(
                batch["patient_id"]
            )
            batch_start_slices = batch_values_to_list(
                batch["start_slice"]
            )

            patient_ids.extend(
                str(pid) for pid in batch_patient_ids
            )

            start_slices.extend(
                int(value)
                if isinstance(value, (int, np.integer))
                or (
                    isinstance(value, float)
                    and value.is_integer()
                )
                else int(value.item())
                if torch.is_tensor(value)
                else value
                for value in batch_start_slices
            )

    if not feature_batches:
        raise RuntimeError(
            f"No features were extracted for {split_name}."
        )

    X = np.concatenate(feature_batches, axis=0).astype(
        np.float32
    )
    y = np.concatenate(label_batches, axis=0).astype(int)

    if not (
        len(X) == len(y) == len(patient_ids) == len(start_slices)
    ):
        raise RuntimeError(
            f"Metadata count mismatch for {split_name}."
        )

    print("Feature matrix shape:", X.shape)
    print("Labels shape:", y.shape)
    print("Negative sequences:", int(np.sum(y == 0)))
    print("Positive sequences:", int(np.sum(y == 1)))

    return X, y, patient_ids, start_slices


try:
    X_train, y_train, _, _ = extract_features(
        train_files, "training"
    )

    X_val, y_val, _, _ = extract_features(
        val_files, "validation"
    )

    X_test, y_test, test_patient_ids, test_start_slices = (
        extract_features(test_files, "test")
    )

finally:
    hook_handle.remove()


# ============================================================
# 6. CHECK CLASS DISTRIBUTION
# ============================================================

if len(np.unique(y_train)) != 2:
    raise RuntimeError(
        "Training sequences must contain both classes."
    )

if len(np.unique(y_val)) != 2:
    raise RuntimeError(
        "Validation sequences must contain both classes "
        "to select a meaningful decision threshold."
    )

if len(np.unique(y_test)) != 2:
    print(
        "Warning: test sequences contain only one class; "
        "ROC-AUC and Average Precision may not be meaningful."
    )

print("\nFeature extractor output dimension:", X_train.shape[1])


# ============================================================
# 7. TRAIN XGBOOST AS AN ALTERNATIVE CLASSIFIER
# ============================================================

# Compute class-balanced sample weights from training labels
# only. Do not use validation or test labels to fit weights.

sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train,
)

xgb_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=250,
    max_depth=2,
    learning_rate=0.03,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=5.0,
    tree_method="hist",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

print("\nTraining XGBoost...")

xgb_model.fit(
    X_train,
    y_train,
    sample_weight=sample_weights,
    eval_set=[(X_val, y_val)],
    verbose=False,
)

print("XGBoost training completed.")

MODEL_PATH = OUTPUT_DIR / "xgboost_sequence_classifier.json"
xgb_model.save_model(str(MODEL_PATH))

print("XGBoost model saved:", MODEL_PATH)


# ============================================================
# 8. SELECT THE DECISION THRESHOLD USING VALIDATION DATA
# ============================================================

val_probabilities = xgb_model.predict_proba(X_val)[:, 1]

best_threshold = 0.5
best_balanced_accuracy = -1.0
best_specificity = -1.0

# Threshold selection uses validation predictions only.
threshold_candidates = np.arange(0.05, 0.951, 0.01)

for threshold in threshold_candidates:
    val_predictions = (
        val_probabilities >= threshold
    ).astype(int)

    score = balanced_accuracy_score(
        y_val, val_predictions
    )

    val_cm = confusion_matrix(
        y_val, val_predictions, labels=[0, 1]
    )

    tn, fp, fn, tp = val_cm.ravel()

    specificity = (
        tn / (tn + fp) if (tn + fp) > 0 else 0.0
    )

    # Maximize balanced accuracy, breaking ties using specificity.
    if (
        score > best_balanced_accuracy
        or (
            np.isclose(score, best_balanced_accuracy)
            and specificity > best_specificity
        )
    ):
        best_threshold = float(threshold)
        best_balanced_accuracy = float(score)
        best_specificity = float(specificity)

print("\nValidation threshold selection:")
print(f"Selected threshold: {best_threshold:.2f}")
print(
    f"Validation balanced accuracy: "
    f"{best_balanced_accuracy:.4f}"
)
print(f"Validation specificity: {best_specificity:.4f}")


# ============================================================
# 9. EVALUATE ON THE HELD-OUT TEST PATIENTS
# ============================================================

# The test set is used only for final evaluation.
test_probabilities = xgb_model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

accuracy = accuracy_score(y_test, test_predictions)

balanced_accuracy = balanced_accuracy_score(
    y_test, test_predictions
)

precision = precision_score(
    y_test, test_predictions, zero_division=0
)

sensitivity = recall_score(
    y_test, test_predictions, zero_division=0
)

f1 = f1_score(
    y_test, test_predictions, zero_division=0
)

cm = confusion_matrix(
    y_test, test_predictions, labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

specificity = (
    tn / (tn + fp) if (tn + fp) > 0 else 0.0
)

if len(np.unique(y_test)) == 2:
    roc_auc = float(
        roc_auc_score(y_test, test_probabilities)
    )
    average_precision = float(
        average_precision_score(y_test, test_probabilities)
    )
else:
    roc_auc = None
    average_precision = None

majority_baseline_accuracy = float(
    max(np.mean(y_test == 0), np.mean(y_test == 1))
)


# ============================================================
# 10. PRINT RESULTS
# ============================================================

print("\n" + "=" * 65)
print("XGBOOST TEST RESULTS")
print("=" * 65)

print("Test sequences:", len(y_test))
print("Selected threshold:", f"{best_threshold:.2f}")
print(f"Accuracy:           {accuracy:.4f}")
print(f"Balanced accuracy:  {balanced_accuracy:.4f}")
print(f"Precision:          {precision:.4f}")
print(f"Sensitivity/Recall: {sensitivity:.4f}")
print(f"Specificity:        {specificity:.4f}")
print(f"F1-score:           {f1:.4f}")

print(
    "ROC-AUC:",
    f"{roc_auc:.4f}" if roc_auc is not None else "N/A",
)

print(
    "Average Precision:",
    (
        f"{average_precision:.4f}"
        if average_precision is not None
        else "N/A"
    ),
)

print(
    "Majority-class baseline accuracy:",
    f"{majority_baseline_accuracy:.4f}",
)

print("\nConfusion Matrix:")
print("                 Predicted")
print("                 Negative  Positive")
print(f"Actual Negative  {tn:8d}  {fp:8d}")
print(f"Actual Positive  {fn:8d}  {tp:8d}")


# ============================================================
# 11. SAVE TEST PREDICTIONS
# ============================================================

predictions_df = pd.DataFrame(
    {
        "patient_id": test_patient_ids,
        "start_slice": test_start_slices,
        "actual_label": y_test,
        "predicted_probability": test_probabilities,
        "predicted_label": test_predictions,
    }
)

predictions_df.to_csv(
    OUTPUT_DIR / "xgboost_test_predictions.csv",
    index=False,
)


# ============================================================
# 12. SAVE METRICS
# ============================================================

metrics = {
    "experiment": (
        "XGBoost on frozen EfficientNet-B0, BiLSTM, "
        "and temporal-attention features"
    ),
    "task": "Sequence-level lesion-presence prediction",
    "device": str(DEVICE),
    "random_state": RANDOM_STATE,
    "training_patients": len(train_files),
    "validation_patients": len(val_files),
    "test_patients": len(test_files),
    "training_sequences": len(y_train),
    "validation_sequences": len(y_val),
    "test_sequences": len(y_test),
    "training_negative_sequences": int(np.sum(y_train == 0)),
    "training_positive_sequences": int(np.sum(y_train == 1)),
    "test_negative_sequences": int(np.sum(y_test == 0)),
    "test_positive_sequences": int(np.sum(y_test == 1)),
    "threshold_selected_on_validation": best_threshold,
    "validation_balanced_accuracy_at_selected_threshold": (
        best_balanced_accuracy
    ),
    "accuracy": float(accuracy),
    "balanced_accuracy": float(balanced_accuracy),
    "precision": float(precision),
    "sensitivity": float(sensitivity),
    "specificity": float(specificity),
    "f1_score": float(f1),
    "roc_auc": roc_auc,
    "average_precision": average_precision,
    "majority_class_baseline_accuracy": (
        majority_baseline_accuracy
    ),
    "confusion_matrix": {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    },
    "note": (
        "These are sequence-level lesion-presence results, "
        "not MS-versus-healthy diagnostic results. "
        "Test sequences from the same patient may be correlated. "
        "This experiment does not retrain the neural feature extractor."
    ),
}

with open(
    OUTPUT_DIR / "xgboost_metrics.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(metrics, file, indent=4)


# ============================================================
# 13. SAVE ROC CURVE
# ============================================================

if len(np.unique(y_test)) == 2:
    fpr, tpr, _ = roc_curve(
        y_test, test_probabilities
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
    plt.title("XGBoost - ROC Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "xgboost_roc_curve.png",
        dpi=300,
    )
    plt.close()


# ============================================================
# 14. SAVE PRECISION-RECALL CURVE
# ============================================================

if len(np.unique(y_test)) == 2:
    pr_precision, pr_recall, _ = precision_recall_curve(
        y_test, test_probabilities
    )

    plt.figure(figsize=(7, 6))
    plt.plot(
        pr_recall,
        pr_precision,
        label=f"Average Precision = {average_precision:.4f}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("XGBoost - Precision-Recall Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "xgboost_precision_recall_curve.png",
        dpi=300,
    )
    plt.close()


# ============================================================
# 15. SAVE FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame(
    {
        "feature": [
            f"attention_context_{i + 1:03d}"
            for i in range(X_train.shape[1])
        ],
        "importance": xgb_model.feature_importances_,
    }
).sort_values(
    "importance", ascending=False
)

importance_df.to_csv(
    OUTPUT_DIR / "xgboost_feature_importance.csv",
    index=False,
)


# ============================================================
# 16. FINISH
# ============================================================

print("\nXGBoost experiment completed.")
print("Results directory:", OUTPUT_DIR)
print("Model:", MODEL_PATH)
print(
    "Predictions:",
    OUTPUT_DIR / "xgboost_test_predictions.csv",
)
print(
    "Metrics:",
    OUTPUT_DIR / "xgboost_metrics.json",
)
print(
    "ROC curve:",
    OUTPUT_DIR / "xgboost_roc_curve.png",
)
print(
    "Precision-recall curve:",
    OUTPUT_DIR / "xgboost_precision_recall_curve.png",
)

print(
    "\nImportant: compare these metrics against the original "
    "model on the same held-out test patients."
)
