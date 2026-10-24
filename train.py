
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from dataset_loader import MRIDataset
from model import SpatioTemporalMSModel


# ---------------- CONFIGURATION ----------------

DATA_DIR = Path(r"D:\MS diagnosis\processed_data")
CHECKPOINT_DIR = Path(r"D:\MS diagnosis\checkpoints")
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"

SEQUENCE_LENGTH = 5
BATCH_SIZE = 2
EPOCHS = 5
LEARNING_RATE = 0.0001
RANDOM_SEED = 42

DEVICE = torch.device("cpu")

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------- REPRODUCIBILITY ----------------

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

torch.set_num_threads(4)


# ---------------- DATA PREPARATION ----------------

def prepare_data():

    files = sorted(
        DATA_DIR.glob("Patient-*.npz"),
        key=lambda p: int(p.stem.split("-")[1])
    )

    if len(files) < 3:
        raise RuntimeError(
            f"Expected patient files in {DATA_DIR}; found {len(files)}"
        )

    # 80% development, 20% held-out testing.
    train_val_files, test_files = train_test_split(
        files,
        test_size=0.20,
        random_state=RANDOM_SEED
    )

    # Split development patients into training and validation.
    train_files, val_files = train_test_split(
        train_val_files,
        test_size=0.20,
        random_state=RANDOM_SEED
    )

    train_dataset = MRIDataset(
        train_files, SEQUENCE_LENGTH
    )
    val_dataset = MRIDataset(
        val_files, SEQUENCE_LENGTH
    )
    test_dataset = MRIDataset(
        test_files, SEQUENCE_LENGTH
    )

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

    # Check that patients never cross split boundaries.
    train_ids = {p.stem for p in train_files}
    val_ids = {p.stem for p in val_files}
    test_ids = {p.stem for p in test_files}

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    print("========== DATA SUMMARY ==========")
    print(f"Training patients:   {len(train_files)}")
    print(f"Validation patients: {len(val_files)}")
    print(f"Testing patients:    {len(test_files)}")
    print(f"Training sequences:  {len(train_dataset)}")
    print(f"Validation sequences:{len(val_dataset)}")
    print(f"Testing sequences:   {len(test_dataset)}")

    return train_dataset, train_loader, val_loader, test_loader


# ---------------- LOSS FUNCTION ----------------

def create_loss_function(train_dataset):

    labels = np.array([
        sample[1] for sample in train_dataset.samples
    ], dtype=np.float32)

    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))

    print("\n========== TRAINING LABELS ==========")
    print(f"Lesion-positive sequences: {positives}")
    print(f"Lesion-negative sequences: {negatives}")

    if positives == 0 or negatives == 0:
        raise RuntimeError(
            "Training data must contain both classes."
        )

    # Compensate for class imbalance in the training set.
    pos_weight = torch.tensor(
        [negatives / positives],
        dtype=torch.float32,
        device=DEVICE
    )

    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


# ---------------- TRAINING / VALIDATION ----------------

def run_epoch(model, loader, criterion, optimizer=None):

    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    total_samples = 0
    all_labels = []
    all_predictions = []

    for batch in loader:

        images = batch["images"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits, attention_weights = model(images)

            loss = criterion(logits, labels)

            if training:
                loss.backward()

                # Reduce the chance of unstable LSTM gradients.
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )

                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).long()

        all_labels.extend(labels.cpu().numpy().astype(int))
        all_predictions.extend(predictions.cpu().numpy())

    average_loss = total_loss / max(total_samples, 1)

    accuracy = accuracy_score(
        all_labels, all_predictions
    )

    return average_loss, accuracy


# ---------------- FINAL TEST EVALUATION ----------------

def evaluate_test_set(model, loader, criterion):

    model.eval()

    total_loss = 0.0
    total_samples = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in loader:

            images = batch["images"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            logits, attention_weights = model(images)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            predictions = (
                torch.sigmoid(logits) >= 0.5
            ).long()

            all_labels.extend(
                labels.cpu().numpy().astype(int)
            )
            all_predictions.extend(
                predictions.cpu().numpy()
            )

    average_loss = total_loss / max(total_samples, 1)

    print("\n========== HELD-OUT TEST RESULTS ==========")
    print(f"Test loss:  {average_loss:.4f}")
    print(
        f"Accuracy:   "
        f"{accuracy_score(all_labels, all_predictions):.4f}"
    )
    print(
        f"Precision:  "
        f"{precision_score(all_labels, all_predictions, zero_division=0):.4f}"
    )
    print(
        f"Recall:     "
        f"{recall_score(all_labels, all_predictions, zero_division=0):.4f}"
    )
    print(
        f"F1-score:   "
        f"{f1_score(all_labels, all_predictions, zero_division=0):.4f}"
    )
    print("Confusion matrix:")
    print(confusion_matrix(all_labels, all_predictions, labels=[0, 1]))

    print(
        "\nThese metrics evaluate lesion presence per sequence, "
        "not MS diagnosis."
    )


# ---------------- MAIN ----------------

def main():

    train_dataset, train_loader, val_loader, test_loader = (
        prepare_data()
    )

    model = SpatioTemporalMSModel().to(DEVICE)

    criterion = create_loss_function(train_dataset)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_loss = float("inf")

    print("\n========== TRAINING STARTED ==========")

    for epoch in range(EPOCHS):

        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, optimizer
        )

        val_loss, val_accuracy = run_epoch(
            model, val_loader, criterion
        )

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

        # Save the model selected by validation loss.
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "random_seed": RANDOM_SEED
                },
                BEST_MODEL_PATH
            )

            print("  Best checkpoint saved.")

    print("\nTraining completed.")
    print(f"Best model saved at: {BEST_MODEL_PATH}")

    # Evaluate the test patients only after model selection.
    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    evaluate_test_set(model, test_loader, criterion)


if __name__ == "__main__":
    main()
