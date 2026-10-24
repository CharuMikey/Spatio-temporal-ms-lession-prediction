
from pathlib import Path
import csv

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.dataset_loader import MRIDataset
from src.model import SpatioTemporalMSModel


# ---------------- SETTINGS ----------------

ROOT = Path(r"D:\MS diagnosis")
DATA_DIR = ROOT / "processed_data"
CHECKPOINT_PATH = ROOT / "checkpoints" / "best_model.pth"

OUTPUT_DIR = ROOT / "patient_reports"
SEQUENCE_LENGTH = 5
DEVICE = torch.device("cpu")

torch.set_num_threads(4)


# ---------------- LOAD DATA ----------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(
        DATA_DIR.glob("Patient-*.npz"),
        key=lambda p: int(p.stem.split("-")[1])
    )

    if not files:
        raise FileNotFoundError(
            f"No patient files found in {DATA_DIR}"
        )

    print("Patients found:", len(files))

    dataset = MRIDataset(
        files,
        sequence_length=SEQUENCE_LENGTH
    )

    # Group sequences by patient.
    patient_sequences = {}

    for sample in dataset.samples:
        images, label, patient_id, start_slice = sample

        patient_id = str(patient_id)

        patient_sequences.setdefault(
            patient_id, []
        ).append(sample)

    # ---------------- LOAD MODEL ----------------

    model = SpatioTemporalMSModel().to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Checkpoint loaded successfully.")

    results = []

    # ---------------- PROCESS EACH PATIENT ----------------

    for patient_number in range(1, 61):
        patient_id = str(patient_number)

        if patient_id not in patient_sequences:
            print(f"Skipping Patient {patient_id}: no sequences")
            continue

        samples = patient_sequences[patient_id]

        # Prefer a lesion-positive sequence for visualization.
        # If none exists, use the first available sequence.
        sample = next(
            (s for s in samples if float(s[1]) == 1.0),
            samples[0]
        )

        images, true_label, _, start_slice = sample

        input_tensor = torch.from_numpy(
            images.copy()
        ).permute(0, 3, 1, 2).float() / 255.0

        input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits, attention = model(input_tensor)
            probability = torch.sigmoid(logits).reshape(-1)[0].item()

        predicted_label = int(probability >= 0.5)
        true_label = int(true_label)

        patient_file = DATA_DIR / f"Patient-{patient_id}.npz"

        with np.load(patient_file) as data:
            masks = data["masks"]

        sequence_masks = masks[
            start_slice:start_slice + SEQUENCE_LENGTH
        ]

        # Show a lesion-containing slice when possible.
        positive_positions = [
            i for i, mask in enumerate(sequence_masks)
            if np.any(mask > 0)
        ]

        display_position = (
            positive_positions[0]
            if positive_positions
            else min(SEQUENCE_LENGTH // 2,
                     len(sequence_masks) - 1)
        )

        display_image = images[display_position]
        display_mask = sequence_masks[display_position]

        weights = attention[0].detach().cpu().numpy()

        # ---------------- CREATE FIGURE ----------------

        fig, axes = plt.subplots(2, 3, figsize=(14, 9))

        for channel, name in enumerate(["FLAIR", "T1", "T2"]):
            axes[0, channel].imshow(
                display_image[:, :, channel],
                cmap="gray"
            )
            axes[0, channel].set_title(name)
            axes[0, channel].axis("off")

        axes[1, 0].imshow(
            display_image[:, :, 0],
            cmap="gray"
        )

        overlay = np.ma.masked_where(
            display_mask == 0,
            display_mask
        )

        axes[1, 0].imshow(
            overlay,
            cmap="Reds",
            alpha=0.65,
            vmin=0,
            vmax=1
        )

        axes[1, 0].set_title("FLAIR + annotated lesion mask")
        axes[1, 0].axis("off")

        slice_numbers = np.arange(
            start_slice,
            start_slice + len(weights)
        )

        axes[1, 1].bar(slice_numbers, weights)
        axes[1, 1].set_title("Temporal Attention")
        axes[1, 1].set_xlabel("Slice index")
        axes[1, 1].set_ylabel("Attention weight")

        axes[1, 2].axis("off")

        result_text = (
            f"Patient: {patient_id}\n\n"
            f"Reference label: {true_label}\n"
            f"Predicted label: {predicted_label}\n"
            f"Probability: {probability:.4f}\n\n"
            f"{'Correct' if predicted_label == true_label else 'Incorrect'} "
            f"sequence prediction"
        )

        axes[1, 2].text(
            0.05, 0.75,
            result_text,
            fontsize=12,
            va="top"
        )

        fig.suptitle(
            f"Patient {patient_id} - MRI Lesion Analysis",
            fontsize=15
        )

        plt.tight_layout()

        output_file = OUTPUT_DIR / f"Patient-{patient_id}.png"

        fig.savefig(
            output_file,
            dpi=130,
            bbox_inches="tight"
        )

        plt.close(fig)

        results.append({
            "patient_id": patient_id,
            "sequences_available": len(samples),
            "selected_slice_start": start_slice,
            "reference_sequence_label": true_label,
            "predicted_sequence_label": predicted_label,
            "predicted_probability": round(probability, 6),
            "sequence_prediction_correct": (
                predicted_label == true_label
            ),
            "attention_weights": " ".join(
                f"{w:.4f}" for w in weights
            )
        })

        print(
            f"Patient {patient_id}: "
            f"label={true_label}, "
            f"prediction={predicted_label}, "
            f"probability={probability:.4f}"
        )

    # ---------------- SAVE CSV REPORT ----------------

    csv_path = OUTPUT_DIR / "all_patient_predictions.csv"

    if results:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=results[0].keys()
            )
            writer.writeheader()
            writer.writerows(results)

    print("\n========== REPORT COMPLETE ==========")
    print("Patient reports generated:", len(results))
    print("Image folder:", OUTPUT_DIR)
    print("CSV report:", csv_path)
    print(
        "\nThese are sequence-level lesion predictions, "
        "not MS diagnostic results."
    )


if __name__ == "__main__":
    main()
