
from pathlib import Path
import csv

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model import SpatioTemporalMSModel


# ==================================================
# CONFIGURATION
# ==================================================

ROOT = Path(r"D:\MS diagnosis")
DATA_DIR = ROOT / "processed_data"
CHECKPOINT_PATH = ROOT / "checkpoints" / "best_model.pth"
OUTPUT_DIR = ROOT / "GradCAM_reports"

SEQUENCE_LENGTH = 5
IMAGE_SIZE = 224
NUM_PATIENTS = 60

DEVICE = torch.device("cpu")
torch.set_num_threads(4)


# ==================================================
# LOAD MODEL
# ==================================================

def load_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    model = SpatioTemporalMSModel().to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Trained model loaded successfully.")
    print("Checkpoint:", CHECKPOINT_PATH)

    return model


# ==================================================
# SELECT ONE FIVE-SLICE SEQUENCE PER PATIENT
# ==================================================

def load_patient_sequence(patient_id):
    patient_file = DATA_DIR / f"Patient-{patient_id}.npz"

    if not patient_file.exists():
        raise FileNotFoundError(
            f"Missing processed file: {patient_file}"
        )

    with np.load(patient_file) as data:
        images = data["images"].copy()
        masks = data["masks"].copy()

    if len(images) != len(masks):
        raise ValueError(
            f"Image/mask slice count differs for Patient {patient_id}"
        )

    if len(images) < SEQUENCE_LENGTH:
        raise ValueError(
            f"Patient {patient_id} has fewer than "
            f"{SEQUENCE_LENGTH} slices"
        )

    # Select the five consecutive slices with the most
    # annotated lesion pixels for visual inspection.
    # This is a visualization choice, not an evaluation split.
    best_start = 0
    best_lesion_pixels = -1

    for start in range(len(images) - SEQUENCE_LENGTH + 1):
        count = int(
            np.count_nonzero(
                masks[start:start + SEQUENCE_LENGTH] > 0
            )
        )

        if count > best_lesion_pixels:
            best_lesion_pixels = count
            best_start = start

    sequence_images = images[
        best_start:best_start + SEQUENCE_LENGTH
    ]

    sequence_masks = masks[
        best_start:best_start + SEQUENCE_LENGTH
    ]

    # Model input: [batch, time, channels, height, width].
    input_tensor = torch.from_numpy(
        sequence_images.copy()
    ).permute(0, 3, 1, 2).float() / 255.0

    input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

    return (
        sequence_images,
        sequence_masks,
        input_tensor,
        best_start
    )


# ==================================================
# GENERATE GRAD-CAM FOR ALL FIVE SLICES
# ==================================================

def generate_gradcam(model, input_tensor):
    captured = {}

    # Last convolutional feature block in EfficientNet-B0.
    target_layer = model.cnn_features[-1]

    def forward_hook(module, inputs, output):
        captured["activations"] = output

        output.register_hook(
            lambda grad: captured.update(
                {"gradients": grad}
            )
        )

    handle = target_layer.register_forward_hook(forward_hook)

    try:
        model.zero_grad(set_to_none=True)

        # Gradients are required for Grad-CAM.
        with torch.enable_grad():
            logits, attention_weights = model(input_tensor)

            # Explain evidence for the positive class.
            logits.reshape(-1)[0].backward()

        activations = captured["activations"].detach()
        gradients = captured["gradients"].detach()

        batch_size, time_steps = input_tensor.shape[:2]

        # Restore the time dimension:
        # [batch*time, channels, height, width]
        # -> [time, channels, height, width]
        activations = activations.reshape(
            batch_size,
            time_steps,
            *activations.shape[1:]
        )[0]

        gradients = gradients.reshape(
            batch_size,
            time_steps,
            *gradients.shape[1:]
        )[0]

        # Average gradients over spatial dimensions.
        # Result: [time, channels, 1, 1].
        channel_weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        # Weighted sum over feature channels.
        # Result: [time, feature_height, feature_width].
        cams = (channel_weights * activations).sum(dim=1)
        cams = F.relu(cams)

        if cams.ndim != 3:
            raise RuntimeError(
                f"Unexpected CAM shape: {tuple(cams.shape)}"
            )

        # Resize all heatmaps to the input MRI resolution.
        cams = F.interpolate(
            cams.unsqueeze(1),
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False
        ).squeeze(1)

        normalized_cams = []

        for cam in cams:
            cam = cam.cpu().numpy()
            maximum = float(cam.max())

            if maximum > 1e-8:
                cam = cam / maximum
            else:
                cam = np.zeros_like(cam)

            normalized_cams.append(cam)

        probability = torch.sigmoid(
            logits.detach().reshape(-1)[0]
        ).item()

        attention = (
            attention_weights.detach()[0].cpu().numpy()
        )

        return normalized_cams, probability, attention

    finally:
        handle.remove()


# ==================================================
# SAVE ONE PATIENT'S VISUAL REPORT
# ==================================================

def save_patient_report(
    patient_id,
    images,
    masks,
    cams,
    probability,
    attention,
    start_slice
):
    predicted_label = int(probability >= 0.5)

    # The sequence reference label is positive if any
    # pixel in any of the five masks is positive.
    reference_label = int(np.any(masks > 0))

    fig, axes = plt.subplots(
        SEQUENCE_LENGTH,
        3,
        figsize=(13, 4 * SEQUENCE_LENGTH),
        squeeze=False
    )

    for i in range(SEQUENCE_LENGTH):
        flair = images[i, :, :, 0]
        mask = masks[i]
        cam = cams[i]

        # Column 1: original FLAIR MRI.
        axes[i, 0].imshow(flair, cmap="gray")
        axes[i, 0].set_title(
            f"FLAIR — Slice {start_slice + i}"
        )

        # Column 2: reference lesion mask.
        axes[i, 1].imshow(flair, cmap="gray")

        mask_overlay = np.ma.masked_where(
            mask == 0,
            mask
        )

        axes[i, 1].imshow(
            mask_overlay,
            cmap="Reds",
            alpha=0.7,
            vmin=0,
            vmax=1
        )

        axes[i, 1].set_title("Annotated lesion mask")

        # Column 3: spatial Grad-CAM heatmap.
        axes[i, 2].imshow(flair, cmap="gray")
        axes[i, 2].imshow(
            cam,
            cmap="jet",
            alpha=0.45,
            vmin=0,
            vmax=1
        )

        axes[i, 2].set_title(
            f"Grad-CAM | Attention: {attention[i]:.3f}"
        )

        for ax in axes[i]:
            ax.axis("off")

    fig.suptitle(
        f"Patient {patient_id} | "
        f"Reference lesion label: {reference_label} | "
        f"Prediction: {predicted_label} | "
        f"Probability: {probability:.3f}",
        fontsize=14
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    output_file = OUTPUT_DIR / f"Patient-{patient_id}.png"

    fig.savefig(
        output_file,
        dpi=130,
        bbox_inches="tight"
    )

    plt.close(fig)

    return reference_label, predicted_label


# ==================================================
# PROCESS ALL 60 PATIENTS
# ==================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model()

    results = []
    failed_patients = []

    print("\n========== GENERATING 60 PATIENT REPORTS ==========")

    for patient_id in range(1, NUM_PATIENTS + 1):
        try:
            (
                images,
                masks,
                input_tensor,
                start_slice
            ) = load_patient_sequence(patient_id)

            cams, probability, attention = generate_gradcam(
                model,
                input_tensor
            )

            reference_label, predicted_label = save_patient_report(
                patient_id,
                images,
                masks,
                cams,
                probability,
                attention,
                start_slice
            )

            results.append({
                "patient_id": patient_id,
                "sequence_start_slice": start_slice,
                "reference_lesion_label": reference_label,
                "predicted_lesion_label": predicted_label,
                "positive_class_probability": round(
                    probability, 6
                ),
                "sequence_prediction_matches_reference": (
                    predicted_label == reference_label
                ),
                "temporal_attention_weights": ";".join(
                    f"{weight:.6f}" for weight in attention
                ),
                "report_file": f"Patient-{patient_id}.png"
            })

            print(
                f"[{patient_id:02d}/{NUM_PATIENTS}] "
                f"Patient {patient_id}: "
                f"reference={reference_label}, "
                f"prediction={predicted_label}, "
                f"probability={probability:.4f}"
            )

        except Exception as error:
            failed_patients.append({
                "patient_id": patient_id,
                "error": str(error)
            })

            print(
                f"[{patient_id:02d}/{NUM_PATIENTS}] "
                f"Patient {patient_id} FAILED: {error}"
            )

    # Save the report index as a CSV.
    csv_path = OUTPUT_DIR / "all_60_patient_gradcam.csv"

    if results:
        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(results[0].keys())
            )
            writer.writeheader()
            writer.writerows(results)

    # Save any failures for troubleshooting.
    if failed_patients:
        failures_path = OUTPUT_DIR / "failed_patients.csv"

        with open(
            failures_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["patient_id", "error"]
            )
            writer.writeheader()
            writer.writerows(failed_patients)

    print("\n========== FINAL SUMMARY ==========")
    print("Reports successfully generated:", len(results))
    print("Patients that failed:", len(failed_patients))
    print("Output folder:", OUTPUT_DIR)
    print("CSV summary:", csv_path)

    if len(results) == NUM_PATIENTS:
        print("SUCCESS: All 60 patient reports were generated.")
    else:
        print("Some reports need attention; check failed_patients.csv.")

    print(
        "\nThese are sequence-level lesion-presence predictions, "
        "not MS-versus-healthy diagnostic results."
    )


if __name__ == "__main__":
    main()
