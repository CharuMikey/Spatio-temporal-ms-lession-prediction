
import os
import glob
import numpy as np
import streamlit as st
import torch
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

SEQUENCE_LENGTH = 5
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Spatio-Temporal MRI Analysis",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Spatio-Temporal MRI Analysis")
st.caption(
    "EfficientNet-B0 + BiLSTM + Temporal Attention"
)

st.warning(
    "Research prototype only. This application predicts "
    "lesion presence in an MRI slice sequence. It does not "
    "diagnose multiple sclerosis or replace a radiologist."
)


# ============================================================
# 3. LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_trained_model():
    if not os.path.isfile(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found: {CHECKPOINT_PATH}"
        )

    model = SpatioTemporalMSModel()

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
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
    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# 4. FIND PROCESSED PATIENT FILES
# ============================================================

@st.cache_data
def get_patient_files():
    files = glob.glob(
        os.path.join(DATA_DIR, "Patient-*.npz")
    )

    return sorted(
        files,
        key=lambda path: int(
            os.path.basename(path)
            .replace("Patient-", "")
            .replace(".npz", "")
        ),
    )


try:
    model = load_trained_model()
    patient_files = get_patient_files()

except Exception as error:
    st.error(f"Unable to initialize the application: {error}")
    st.stop()


if not patient_files:
    st.error(
        f"No processed patient files were found in {DATA_DIR}"
    )
    st.stop()


# ============================================================
# 5. SIDEBAR: PATIENT AND SEQUENCE SELECTION
# ============================================================

st.sidebar.header("MRI Sequence Selection")

selected_file = st.sidebar.selectbox(
    "Select a processed patient",
    patient_files,
    format_func=lambda path: os.path.basename(path)
    .replace(".npz", ""),
)

try:
    # Reuse the same dataset loader as the existing project.
    # This preserves its preprocessing and label handling.
    patient_dataset = MRIDataset(
        [selected_file],
        sequence_length=SEQUENCE_LENGTH,
    )

except Exception as error:
    st.error(f"Unable to load the patient dataset: {error}")
    st.stop()


if len(patient_dataset) == 0:
    st.error(
        "This patient does not contain enough slices "
        "to form a five-slice sequence."
    )
    st.stop()


sequence_index = st.sidebar.slider(
    "Sequence index",
    min_value=0,
    max_value=len(patient_dataset) - 1,
    value=0,
    step=1,
)

st.sidebar.caption(
    f"Available sequences: {len(patient_dataset)}"
)

st.sidebar.caption(f"Running on: {DEVICE}")


# ============================================================
# 6. LOAD SELECTED SEQUENCE
# ============================================================

try:
    sample = patient_dataset[sequence_index]

    if not isinstance(sample, dict):
        st.error(
            "The dataset loader did not return the expected "
            "dictionary format."
        )
        st.stop()

    required_keys = [
        "images",
        "labels",
        "patient_id",
        "start_slice",
    ]

    missing_keys = [
        key for key in required_keys
        if key not in sample
    ]

    if missing_keys:
        st.error(
            f"Missing fields in dataset sample: {missing_keys}"
        )
        st.stop()

    images = sample["images"].float()
    labels = torch.as_tensor(sample["labels"])
    patient_id = sample["patient_id"]
    start_slice = sample["start_slice"]

    if images.ndim != 4:
        st.error(
            "Unexpected image tensor shape: "
            f"{tuple(images.shape)}. "
            "Expected (5, 3, 224, 224)."
        )
        st.stop()

    if images.shape[0] != SEQUENCE_LENGTH:
        st.error(
            f"Expected {SEQUENCE_LENGTH} slices, "
            f"but received {images.shape[0]}."
        )
        st.stop()

except Exception as error:
    st.error(f"Unable to prepare the selected sequence: {error}")
    st.stop()


# ============================================================
# 7. DISPLAY SELECTED MRI SLICES
# ============================================================

st.subheader("Selected MRI Sequence")

st.write(
    f"**Patient:** {patient_id}  |  "
    f"**Sequence index:** {sequence_index}  |  "
    f"**Starting slice:** {start_slice}"
)

st.write(
    "The five processed slices below are the inputs to "
    "the spatio-temporal model."
)

columns = st.columns(SEQUENCE_LENGTH)


def prepare_display_image(image_tensor):
    """
    Convert a processed 3-channel tensor to a displayable RGB
    image. Display normalization does not alter model input.
    """
    image = image_tensor.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))

    image = np.nan_to_num(image)

    low = image.min()
    high = image.max()

    if high > low:
        image = (image - low) / (high - low)
    else:
        image = np.zeros_like(image)

    image = np.clip(image * 255, 0, 255).astype(np.uint8)

    return image


for index, column in enumerate(columns):
    with column:
        st.markdown(f"**Slice {index + 1}**")

        display_image = prepare_display_image(images[index])

        st.image(
            display_image,
            use_container_width=True,
        )


# ============================================================
# 8. RUN PREDICTION
# ============================================================

st.divider()
st.subheader("Model Prediction")

if st.button(
    "Analyze MRI Sequence",
    type="primary",
    use_container_width=True,
):

    with st.spinner("Analyzing the selected sequence..."):

        try:
            model_input = images.unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits, attention_weights = model(model_input)
                probability = torch.sigmoid(logits).item()

                attention = (
                    attention_weights.squeeze(0)
                    .detach()
                    .cpu()
                    .numpy()
                )

            prediction = int(probability >= 0.5)

            st.session_state["prediction_result"] = {
                "patient_id": str(patient_id),
                "sequence_index": int(sequence_index),
                "start_slice": int(start_slice),
                "probability": float(probability),
                "prediction": prediction,
                "attention": attention.tolist(),
            }

        except Exception as error:
            st.error(f"Prediction failed: {error}")


# ============================================================
# 9. DISPLAY PREDICTION RESULTS
# ============================================================

result = st.session_state.get("prediction_result")

if result is not None:

    # Avoid presenting a previous sequence's results as though
    # they belong to the currently selected sequence.
    if (
        result["patient_id"] != str(patient_id)
        or result["sequence_index"] != sequence_index
    ):
        st.info(
            "Select Analyze MRI Sequence to generate a "
            "prediction for the current selection."
        )

    else:
        st.markdown("### Results")

        probability = result["probability"]
        prediction = result["prediction"]

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Predicted lesion-presence probability",
                f"{probability * 100:.2f}%",
            )
            st.progress(float(probability))

        with col2:
            if prediction == 1:
                st.error(
                    "Model output: Lesion-positive sequence"
                )
            else:
                st.success(
                    "Model output: Lesion-negative sequence"
                )

            st.caption(
                "Binary threshold: 0.50. This is a model "
                "classification, not a clinical diagnosis."
            )

        # ----------------------------------------------------
        # 10. TEMPORAL ATTENTION VISUALIZATION
        # ----------------------------------------------------

        st.divider()
        st.subheader("Temporal Attention Weights")

        st.write(
            "These weights indicate how the model distributes "
            "attention across the five slices when forming "
            "its sequence representation."
        )

        attention = np.asarray(result["attention"])

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.bar(
            [f"Slice {i + 1}" for i in range(len(attention))],
            attention,
        )

        ax.set_xlabel("MRI slice")
        ax.set_ylabel("Attention weight")
        ax.set_title("Temporal Attention Across Five Slices")
        ax.set_ylim(0, max(1.0, float(attention.max()) * 1.15))

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Attention weights describe the model's internal "
            "weighting; they do not prove that a slice caused "
            "the prediction or that a lesion is clinically present."
        )


# ============================================================
# 11. PROJECT INFORMATION
# ============================================================

with st.expander("About this research prototype"):

    st.markdown(
        """
        **Model architecture**

        1. EfficientNet-B0 extracts spatial features from each MRI slice.
        2. A bidirectional LSTM processes the ordered slice features.
        3. Temporal attention combines the sequence representations.
        4. A binary classifier estimates lesion presence.

        **Current scope**

        - Uses already processed patient data.
        - Uses the existing trained checkpoint.
        - Analyzes five consecutive slices per prediction.
        - Displays temporal attention weights.

        **Limitations**

        - The model has not been established as a clinical diagnostic tool.
        - Lesion-presence prediction is not equivalent to MS diagnosis.
        - The displayed processed images are normalized for visualization.
        """
    )
