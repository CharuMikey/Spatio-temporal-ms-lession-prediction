# Spatio-Temporal Attention for MRI Lesion-Presence Prediction

An academic project exploring spatio-temporal deep learning on brain MRI slices. The model combines EfficientNet-B0 for spatial feature extraction, a bidirectional LSTM (BiLSTM) for sequence modeling, and temporal attention to aggregate information across consecutive slices.

> **Scientific scope:** The current dataset and labels support sequence-level MRI lesion-presence prediction from lesion masks. These results are not clinically validated multiple-sclerosis (MS) versus healthy-control diagnostic results. This is an academic prototype, not a medical device.

## Model architecture

1. Input: five consecutive MRI slices, tensor shape `(batch, 5, 3, 224, 224)`.
2. EfficientNet-B0 extracts spatial features from each slice.
3. BiLSTM models information across the ordered slices.
4. Temporal attention weights sequence time steps and forms a context vector.
5. A classifier predicts whether the sequence contains lesion-positive labels.

Grad-CAM and prediction visualization scripts support exploratory interpretation.

## Repository files

| File | Purpose |
|---|---|
| `model.py` | EfficientNet-B0, BiLSTM, temporal attention, classifier |
| `dataset_loader.py` | Loads processed patient data and creates slice sequences |
| `train.py` | Trains the spatio-temporal model |
| `evaluate_model.py` | Evaluates the model on a held-out patient split |
| `train_xgboost.py` | Experimental XGBoost approach |
| `gradcam_visualization.py` | Grad-CAM visualization |
| `visualize_predictions.py`, `visualize_mri.py` | Prediction and MRI visualization |
| `prepare_dataset.py`, `preprocess_dataset.py` | Dataset preparation and preprocessing |
| `check_dataset.py`, `inspect_dataset.py`, `inspect_modalities.py` | Dataset inspection and validation |
| `save_patient_split.py` | Reconstructs and records a patient-level split |

Review each script's configuration and expected inputs before running it; local paths may need adjustment.

## Requirements and setup

Python 3.11 is recommended. Install dependencies from `requirements.txt`. Choose a PyTorch/torchvision build compatible with your OS and hardware.

To verify GPU support:

```python
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
```

The previously checked local environment had a CPU-only PyTorch build. An NVIDIA GPU will not be used until a CUDA-enabled build is installed and the scripts select CUDA when available.

## Data privacy

This repository should not contain raw MRI scans, processed patient `.npz` files, patient-level records, trained checkpoints, or generated patient reports.

1. Obtain data from an authorized source and check its license and permitted use.
2. Keep raw data outside version control.
3. Review the expected directory structure in `prepare_dataset.py` and `preprocess_dataset.py`.
4. Run the required preprocessing to create files expected by `dataset_loader.py`.
5. Verify the patient-level train/validation/test split before training and evaluation.

Do not upload identifiable or sensitive health data to GitHub. Pseudonymized patient-level outputs may also be sensitive.

## Running the project

Run commands from the project directory in an activated environment. Check the configured paths in each script first.

```bash
python check_dataset.py
python prepare_dataset.py
python preprocess_dataset.py
python train.py
python evaluate_model.py
```

The preparation scripts may be alternatives or sequential stages; inspect their code to determine which are required for your dataset. Evaluation and visualization scripts may require processed data and a trained checkpoint.

Optional experiments:

```bash
python train_xgboost.py
python gradcam_visualization.py
python visualize_predictions.py
```

## Previously reported baseline results

A previous test evaluation reported these **sequence-level lesion-presence** metrics:

| Metric | Reported value |
|---|---:|
| Accuracy | 78.12% |
| Precision | 86.80% |
| Sensitivity / Recall | 85.07% |
| Specificity | 52.73% |
| F1-score | 85.93% |
| ROC-AUC | 0.8113 |
| PR-AUC / Average Precision | 0.9487 |

Reported confusion matrix:

| | Predicted negative | Predicted positive |
|---|---:|---:|
| Actual negative | 29 | 26 |
| Actual positive | 30 | 171 |

These are historical results from one run and split; reproduce the evaluation before relying on them. Overlapping sequences can be correlated and are not statistically independent. Sequence-level metrics are not equivalent to patient-level diagnostic performance. The described dataset contains MS-patient MRI and lesion masks, without a confirmed healthy-control cohort, so these results do not establish MS-versus-healthy diagnostic accuracy.

## Reproducibility and limitations

- Split by patient, not by slice, to reduce patient leakage.
- Keep the test set untouched during model selection and threshold tuning.
- Record patient IDs, random seed, dependency versions, preprocessing steps, and checkpoint for each experiment.
- A split reconstructed after training does not independently prove the exact patients used for the original checkpoint.
- Grad-CAM highlights model-associated regions; it does not establish clinical validity.

## License

No license is included. Add one only after confirming that you have the right to license the code and that third-party dataset terms are respected.
