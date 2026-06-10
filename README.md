# CAST-GNN: Adaptive Spectro-Temporal Graph Learning for Motor EEG Decoding

This repository implements **CAST-GNN**: a Cross-Dataset Adaptive Spectro-Temporal Graph Neural Network for motor execution and motor imagery EEG decoding across **PhysioNet EEGMMIDB** and **MILimbEEG**.

The codebase is designed for reproducible research, leakage-safe evaluation, cross-dataset harmonization, calibration analysis, robustness testing, ablation experiments, and perturbation-validated interpretability.

---

## 1. Project Scope

CAST-GNN is designed to decode motor-related EEG activity from heterogeneous public datasets with different acquisition formats, channel montages, sampling rates, annotation structures, and label taxonomies.

The implementation supports:

- Dataset audit for EEGMMIDB and MILimbEEG.
- EEGMMIDB EDF loading and run-specific `T0/T1/T2` label reconstruction.
- MILimbEEG CSV loading and task/mode parsing from file or folder metadata.
- Leakage-safe subject-wise train/validation/test splitting.
- Band-limited preprocessing using 5–40 Hz EEG activity.
- Four-band spectral representation: 5–8 Hz, 8–13 Hz, 13–30 Hz, and 30–40 Hz.
- Dataset-specific input stems for 64-channel EEGMMIDB and 16-channel MILimbEEG.
- Hybrid anatomical–functional–learnable graph construction.
- Dataset-specific graph adapters.
- Adaptive graph attention.
- Lightweight temporal convolutional modelling.
- Task-specific prediction heads.
- Within-dataset, transfer, and joint-training experiments.
- Calibration, robustness, ablation, and interpretability modules.

---

## 2. Repository Structure

```text
cast_gnn_project/
├── configs/
│   ├── default.yaml
│   ├── eegmmidb.yaml
│   ├── milimbeeg.yaml
│   └── joint.yaml
│
├── data/
│   ├── README.md
│   ├── eegmmidb_raw/
│   └── milimbeeg_raw/
│
├── src/
│   ├── data/
│   │   ├── eegmmidb_loader.py
│   │   ├── milimbeeg_loader.py
│   │   ├── label_harmonization.py
│   │   ├── audit.py
│   │   ├── preprocessing.py
│   │   ├── splits.py
│   │   └── dataset.py
│   │
│   ├── models/
│   │   ├── cast_gnn.py
│   │   ├── temporal_encoder.py
│   │   ├── graph_builder.py
│   │   ├── graph_adapter.py
│   │   ├── graph_attention.py
│   │   ├── tcn.py
│   │   ├── heads.py
│   │   └── baselines.py
│   │
│   ├── training/
│   │   ├── losses.py
│   │   ├── trainer.py
│   │   ├── train_within_dataset.py
│   │   ├── train_transfer.py
│   │   ├── train_joint.py
│   │   └── early_stopping.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── calibration.py
│   │   ├── robustness.py
│   │   ├── ablation.py
│   │   ├── interpretability.py
│   │   ├── statistical_tests.py
│   │   └── report_tables.py
│   │
│   ├── utils/
│   │   ├── config.py
│   │   ├── seed.py
│   │   ├── logging.py
│   │   ├── io.py
│   │   └── device.py
│   │
│   └── main.py
│
├── scripts/
│   ├── audit_eegmmidb.py
│   ├── audit_milimbeeg.py
│   ├── preprocess_eegmmidb.py
│   ├── preprocess_milimbeeg.py
│   ├── run_within_eegmmidb.py
│   ├── run_within_milimbeeg.py
│   ├── run_transfer.py
│   ├── run_joint_training.py
│   ├── run_ablation.py
│   ├── run_robustness.py
│   ├── run_calibration.py
│   └── run_interpretability.py
│
├── outputs/
│   ├── audits/
│   ├── checkpoints/
│   ├── logs/
│   ├── tables/
│   └── figures/
│
├── requirements.txt
├── environment.yml
├── run_all.sh
└── README.md
```

---

## 3. Environment Setup

### 3.1 Create a Conda Environment

```bash
conda env create -f environment.yml
conda activate cast-gnn
```

### 3.2 Alternative Pip Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3.3 Recommended Python Version

Use Python 3.10 or later.

---

## 4. Data Preparation

### 4.1 EEGMMIDB

Place raw PhysioNet EEGMMIDB EDF files in:

```text
data/eegmmidb_raw/
```

Expected properties:

- EDF format.
- 109 subjects.
- 14 runs per subject.
- 64 EEG channels.
- Runs `R01` and `R02` are baseline runs.
- Runs `R03`–`R14` contain motor execution and motor imagery tasks.
- Raw annotations `T0`, `T1`, and `T2` must be reconstructed using run-specific mappings.

Important: do not treat `T1` and `T2` as fixed labels across all runs.

### 4.2 MILimbEEG

Place raw MILimbEEG CSV files in:

```text
data/milimbeeg_raw/
```

Expected properties:

- CSV format.
- 60 subjects.
- 16 OpenBCI channels.
- Approximately 4 s trials at 125 Hz.
- Labels include `BEO`, `Rest`, `CLH`, `CRH`, `DLF`, `PLF`, `DRF`, and `PRF`.
- Execution/imagery mode should be parsed from folder or file metadata.

Important: `BEO` and `Rest` should remain separate unless an experiment explicitly defines a merged non-task state.

---

## 5. Configuration Files

Configuration files are stored in:

```text
configs/
```

Recommended usage:

```bash
python -m src.main \
  --mode train_within \
  --dataset eegmmidb \
  --task B_execution_vs_imagery \
  --config configs/default.yaml configs/eegmmidb.yaml
```

Multiple YAML files can be supplied. Later files override earlier files.

Example configuration fields:

```yaml
seed: 42
batch_size: 32
learning_rate: 0.001
weight_decay: 0.0001
num_epochs: 100
device: auto
sampling_rate: 125
window_seconds: 4.0
bands:
  - [5, 8]
  - [8, 13]
  - [13, 30]
  - [30, 40]
```

---

## 6. Supported Task Definitions

The implementation follows leakage-safe label harmonization.

| Task | Name | Cross-dataset use |
|---|---|---|
| A1 | Baseline/non-task vs task | Partial |
| A2 | Within/inter-task rest vs task | Moderate |
| B | Execution vs imagery | Yes |
| C | Upper vs lower limb | Coarse |
| D | Left vs right hand/fist | Yes |
| E | Multiclass motor task | Within-dataset only |
| F | Fine-grained action recognition | Within-dataset only |

Use direct cross-dataset transfer primarily for Tasks **B**, **C**, and **D**.

---

## 7. Scientific Validity Rules

This repository enforces several rules to reduce leakage and invalid comparison.

1. **Subject-wise splitting only**  
   Epochs or trials from the same subject must not appear in more than one split.

2. **Training-only normalization**  
   Normalization statistics must be estimated from training subjects only.

3. **Training-only functional adjacency**  
   Functional connectivity graphs must be estimated using training subjects only.

4. **Run-specific EEGMMIDB labels**  
   `T0`, `T1`, and `T2` must be interpreted according to the run number.

5. **Dataset-specific graph adapters**  
   EEGMMIDB and MILimbEEG should not be forced into direct raw-channel equivalence.

6. **Validation-only temperature scaling**  
   Temperature scaling must be fitted on validation logits only.

7. **Interpretability caution**  
   Channel, edge, and band importance results describe model behaviour, not causal neurophysiology.

---

## 8. Audit Pipeline

Run dataset audits before preprocessing or training.

```bash
python scripts/audit_eegmmidb.py --config configs/eegmmidb.yaml
python scripts/audit_milimbeeg.py --config configs/milimbeeg.yaml
```

Audit outputs are saved in:

```text
outputs/audits/
```

The audit stage should verify:

- File readability.
- Subject IDs.
- Run IDs.
- Channel counts.
- Sampling rates.
- Signal shapes.
- Missing values.
- Non-finite values.
- Flat channels.
- Abnormal amplitude.
- Duration or sample-count outliers.
- Malformed filenames.

---

## 9. Preprocessing Pipeline

Run preprocessing after audit completion.

```bash
python scripts/preprocess_eegmmidb.py --config configs/eegmmidb.yaml
python scripts/preprocess_milimbeeg.py --config configs/milimbeeg.yaml
```

Preprocessing includes:

- 5–40 Hz bandpass filtering.
- Frequency-band decomposition.
- EEGMMIDB resampling to 125 Hz for cross-dataset experiments.
- 4 s epoch/trial standardization.
- Subject-wise metadata preservation.
- Training-only normalization.
- Processed metadata and split-file generation.

Recommended sample shape:

```text
[B, C, T]
```

Where:

- `B = 4` spectral bands.
- `C = 64` for EEGMMIDB or `16` for MILimbEEG.
- `T ≈ 500` time samples.

---

## 10. Training Commands

### 10.1 Within-Dataset EEGMMIDB

```bash
python -m src.main \
  --mode train_within \
  --dataset eegmmidb \
  --task B_execution_vs_imagery \
  --config configs/default.yaml configs/eegmmidb.yaml
```

### 10.2 Within-Dataset MILimbEEG

```bash
python -m src.main \
  --mode train_within \
  --dataset milimbeeg \
  --task B_execution_vs_imagery \
  --config configs/default.yaml configs/milimbeeg.yaml
```

### 10.3 EEGMMIDB to MILimbEEG Transfer

```bash
python -m src.main \
  --mode train_transfer \
  --source eegmmidb \
  --target milimbeeg \
  --task B_execution_vs_imagery \
  --config configs/default.yaml configs/eegmmidb.yaml configs/milimbeeg.yaml configs/joint.yaml
```

### 10.4 MILimbEEG to EEGMMIDB Transfer

```bash
python -m src.main \
  --mode train_transfer \
  --source milimbeeg \
  --target eegmmidb \
  --task B_execution_vs_imagery \
  --config configs/default.yaml configs/milimbeeg.yaml configs/eegmmidb.yaml configs/joint.yaml
```

### 10.5 Joint Training

```bash
python -m src.main \
  --mode train_joint \
  --task B_execution_vs_imagery \
  --config configs/default.yaml configs/joint.yaml
```

---

## 11. Model Components

CAST-GNN includes the following modules:

1. **Dataset-specific input stems**  
   Separate stems are used for EEGMMIDB and MILimbEEG.

2. **Band-aware temporal encoder**  
   Multi-scale temporal convolutions process frequency-band-specific EEG signals.

3. **Hybrid graph construction**  
   The final adjacency combines anatomical, functional, and learnable graphs.

4. **Graph adapter**  
   Dataset-specific channel graphs are projected to a shared latent graph space.

5. **Graph attention encoder**  
   Adaptive graph attention learns task-relevant node interactions.

6. **Lightweight temporal dependency module**  
   A compact temporal convolutional network models sequence-level dynamics.

7. **Task-specific heads**  
   Separate prediction heads are used for valid task definitions.

---

## 12. Loss Functions

The training objective is:

```text
L_total = L_cls + lambda1 * L_CORAL + lambda2 * L_graph
```

Where:

- `L_cls` is supervised classification loss.
- `L_CORAL` aligns second-order source and target feature statistics.
- `L_graph` regularizes the learnable graph.
- `lambda1` is typically `0.05` for transfer and joint experiments.
- `lambda2` is typically `1e-4`.

---

## 13. Evaluation

The evaluation package supports:

- Accuracy.
- Macro-F1.
- Weighted-F1.
- Balanced accuracy.
- Cohen’s kappa.
- Matthews correlation coefficient.
- ROC-AUC where valid.
- Confusion matrix.
- Expected calibration error.
- Brier score.
- Mean ± SD across seeds.
- 95% confidence intervals.

Generated result tables are stored in:

```text
outputs/tables/
```

Generated figures are stored in:

```text
outputs/figures/
```

---

## 14. Calibration

Calibration analysis includes:

- Expected calibration error with 15 bins.
- Brier score.
- Validation-only temperature scaling.
- Reliability diagram generation.

Temperature scaling must be fitted only on validation logits and then applied to test logits.

---

## 15. Robustness Testing

Robustness testing supports:

- Gaussian noise.
- Channel dropout.
- Temporal cropping.
- Amplitude scaling.
- Band-specific perturbation.

Robustness outputs should report both clean performance and perturbed performance, including relative degradation.

---

## 16. Interpretability

Interpretability utilities include:

- Channel perturbation importance.
- Band masking importance.
- Attention relevance aggregation.
- Edge occlusion interface.

Recommended output files:

```text
outputs/tables/channel_relevance.csv
outputs/tables/band_importance.csv
outputs/tables/edge_importance.csv
outputs/figures/interpretability_summary.png
```

Interpretability outputs should be described as model-behaviour evidence only.

---

## 17. Repeated-Seed Experiments

For publication-level reporting, run each experiment over multiple seeds.

Recommended seed list:

```text
42, 123, 2024, 3407, 777
```

Example:

```bash
for SEED in 42 123 2024 3407 777
do
  python -m src.main \
    --mode train_within \
    --dataset eegmmidb \
    --task B_execution_vs_imagery \
    --seed $SEED \
    --config configs/default.yaml configs/eegmmidb.yaml
done
```

Report results as:

```text
mean ± standard deviation
```

or:

```text
mean (95% CI)
```

---

## 18. Outputs

The expected output structure is:

```text
outputs/
├── audits/
│   ├── eegmmidb_audit.csv
│   └── milimbeeg_audit.csv
│
├── checkpoints/
│   └── best_model.pt
│
├── logs/
│   └── train.log
│
├── tables/
│   ├── within_dataset_results.csv
│   ├── transfer_results.csv
│   ├── joint_training_results.csv
│   ├── ablation_results.csv
│   ├── robustness_results.csv
│   └── calibration_results.csv
│
└── figures/
    ├── confusion_matrix.png
    ├── reliability_diagram.png
    └── interpretability_summary.png
```

---

## 19. Troubleshooting

### Import errors

Run commands from the project root:

```bash
python -m src.main --help
```

If needed, set the Python path manually:

```bash
export PYTHONPATH=.
```

For Windows PowerShell:

```powershell
$env:PYTHONPATH="."
```

### CUDA unavailable

Use CPU mode:

```yaml
device: cpu
```

or:

```bash
python -m src.main --mode train_within --dataset eegmmidb --task B_execution_vs_imagery --config configs/default.yaml
```

### Empty task labels

Check that label harmonization has generated the selected task column. For cross-dataset transfer, use only valid shared tasks.

### Poor or suspiciously high performance

Check:

- Whether subject-wise splitting was preserved.
- Whether test subjects were used in normalization.
- Whether functional adjacency was estimated using test data.
- Whether EEGMMIDB `T1/T2` labels were mapped correctly.
- Whether `BEO` and `Rest` were merged unintentionally.

---

## 20. Citation and Reporting Notes

When using this repository in a manuscript, report:

- Dataset version and download source.
- Number of included subjects.
- Number of excluded files or trials.
- Sampling-rate handling.
- Label harmonization rules.
- Subject-wise split protocol.
- Number of random seeds.
- Primary metrics.
- Calibration method.
- Robustness perturbations.
- Interpretability method and limitations.

Do not report generated results unless they were obtained from actual model runs on the audited datasets.

---

## 21. Current Development Status

The repository currently contains:

- Project configuration.
- Data loading and preprocessing modules.
- CAST-GNN model modules.
- Baseline model modules.
- Training modules.
- Evaluation modules.
- Utility modules.
- Unified command-line entry point.

Before running full experiments, verify that your dataset paths, metadata column names, and processed-file paths match the configuration files.

---

## 22. License

Add a license before public release. If the project is intended for academic sharing, consider MIT, Apache-2.0, or BSD-3-Clause depending on institutional requirements.

---

## 23. Contact

For project-specific questions, update this section with the corresponding author or maintainer information.
