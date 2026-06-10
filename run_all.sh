#!/usr/bin/env bash
#
# End-to-end CAST-GNN pipeline runner.
#
# This script assumes that:
#   1. Raw EEGMMIDB EDF files are placed under data/eegmmidb_raw/
#   2. Raw MILimbEEG CSV files are placed under data/milimbeeg_raw/
#   3. Config files exist under configs/
#   4. The src/ package has been copied into the project root
#
# Usage examples:
#   bash run_all.sh
#   bash run_all.sh --skip-audit
#   bash run_all.sh --task B_execution_vs_imagery --seeds "42 43 44 45 46"

set -euo pipefail

TASK="B_execution_vs_imagery"
SEEDS="42 43 44 45 46"
SKIP_AUDIT=0
SKIP_PREPROCESS=0
RUN_WITHIN=1
RUN_TRANSFER=1
RUN_JOINT=1
RUN_ABLATION=1
RUN_ROBUSTNESS=1
RUN_CALIBRATION=1
RUN_INTERPRETABILITY=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)
      TASK="$2"
      shift 2
      ;;
    --seeds)
      SEEDS="$2"
      shift 2
      ;;
    --skip-audit)
      SKIP_AUDIT=1
      shift
      ;;
    --skip-preprocess)
      SKIP_PREPROCESS=1
      shift
      ;;
    --within-only)
      RUN_TRANSFER=0
      RUN_JOINT=0
      RUN_ABLATION=0
      RUN_ROBUSTNESS=0
      RUN_CALIBRATION=0
      RUN_INTERPRETABILITY=0
      shift
      ;;
    --no-transfer)
      RUN_TRANSFER=0
      shift
      ;;
    --no-joint)
      RUN_JOINT=0
      shift
      ;;
    --no-ablation)
      RUN_ABLATION=0
      shift
      ;;
    --no-robustness)
      RUN_ROBUSTNESS=0
      shift
      ;;
    --no-calibration)
      RUN_CALIBRATION=0
      shift
      ;;
    --no-interpretability)
      RUN_INTERPRETABILITY=0
      shift
      ;;
    -h|--help)
      echo "Usage: bash run_all.sh [--task TASK] [--seeds "42 43 44"] [--skip-audit] [--skip-preprocess]"
      echo "                         [--within-only] [--no-transfer] [--no-joint]"
      echo "                         [--no-ablation] [--no-robustness] [--no-calibration] [--no-interpretability]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

mkdir -p outputs/audits outputs/checkpoints outputs/logs outputs/tables outputs/figures

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "=============================================="
echo "CAST-GNN end-to-end pipeline"
echo "Task: ${TASK}"
echo "Seeds: ${SEEDS}"
echo "Project root: ${PROJECT_ROOT}"
echo "=============================================="

echo "Checking Python imports..."
python - <<'PY'
import importlib
required = ["numpy", "pandas", "scipy", "sklearn", "mne", "torch", "matplotlib", "yaml", "tqdm"]
missing = []
for pkg in required:
    try:
        importlib.import_module(pkg)
    except Exception:
        missing.append(pkg)
if missing:
    raise SystemExit(f"Missing required packages: {missing}")
print("Import check passed.")
PY

if [[ "${SKIP_AUDIT}" -eq 0 ]]; then
  echo "Running EEGMMIDB audit..."
  python scripts/audit_eegmmidb.py --config configs/eegmmidb.yaml

  echo "Running MILimbEEG audit..."
  python scripts/audit_milimbeeg.py --config configs/milimbeeg.yaml
else
  echo "Skipping audit."
fi

if [[ "${SKIP_PREPROCESS}" -eq 0 ]]; then
  echo "Preprocessing EEGMMIDB..."
  python scripts/preprocess_eegmmidb.py --config configs/eegmmidb.yaml

  echo "Preprocessing MILimbEEG..."
  python scripts/preprocess_milimbeeg.py --config configs/milimbeeg.yaml
else
  echo "Skipping preprocessing."
fi

for SEED in ${SEEDS}; do
  echo "----------------------------------------------"
  echo "Seed ${SEED}"
  echo "----------------------------------------------"

  if [[ "${RUN_WITHIN}" -eq 1 ]]; then
    echo "Within-dataset training: EEGMMIDB"
    python -m src.main \
      --mode train_within \
      --dataset eegmmidb \
      --task "${TASK}" \
      --seed "${SEED}" \
      --config configs/default.yaml configs/eegmmidb.yaml

    echo "Within-dataset training: MILimbEEG"
    python -m src.main \
      --mode train_within \
      --dataset milimbeeg \
      --task "${TASK}" \
      --seed "${SEED}" \
      --config configs/default.yaml configs/milimbeeg.yaml
  fi

  if [[ "${RUN_TRANSFER}" -eq 1 ]]; then
    echo "Transfer training: EEGMMIDB -> MILimbEEG"
    python -m src.main \
      --mode train_transfer \
      --source eegmmidb \
      --target milimbeeg \
      --task "${TASK}" \
      --seed "${SEED}" \
      --config configs/default.yaml configs/eegmmidb.yaml configs/milimbeeg.yaml configs/joint.yaml

    echo "Transfer training: MILimbEEG -> EEGMMIDB"
    python -m src.main \
      --mode train_transfer \
      --source milimbeeg \
      --target eegmmidb \
      --task "${TASK}" \
      --seed "${SEED}" \
      --config configs/default.yaml configs/milimbeeg.yaml configs/eegmmidb.yaml configs/joint.yaml
  fi

  if [[ "${RUN_JOINT}" -eq 1 ]]; then
    echo "Joint training with graph adapters and CORAL"
    python -m src.main \
      --mode train_joint \
      --task "${TASK}" \
      --seed "${SEED}" \
      --config configs/default.yaml configs/joint.yaml
  fi
done

if [[ "${RUN_ABLATION}" -eq 1 ]]; then
  echo "Running ablation experiments..."
  python scripts/run_ablation.py --config configs/default.yaml
fi

if [[ "${RUN_ROBUSTNESS}" -eq 1 ]]; then
  echo "Running robustness analysis..."
  python scripts/run_robustness.py --config configs/default.yaml
fi

if [[ "${RUN_CALIBRATION}" -eq 1 ]]; then
  echo "Running calibration analysis..."
  python scripts/run_calibration.py --config configs/default.yaml
fi

if [[ "${RUN_INTERPRETABILITY}" -eq 1 ]]; then
  echo "Running interpretability analysis..."
  python scripts/run_interpretability.py --config configs/default.yaml
fi

echo "=============================================="
echo "CAST-GNN pipeline completed."
echo "Outputs are available under outputs/"
echo "=============================================="
