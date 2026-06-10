"""Label reconstruction and task harmonisation for CAST-GNN.

The functions in this module implement the scientifically constrained task
mapping used by the project.  The central rule is that EEGMMIDB T0/T1/T2
annotations are not fixed semantic labels; T1 and T2 depend on the run number.
MILimbEEG labels are parsed from file/folder metadata and BEO and Rest are kept
separate unless a task explicitly permits a merged non-task state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import pandas as pd


EEGMMIDB_BASELINE_RUNS = {1: "baseline_eyes_open", 2: "baseline_eyes_closed"}

# Official EEGMMIDB motor run semantics.
# T0 is rest within a task run. T1/T2 depend on the run group.
EEGMMIDB_RUN_LABEL_MAP: Dict[int, Dict[str, Dict[str, str]]] = {
    3: {
        "T0": {"state": "rest", "mode": "rest", "body_part": "none", "semantic_label": "rest"},
        "T1": {"state": "task", "mode": "execution", "body_part": "left_fist", "semantic_label": "left_fist_execution"},
        "T2": {"state": "task", "mode": "execution", "body_part": "right_fist", "semantic_label": "right_fist_execution"},
    },
    4: {
        "T0": {"state": "rest", "mode": "rest", "body_part": "none", "semantic_label": "rest"},
        "T1": {"state": "task", "mode": "imagery", "body_part": "left_fist", "semantic_label": "left_fist_imagery"},
        "T2": {"state": "task", "mode": "imagery", "body_part": "right_fist", "semantic_label": "right_fist_imagery"},
    },
    5: {
        "T0": {"state": "rest", "mode": "rest", "body_part": "none", "semantic_label": "rest"},
        "T1": {"state": "task", "mode": "execution", "body_part": "both_fists", "semantic_label": "both_fists_execution"},
        "T2": {"state": "task", "mode": "execution", "body_part": "both_feet", "semantic_label": "both_feet_execution"},
    },
    6: {
        "T0": {"state": "rest", "mode": "rest", "body_part": "none", "semantic_label": "rest"},
        "T1": {"state": "task", "mode": "imagery", "body_part": "both_fists", "semantic_label": "both_fists_imagery"},
        "T2": {"state": "task", "mode": "imagery", "body_part": "both_feet", "semantic_label": "both_feet_imagery"},
    },
}

# Repeat run semantics for the three task blocks.
for _base in (7, 11):
    EEGMMIDB_RUN_LABEL_MAP[_base] = EEGMMIDB_RUN_LABEL_MAP[3]
for _base in (8, 12):
    EEGMMIDB_RUN_LABEL_MAP[_base] = EEGMMIDB_RUN_LABEL_MAP[4]
for _base in (9, 13):
    EEGMMIDB_RUN_LABEL_MAP[_base] = EEGMMIDB_RUN_LABEL_MAP[5]
for _base in (10, 14):
    EEGMMIDB_RUN_LABEL_MAP[_base] = EEGMMIDB_RUN_LABEL_MAP[6]

MILIMBEEG_MOVEMENT_MAP: Dict[str, Dict[str, str]] = {
    "BEO": {"state": "baseline", "mode": "baseline", "body_part": "none", "semantic_label": "baseline_eyes_open"},
    "REST": {"state": "rest", "mode": "rest", "body_part": "none", "semantic_label": "rest"},
    "CLH": {"state": "task", "body_part": "left_hand", "semantic_label": "closing_left_hand"},
    "CRH": {"state": "task", "body_part": "right_hand", "semantic_label": "closing_right_hand"},
    "DLF": {"state": "task", "body_part": "left_foot", "semantic_label": "dorsiflexion_left_foot"},
    "PLF": {"state": "task", "body_part": "left_foot", "semantic_label": "plantar_flexion_left_foot"},
    "DRF": {"state": "task", "body_part": "right_foot", "semantic_label": "dorsiflexion_right_foot"},
    "PRF": {"state": "task", "body_part": "right_foot", "semantic_label": "plantar_flexion_right_foot"},
}

TASK_COLUMNS = {
    "A1_non_task_vs_task": "task_a1",
    "A2_rest_vs_task": "task_a2",
    "B_execution_vs_imagery": "task_b",
    "C_upper_vs_lower": "task_c",
    "D_left_vs_right": "task_d",
    "E_multiclass_motor": "task_e",
    "F_fine_grained": "task_f",
}

TASK_NUM_CLASSES = {
    "A1_non_task_vs_task": 2,
    "A2_rest_vs_task": 2,
    "B_execution_vs_imagery": 2,
    "C_upper_vs_lower": 2,
    "D_left_vs_right": 2,
    "E_multiclass_motor": {"eegmmidb": 4, "milimbeeg": 6},
    "F_fine_grained": {"eegmmidb": 8, "milimbeeg": 12},
}

CROSS_DATASET_TASKS = {"B_execution_vs_imagery", "C_upper_vs_lower", "D_left_vs_right"}
WITHIN_DATASET_ONLY_TASKS = {"E_multiclass_motor", "F_fine_grained"}


@dataclass(frozen=True)
class HarmonizedLabel:
    """Container for a reconstructed EEG semantic label."""

    dataset: str
    state: str
    mode: str
    body_part: str
    semantic_label: str
    raw_label: str
    run_id: Optional[int] = None


def _safe_upper(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().upper()


def reconstruct_eegmmidb_label(run_id: int, annotation: str) -> Optional[Dict[str, Any]]:
    """Reconstruct a semantic EEGMMIDB label from run ID and raw annotation.

    Parameters
    ----------
    run_id:
        EEGMMIDB run number. R01/R02 are baseline; R03-R14 are task runs.
    annotation:
        Raw EDF annotation label, usually T0, T1, or T2.

    Returns
    -------
    dict or None
        Semantic label dictionary, or None if the annotation cannot be mapped.
    """
    ann = _safe_upper(annotation)
    run_id = int(run_id)

    if run_id in EEGMMIDB_BASELINE_RUNS:
        return {
            "dataset": "eegmmidb",
            "state": "baseline",
            "mode": "baseline",
            "body_part": "none",
            "semantic_label": EEGMMIDB_BASELINE_RUNS[run_id],
            "raw_label": ann,
            "run_id": run_id,
        }

    if run_id not in EEGMMIDB_RUN_LABEL_MAP:
        return None
    if ann not in EEGMMIDB_RUN_LABEL_MAP[run_id]:
        return None

    mapped = dict(EEGMMIDB_RUN_LABEL_MAP[run_id][ann])
    mapped.update({"dataset": "eegmmidb", "raw_label": ann, "run_id": run_id})
    return mapped


def parse_milimbeeg_label(metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert parsed MILimbEEG file metadata to a semantic label dictionary.

    Parameters
    ----------
    metadata:
        Metadata dictionary produced by the MILimbEEG loader. It should include
        at least `movement_label` and, for task trials, a `mode` value.
    """
    movement = _safe_upper(metadata.get("movement_label") or metadata.get("label"))
    if movement == "REST":
        movement = "REST"
    if movement not in MILIMBEEG_MOVEMENT_MAP:
        return None

    mapped = dict(MILIMBEEG_MOVEMENT_MAP[movement])
    raw_mode = str(metadata.get("mode", "")).lower().strip()
    if mapped["state"] == "task":
        if raw_mode in {"execution", "exec", "me", "motor_execution", "real", "movement"}:
            mode = "execution"
        elif raw_mode in {"imagery", "imagination", "imagined", "mi", "motor_imagery"}:
            mode = "imagery"
        else:
            mode = "unknown"
        mapped["mode"] = mode
        mapped["semantic_label"] = f"{mapped['semantic_label']}_{mode}"
    else:
        mapped.setdefault("mode", mapped["state"])

    mapped.update({
        "dataset": "milimbeeg",
        "raw_label": movement,
        "run_id": None,
    })
    return mapped


def is_task(row: Dict[str, Any] | pd.Series) -> bool:
    return str(row.get("state", "")).lower() == "task"


def is_rest(row: Dict[str, Any] | pd.Series) -> bool:
    return str(row.get("state", "")).lower() == "rest"


def is_baseline(row: Dict[str, Any] | pd.Series) -> bool:
    return str(row.get("state", "")).lower() == "baseline"


def assign_task_a1(row: Dict[str, Any] | pd.Series) -> Optional[int]:
    """Task A1: baseline/non-task versus motor task.

    Label convention: 0 = baseline/non-task, 1 = task.
    """
    if is_task(row):
        return 1
    if is_baseline(row):
        return 0
    return None


def assign_task_a2(row: Dict[str, Any] | pd.Series) -> Optional[int]:
    """Task A2: within/inter-task rest versus motor task.

    Label convention: 0 = rest, 1 = task.
    """
    if is_task(row):
        return 1
    if is_rest(row):
        return 0
    return None


def assign_task_b_execution_vs_imagery(row: Dict[str, Any] | pd.Series) -> Optional[int]:
    """Task B: motor execution versus motor imagery.

    Label convention: 0 = execution, 1 = imagery. Non-task samples are excluded.
    """
    if not is_task(row):
        return None
    mode = str(row.get("mode", "")).lower()
    if mode == "execution":
        return 0
    if mode == "imagery":
        return 1
    return None


def assign_task_c_upper_vs_lower(row: Dict[str, Any] | pd.Series) -> Optional[int]:
    """Task C: upper-limb versus lower-limb motor decoding.

    Label convention: 0 = upper limb, 1 = lower limb.
    """
    if not is_task(row):
        return None
    body = str(row.get("body_part", "")).lower()
    if body in {"left_fist", "right_fist", "both_fists", "left_hand", "right_hand"}:
        return 0
    if body in {"both_feet", "left_foot", "right_foot"}:
        return 1
    return None


def assign_task_d_left_vs_right(row: Dict[str, Any] | pd.Series) -> Optional[int]:
    """Task D: left-versus-right hand/fist laterality.

    Label convention: 0 = left hand/fist, 1 = right hand/fist. Bilateral and foot
    labels are excluded to avoid invalid cross-dataset mappings.
    """
    if not is_task(row):
        return None
    body = str(row.get("body_part", "")).lower()
    if body in {"left_fist", "left_hand"}:
        return 0
    if body in {"right_fist", "right_hand"}:
        return 1
    return None


def assign_task_e_multiclass(row: Dict[str, Any] | pd.Series) -> Optional[int]:
    """Task E: within-dataset multiclass motor-task classification.

    EEGMMIDB: left fist, right fist, both fists, both feet.
    MILimbEEG: CLH, CRH, DLF, PLF, DRF, PRF.
    """
    if not is_task(row):
        return None
    dataset = str(row.get("dataset", "")).lower()
    body = str(row.get("body_part", "")).lower()
    raw = _safe_upper(row.get("raw_label"))
    if dataset == "eegmmidb":
        mapping = {"left_fist": 0, "right_fist": 1, "both_fists": 2, "both_feet": 3}
        return mapping.get(body)
    if dataset == "milimbeeg":
        mapping = {"CLH": 0, "CRH": 1, "DLF": 2, "PLF": 3, "DRF": 4, "PRF": 5}
        return mapping.get(raw)
    return None


def assign_task_f_fine_grained(row: Dict[str, Any] | pd.Series) -> Optional[int]:
    """Task F: dataset-specific fine-grained action recognition.

    EEGMMIDB has eight body-part × mode labels. MILimbEEG has twelve
    movement-label × mode labels.
    """
    if not is_task(row):
        return None
    dataset = str(row.get("dataset", "")).lower()
    mode = str(row.get("mode", "")).lower()
    e_label = assign_task_e_multiclass(row)
    if e_label is None or mode not in {"execution", "imagery"}:
        return None
    mode_offset = 0 if mode == "execution" else 1
    if dataset == "eegmmidb":
        return e_label * 2 + mode_offset
    if dataset == "milimbeeg":
        return e_label * 2 + mode_offset
    return None


def add_all_task_labels(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Add task-specific label columns to a metadata DataFrame.

    The returned DataFrame uses pandas nullable integer dtype (`Int64`) for task
    labels, so missing labels are stored as NA rather than a fabricated class.
    """
    df = metadata_df.copy()
    df["task_a1"] = df.apply(assign_task_a1, axis=1).astype("Int64")
    df["task_a2"] = df.apply(assign_task_a2, axis=1).astype("Int64")
    df["task_b"] = df.apply(assign_task_b_execution_vs_imagery, axis=1).astype("Int64")
    df["task_c"] = df.apply(assign_task_c_upper_vs_lower, axis=1).astype("Int64")
    df["task_d"] = df.apply(assign_task_d_left_vs_right, axis=1).astype("Int64")
    df["task_e"] = df.apply(assign_task_e_multiclass, axis=1).astype("Int64")
    df["task_f"] = df.apply(assign_task_f_fine_grained, axis=1).astype("Int64")
    return df


def validate_cross_dataset_task(task_name: str) -> None:
    """Raise a ValueError if a task is invalid for cross-dataset evaluation."""
    if task_name not in CROSS_DATASET_TASKS:
        raise ValueError(
            f"Task {task_name!r} is not valid for direct cross-dataset evaluation. "
            f"Use one of {sorted(CROSS_DATASET_TASKS)}."
        )


def get_task_column(task_name: str) -> str:
    """Return the metadata column associated with a task name."""
    if task_name not in TASK_COLUMNS:
        raise KeyError(f"Unknown task name {task_name!r}. Available tasks: {list(TASK_COLUMNS)}")
    return TASK_COLUMNS[task_name]
