
from __future__ import annotations

from pathlib import Path
import json
from typing import Any
import numpy as np
import matplotlib.pyplot as plt

STAGE3_LABELS = ["quiet", "B", "C", "M", "X"]


def row_pct_matrix(cm) -> np.ndarray:
    cm = np.asarray(cm, dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return (cm / row_sums) * 100.0


def _coerce_cm(cm_like: Any) -> np.ndarray:
    """
    Convert a confusion-matrix-like object into a dense numpy array.

    Supports:
    - list[list[int]]
    - numpy arrays
    - dict-of-dicts as produced by DataFrame.to_dict()
    - dict with numeric string keys
    """
    if cm_like is None:
        raise ValueError("Confusion matrix is None.")

    if isinstance(cm_like, np.ndarray):
        return cm_like.astype(int)

    if isinstance(cm_like, list):
        return np.asarray(cm_like, dtype=int)

    if isinstance(cm_like, dict):
        # dict-of-dicts style
        try:
            outer_keys = sorted(cm_like.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
            rows = []
            for rk in outer_keys:
                row_obj = cm_like[rk]
                if isinstance(row_obj, dict):
                    inner_keys = sorted(row_obj.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
                    rows.append([row_obj[ck] for ck in inner_keys])
                else:
                    rows.append(row_obj)
            return np.asarray(rows, dtype=int)
        except Exception as e:
            raise ValueError(f"Could not coerce confusion matrix from dict format: {e}") from e

    raise TypeError(f"Unsupported confusion matrix type: {type(cm_like)}")


def show_stage3_cm(
    cm_like: Any,
    labels=STAGE3_LABELS,
    title: str = "Stage 3 confusion matrix",
    figsize=(6.2, 5.2),
    cmap="Blues",
    vmin: float = 0.0,
    vmax: float = 100.0,
):
    cm = _coerce_cm(cm_like)
    pct = row_pct_matrix(cm)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    im = ax.imshow(pct, cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i,
                f"{int(cm[i, j])}\n{pct[i, j]:.1f}%",
                ha="center", va="center", fontsize=9
            )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row %")
    plt.show()


def show_stage3_cm_pair(
    val_cm_like: Any,
    test_cm_like: Any,
    labels=STAGE3_LABELS,
    title_prefix: str = "Stage 3",
    figsize=(13.2, 5.2),
    cmap="Blues",
    vmin: float = 0.0,
    vmax: float = 100.0,
):
    val_cm = _coerce_cm(val_cm_like)
    test_cm = _coerce_cm(test_cm_like)

    val_pct = row_pct_matrix(val_cm)
    test_pct = row_pct_matrix(test_cm)

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    for ax, cm, pct, split in [
        (axes[0], val_cm, val_pct, "VAL"),
        (axes[1], test_cm, test_pct, "TEST"),
    ]:
        im = ax.imshow(pct, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{title_prefix} / {split}")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i,
                    f"{int(cm[i, j])}\n{pct[i, j]:.1f}%",
                    ha="center", va="center", fontsize=9
                )

    cbar = fig.colorbar(im, ax=axes, shrink=0.92, pad=0.02)
    cbar.set_label("Row %")
    plt.show()


def load_stage3_summary(summary_path: str | Path) -> dict:
    path = Path(summary_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing summary file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def choose_stage3_best_run(obj: dict) -> dict | None:
    """
    Match the chosen run from a stage3 summary object using:
    - best.t_reject
    - best.ckpt_path (if available)
    """
    best = obj.get("best", {})
    best_t_reject = best.get("t_reject")
    best_ckpt = best.get("ckpt_path")

    chosen = None
    for run in obj.get("runs", []):
        same_t = run.get("t_reject") == best_t_reject
        same_ckpt = (best_ckpt is None) or (run.get("ckpt_path") == best_ckpt)
        if same_t and same_ckpt:
            chosen = run
            break

    if chosen is None:
        for run in obj.get("runs", []):
            if run.get("t_reject") == best_t_reject:
                chosen = run
                break

    return chosen


def _extract_cm_from_split_obj(split_obj: dict | None) -> Any | None:
    if not split_obj:
        return None

    # Try common locations
    for key in ["confusion_matrix", "cm", "confusion"]:
        if key in split_obj and split_obj[key] is not None:
            return split_obj[key]

    # Nested metrics/report structures if present
    metrics = split_obj.get("metrics")
    if isinstance(metrics, dict):
        for key in ["confusion_matrix", "cm", "confusion"]:
            if key in metrics and metrics[key] is not None:
                return metrics[key]

    return None


def get_stage3_best_cms(summary_path: str | Path) -> tuple[Any | None, Any | None, dict, dict | None]:
    """
    Returns:
      val_cm, test_cm, full_summary_obj, chosen_run
    """
    obj = load_stage3_summary(summary_path)
    chosen = choose_stage3_best_run(obj)

    val_cm = None
    test_cm = None

    # First try chosen run
    if chosen is not None:
        val_cm = _extract_cm_from_split_obj(chosen.get("val"))
        test_cm = _extract_cm_from_split_obj(chosen.get("test"))

    # Fallback to summary-level val/test objects
    if val_cm is None:
        val_cm = _extract_cm_from_split_obj(obj.get("val"))
    if test_cm is None:
        test_cm = _extract_cm_from_split_obj(obj.get("test"))

    return val_cm, test_cm, obj, chosen


def show_stage3_best_pair_from_summary(
    summary_path: str | Path,
    labels=STAGE3_LABELS,
    title_prefix: str | None = None,
):
    summary_path = Path(summary_path)
    val_cm, test_cm, obj, chosen = get_stage3_best_cms(summary_path)

    if val_cm is None and test_cm is None:
        raise ValueError(f"No confusion matrices found in: {summary_path}")

    if title_prefix is None:
        title_prefix = summary_path.parent.name

    if val_cm is not None and test_cm is not None:
        show_stage3_cm_pair(val_cm, test_cm, labels=labels, title_prefix=title_prefix)
    elif val_cm is not None:
        show_stage3_cm(val_cm, labels=labels, title=f"{title_prefix} / VAL")
    elif test_cm is not None:
        show_stage3_cm(test_cm, labels=labels, title=f"{title_prefix} / TEST")


def print_stage3_best_settings(summary_path: str | Path):
    """
    Print a compact chosen-setting block under the matrix, similar to the Stage 2 chosen-row print style.
    """
    _, _, obj, chosen = get_stage3_best_cms(summary_path)
    best = obj.get("best", {})
    run = chosen if chosen is not None else {}

    t_flare = run.get("t_flare", best.get("t_flare"))
    t_reject = run.get("t_reject", best.get("t_reject"))
    t_conf = best.get("t_conf_bcmx", run.get("t_conf_bcmx"))

    print("Chosen Stage 3 operating point:")
    if t_flare is not None:
        print(f"  t_flare      = {t_flare}")
    if t_reject is not None:
        print(f"  t_reject     = {t_reject}")
    if t_conf is not None:
        print(f"  t_conf_bcmx  = {t_conf}")

    
