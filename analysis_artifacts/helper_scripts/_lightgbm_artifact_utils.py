from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import classification_report, confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_ROOT = PROJECT_ROOT / "analysis_artifacts"

RUN_INV_DIR = PROJECT_ROOT / "lightgbm_run_inventory_20260412"
SECTION6_NOTEBOOK = PROJECT_ROOT / "section6_results_with_new_lightgbm.ipynb"

CLASSES = ["quiet", "B", "C", "M", "X"]
LABEL_MAPPING = {name: idx for idx, name in enumerate(CLASSES)}
ID_TO_NAME = {idx: name for name, idx in LABEL_MAPPING.items()}


def bootstrap_project_imports() -> None:
    import sys

    root_str = str(PROJECT_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


bootstrap_project_imports()

from train_lightgbm_section4 import (  # noqa: E402
    encode_y,
    iter_manifest_file_slices,
    load_manifest_obj,
    load_npz_arrays_with_retry,
    pooled_feature_names,
    pool_windows_stats_safe,
    predict_proba,
    predict_with_thresholds_ovr,
    resolve_npz_path,
)


def relpath_str(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT.resolve()))
    except Exception:
        return str(p).replace("/", "\\")


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text, encoding="utf-8")


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def compute_run_ranking() -> pd.DataFrame:
    comp = pd.read_csv(RUN_INV_DIR / "lightgbm_comparison_matrix.csv")
    pr = pd.read_csv(RUN_INV_DIR / "lightgbm_precision_recall_compare.csv")

    usable = comp.copy()
    usable["usable_for_comparison"] = usable["usable_for_comparison"].astype(bool)
    usable = usable[usable["usable_for_comparison"]].copy()

    best_exp_map = dict(zip(usable["run_key"], usable["best_5c_experiment"]))

    rows: list[dict[str, Any]] = []
    for _, row in usable.iterrows():
        run_key = row["run_key"]
        exp_name = f"{best_exp_map[run_key]}_5c"
        sub = pr[
            (pr["run_key"] == run_key)
            & (pr["task"] == "5c")
            & (pr["experiment"] == exp_name)
            & (pr["split"] == "test")
        ]
        if len(sub) != 1:
            macro_precision_5c = np.nan
            mean_flare_precision = np.nan
        else:
            s = sub.iloc[0]
            macro_precision_5c = float(
                np.mean(
                    [
                        s["quiet_precision"],
                        s["B_precision"],
                        s["C_precision"],
                        s["M_precision"],
                        s["X_precision"],
                    ]
                )
            )
            mean_flare_precision = float(
                np.mean([s["B_precision"], s["C_precision"], s["M_precision"], s["X_precision"]])
            )

        mean_flare_recall = float(
            np.mean(
                [
                    row["best_5c_test_B_recall"],
                    row["best_5c_test_C_recall"],
                    row["best_5c_test_M_recall"],
                    row["best_5c_test_X_recall"],
                ]
            )
        )

        rows.append(
            {
                "run_key": run_key,
                "window_hours": int(row["window_hours"]),
                "horizon_hours": int(row["horizon_hours"]),
                "experiment": str(row["best_5c_experiment"]),
                "split": "test",
                "macro_f1_5c": float(row["best_5c_test_macro_f1"]),
                "macro_precision_5c": macro_precision_5c,
                "mean_flare_recall": mean_flare_recall,
                "mean_flare_precision": mean_flare_precision,
                "usable_for_comparison": bool(row["usable_for_comparison"]),
                "selected_as_best": False,
                "source": row["source"],
                "status": row["status"],
                "interpretation": row["interpretation"],
            }
        )

    ranking = pd.DataFrame(rows).sort_values(
        ["macro_f1_5c", "mean_flare_recall", "macro_precision_5c"],
        ascending=[False, False, False],
    )
    if not ranking.empty:
        ranking.loc[ranking.index[0], "selected_as_best"] = True
    return ranking.reset_index(drop=True)


def choose_best_run(ranking: pd.DataFrame) -> dict[str, Any]:
    if ranking.empty:
        raise RuntimeError("No usable LightGBM runs were found.")

    best = ranking.iloc[0].to_dict()
    caveats: list[str] = []

    if best["run_key"] != "W72_H24":
        caveats.append(
            "Notebook text says W72_H24 is the strongest overall LightGBM baseline; computed ranking disagrees."
        )

    note = (
        "Selected by the Section 6 LightGBM ranking logic: among runs marked usable_for_comparison, "
        "choose the highest best 5-class test Macro-F1. The notebook explicitly states that W72_H24 is "
        "the strongest balanced 5-class LightGBM baseline."
    )
    best["reason_selected"] = note
    best["selection_caveats"] = caveats
    return best


def best_run_dir(run_key: str) -> Path:
    comp = pd.read_csv(RUN_INV_DIR / "lightgbm_comparison_matrix.csv")
    sub = comp[comp["run_key"] == run_key]
    if len(sub) != 1:
        raise RuntimeError(f"Could not uniquely locate comparison row for {run_key}")
    row = sub.iloc[0]
    source = str(row["source"]).strip().lower()
    if source == "night":
        return PROJECT_ROOT / "runs_lightgbm_night" / "20260411_124138" / run_key
    if source == "notebook":
        return PROJECT_ROOT / "runs_lightgbm_notebook" / run_key
    raise RuntimeError(f"Unsupported LightGBM run source for {run_key}: {row['source']!r}")


def selected_model_file(run_dir: Path, experiment: str) -> Path:
    if experiment == "baseline":
        name = "lgbm_baseline_5c.txt"
    elif experiment in {"weighted", "weighted_thresholded"}:
        name = "lgbm_weighted_5c.txt"
    else:
        raise ValueError(f"Unsupported experiment for 5c selection: {experiment}")
    return run_dir / name


def selected_thresholds_file(run_dir: Path, experiment: str) -> Path | None:
    if experiment == "weighted_thresholded":
        return run_dir / "thresholds_weighted_5c.json"
    return None


def relative_run_paths(run_key: str, experiment: str) -> dict[str, str]:
    run_dir = best_run_dir(run_key)
    model_path = selected_model_file(run_dir, experiment)
    metrics_path = run_dir / "metrics_full.json"
    thresholds_path = selected_thresholds_file(run_dir, experiment)
    payload = {
        "run_dir": relpath_str(run_dir),
        "model_file": relpath_str(model_path),
        "metrics_file": relpath_str(metrics_path),
    }
    if thresholds_path is not None:
        payload["thresholds_file"] = relpath_str(thresholds_path)
    return payload


def load_selected_run_payload() -> tuple[pd.DataFrame, dict[str, Any], Path, dict[str, Any]]:
    ranking = compute_run_ranking()
    best = choose_best_run(ranking)
    run_dir = best_run_dir(best["run_key"])
    metrics = load_json(run_dir / "metrics_full.json")
    return ranking, best, run_dir, metrics


def threshold_array_from_json(path: Path) -> np.ndarray:
    payload = load_json(path)
    return np.array([float(payload[c]) for c in CLASSES], dtype=np.float32)


def make_feature_names(feature_dim: int) -> list[str]:
    raw_f = feature_dim // 4
    return pooled_feature_names(raw_f)


def iter_pooled_rows(manifest_path: Path) -> Iterable[dict[str, Any]]:
    for chunk in iter_pooled_rows_with_status(manifest_path, allow_skip=False):
        if chunk["missing"]:
            raise RuntimeError(f"Missing chunk encountered unexpectedly for {chunk['source_file']}")
        yield chunk


def iter_pooled_rows_with_status(manifest_path: Path, allow_skip: bool = False) -> Iterable[dict[str, Any]]:
    manifest_obj, manifest_path = load_manifest_obj(manifest_path)
    entries = manifest_obj["entries"]

    for manifest_idx, ent in enumerate(entries):
        file_rel = str(ent["file"])
        fp = resolve_npz_path(file_rel, manifest_obj, manifest_path)
        slices = ent.get("slices") or [[0, None]]
        x_all, y_all = load_npz_arrays_with_retry(fp, allow_skip=allow_skip)

        if x_all is None or y_all is None:
            for slice_idx, (a, b) in enumerate(slices):
                start = int(a or 0)
                stop = int(b or start)
                yield {
                    "missing": True,
                    "manifest_entry_index": manifest_idx,
                    "slice_index": slice_idx,
                    "source_file": file_rel.replace("/", "\\"),
                    "row_in_file_start": start,
                    "row_in_file_stop": stop,
                    "row_count_expected": max(stop - start, 0),
                    "features": None,
                    "y_int": None,
                    "row_in_file": None,
                }
            continue

        for slice_idx, (a, b) in enumerate(slices):
            start = int(a or 0)
            stop = x_all.shape[0] if b is None else int(b)
            x_block = x_all[start:stop]
            y_block = y_all[start:stop]
            if len(y_block) == 0:
                continue
            e_block = pool_windows_stats_safe(x_block)
            y_int = encode_y(y_block)
            row_in_file = np.arange(start, stop, dtype=np.int64)

            yield {
                "missing": False,
                "manifest_entry_index": manifest_idx,
                "slice_index": slice_idx,
                "source_file": file_rel.replace("/", "\\"),
                "row_in_file_start": start,
                "row_in_file_stop": stop,
                "row_count_expected": int(len(y_block)),
                "row_in_file": row_in_file,
                "features": e_block,
                "y_int": y_int,
            }


def make_x_schema(feature_names: list[str]) -> pa.Schema:
    fields = [pa.field("row_id", pa.int64())]
    fields.extend(pa.field(name, pa.float32()) for name in feature_names)
    return pa.schema(fields)


def append_table(writer: pq.ParquetWriter | None, table: pa.Table, path: Path) -> pq.ParquetWriter:
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema, compression="zstd")
    writer.write_table(table)
    return writer


def predict_classes(
    booster: lgb.Booster,
    features: np.ndarray,
    thresholds: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    probs = predict_proba(booster, features, len(CLASSES))
    if thresholds is None:
        y_pred = probs.argmax(axis=1).astype(np.int64)
    else:
        y_pred = predict_with_thresholds_ovr(probs, thresholds).astype(np.int64)
    return probs, y_pred


def report_df_from_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    rep = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        zero_division=0,
        output_dict=True,
    )
    df = pd.DataFrame(rep).T.reset_index().rename(columns={"index": "label"})
    return df


def confusion_df_from_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    return pd.DataFrame(cm, index=CLASSES, columns=CLASSES)


def classification_report_df_from_metrics(test_meta: dict[str, Any], result_block: dict[str, Any]) -> pd.DataFrame:
    supports = [int(test_meta["class_dist"][name]) for name in CLASSES]
    precisions = [float(result_block["per_class_precision"][name]) for name in CLASSES]
    recalls = [float(result_block["per_class_recall"][name]) for name in CLASSES]

    rows: list[dict[str, Any]] = []
    f1s: list[float] = []
    total_support = int(sum(supports))
    weighted_precision_num = 0.0
    weighted_recall_num = 0.0
    weighted_f1_num = 0.0
    correct_total = 0.0

    for label, support, precision, recall in zip(CLASSES, supports, precisions, recalls):
        f1 = 0.0 if precision + recall == 0 else (2.0 * precision * recall) / (precision + recall)
        rows.append(
            {
                "label": label,
                "precision": precision,
                "recall": recall,
                "f1-score": f1,
                "support": support,
            }
        )
        f1s.append(f1)
        weighted_precision_num += precision * support
        weighted_recall_num += recall * support
        weighted_f1_num += f1 * support
        correct_total += recall * support

    rows.append(
        {
            "label": "accuracy",
            "precision": np.nan,
            "recall": np.nan,
            "f1-score": correct_total / total_support if total_support else np.nan,
            "support": total_support,
        }
    )
    rows.append(
        {
            "label": "macro avg",
            "precision": float(np.mean(precisions)) if precisions else np.nan,
            "recall": float(np.mean(recalls)) if recalls else np.nan,
            "f1-score": float(result_block["macro_f1"]),
            "support": total_support,
        }
    )
    rows.append(
        {
            "label": "weighted avg",
            "precision": weighted_precision_num / total_support if total_support else np.nan,
            "recall": weighted_recall_num / total_support if total_support else np.nan,
            "f1-score": float(result_block["weighted_f1"]),
            "support": total_support,
        }
    )
    return pd.DataFrame(rows)
