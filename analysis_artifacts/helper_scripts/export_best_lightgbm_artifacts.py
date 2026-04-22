from __future__ import annotations

import csv
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa

from _lightgbm_artifact_utils import (
    ANALYSIS_ROOT,
    CLASSES,
    ID_TO_NAME,
    LABEL_MAPPING,
    PROJECT_ROOT,
    append_table,
    classification_report_df_from_metrics,
    choose_best_run,
    compute_run_ranking,
    confusion_df_from_arrays,
    copy_file,
    iter_pooled_rows_with_status,
    load_json,
    make_feature_names,
    make_x_schema,
    predict_classes,
    relpath_str,
    report_df_from_arrays,
    save_json,
    selected_model_file,
    selected_thresholds_file,
    threshold_array_from_json,
    write_markdown,
    best_run_dir,
)


def write_csv_header(path: Path, header: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def append_csv_rows(path: Path, rows: list[list[object]]) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def main() -> None:
    run_selection_dir = ANALYSIS_ROOT / "run_selection"
    best_dir = ANALYSIS_ROOT / "lightgbm_best_run"
    helper_dir = ANALYSIS_ROOT / "helper_scripts"
    overlap_dir = ANALYSIS_ROOT / "overlap_analysis"
    shap_dir = ANALYSIS_ROOT / "shap_analysis"
    for p in [run_selection_dir, best_dir, helper_dir, overlap_dir, shap_dir]:
        p.mkdir(parents=True, exist_ok=True)

    ranking = compute_run_ranking()
    best = choose_best_run(ranking)
    ranking.to_csv(run_selection_dir / "lightgbm_run_ranking.csv", index=False)

    run_dir = best_run_dir(best["run_key"])
    metrics_path = run_dir / "metrics_full.json"
    metrics = load_json(metrics_path)

    model_src = selected_model_file(run_dir, best["experiment"])
    thresholds_src = selected_thresholds_file(run_dir, best["experiment"])

    selected_json = {
        "run_key": best["run_key"],
        "window_hours": int(best["window_hours"]),
        "horizon_hours": int(best["horizon_hours"]),
        "experiment": best["experiment"],
        "exact_model_file_path": relpath_str(model_src),
        "exact_metrics_file_path": relpath_str(metrics_path),
        "exact_thresholds_file_path": relpath_str(thresholds_src) if thresholds_src else None,
        "reason_selected": best["reason_selected"],
        "any_caveats": best["selection_caveats"],
    }
    save_json(run_selection_dir / "selected_best_lightgbm_run.json", selected_json)

    notes_md = (
        f"# Selected best LightGBM run\n\n"
        f"- `run_key`: `{best['run_key']}`\n"
        f"- `window_hours`: `{int(best['window_hours'])}`\n"
        f"- `horizon_hours`: `{int(best['horizon_hours'])}`\n"
        f"- `experiment`: `{best['experiment']}`\n"
        f"- `macro_f1_5c`: `{best['macro_f1_5c']:.6f}`\n\n"
        f"{best['reason_selected']}\n\n"
        f"Notebook Section 6 also explicitly states that `W72_H24` is the strongest overall "
        f"LightGBM baseline by 5-class test Macro-F1.\n"
    )
    if best["selection_caveats"]:
        notes_md += "\n## Caveats\n\n" + "\n".join(f"- {x}" for x in best["selection_caveats"]) + "\n"
    write_markdown(run_selection_dir / "selected_best_lightgbm_run_notes.md", notes_md)

    # Copy selected model artifacts
    model_dst = best_dir / f"model_artifact{model_src.suffix}"
    copy_file(model_src, model_dst)
    if thresholds_src is not None:
        copy_file(thresholds_src, best_dir / "model_thresholds.json")

    save_json(best_dir / "label_mapping.json", {"quiet": 0, "B": 1, "C": 2, "M": 3, "X": 4})

    config = metrics["config"]
    test_meta = metrics["test_meta"]
    feature_names = make_feature_names(int(test_meta["feature_dim"]))
    save_json(best_dir / "feature_names.json", {"feature_names": feature_names})

    model = lgb.Booster(model_file=str(model_src))
    thresholds = threshold_array_from_json(thresholds_src) if thresholds_src is not None else None

    model_metadata = {
        "run_key": best["run_key"],
        "window_hours": int(best["window_hours"]),
        "horizon_hours": int(best["horizon_hours"]),
        "experiment": best["experiment"],
        "train_manifest": config["train_manifest"],
        "val_manifest": config["val_manifest"],
        "test_manifest": config["test_manifest"],
        "train_meta": metrics["train_meta"],
        "val_meta": metrics["val_meta"],
        "test_meta": metrics["test_meta"],
        "feature_count": len(feature_names),
        "class_count": len(CLASSES),
        "class_labels": CLASSES,
        "label_mapping": LABEL_MAPPING,
        "preprocessing_notes": {
            "pooling": "Each (T,F) window is converted to a 4F embedding using mean/std/max/last pooling.",
            "scaling": "No separate scaler artifact is used in the LightGBM pipeline.",
            "feature_name_note": "Feature names follow the exact pooled feature order used by the model: mean_f*, std_f*, max_f*, last_f*.",
        },
        "evaluation_split_used_for_final_reported_test_results": "full test manifest",
        "original_run_dir": relpath_str(run_dir),
        "model_file": relpath_str(model_src),
        "metrics_file": relpath_str(metrics_path),
        "thresholds_file": relpath_str(thresholds_src) if thresholds_src else None,
    }
    save_json(best_dir / "model_metadata.json", model_metadata)

    x_test_path = best_dir / "X_test.parquet"
    y_test_path = best_dir / "y_test.csv"
    row_ids_path = best_dir / "test_row_ids.csv"
    missing_rows_path = best_dir / "missing_test_rows.csv"
    pred_path = best_dir / "test_predictions.csv"
    probs_path = best_dir / "test_class_probabilities.csv"

    write_csv_header(y_test_path, ["row_id", "y_true", "y_true_name"])
    write_csv_header(
        row_ids_path,
        ["row_id", "manifest_entry_index", "slice_index", "source_file", "row_in_file"],
    )
    write_csv_header(
        missing_rows_path,
        ["row_id", "manifest_entry_index", "slice_index", "source_file", "row_in_file", "status"],
    )
    write_csv_header(pred_path, ["row_id", "y_pred", "y_pred_name"])
    write_csv_header(
        probs_path,
        ["row_id", "prob_quiet", "prob_B", "prob_C", "prob_M", "prob_X"],
    )

    parquet_writer = None
    schema = make_x_schema(feature_names)
    row_id = 0
    y_true_chunks: list[np.ndarray] = []
    y_pred_chunks: list[np.ndarray] = []
    available_row_ids: list[np.ndarray] = []
    saved_y_pred = np.asarray(metrics["results_5c"][best["experiment"]]["test"]["y_pred"], dtype=np.int64)
    match_count = 0
    compared_prediction_rows = 0
    missing_rows = 0
    missing_chunks = 0

    manifest_path = Path(config["test_manifest"])
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path
    for chunk in iter_pooled_rows_with_status(manifest_path, allow_skip=True):
        if chunk["missing"]:
            n_missing = int(chunk["row_count_expected"])
            row_ids = np.arange(row_id, row_id + n_missing, dtype=np.int64)
            append_csv_rows(
                missing_rows_path,
                [
                    [
                        int(rid),
                        int(chunk["manifest_entry_index"]),
                        int(chunk["slice_index"]),
                        chunk["source_file"],
                        int(chunk["row_in_file_start"]) + offset,
                        "unreadable_npz_shard",
                    ]
                    for offset, rid in enumerate(row_ids.tolist())
                ],
            )
            row_id += n_missing
            missing_rows += n_missing
            missing_chunks += 1
            continue

        feat = np.asarray(chunk["features"], dtype=np.float32)
        y_true = np.asarray(chunk["y_int"], dtype=np.int64)
        probs, y_pred = predict_classes(model, feat, thresholds)

        n = feat.shape[0]
        row_ids = np.arange(row_id, row_id + n, dtype=np.int64)
        valid_saved_mask = row_ids < len(saved_y_pred)
        if valid_saved_mask.any():
            expected = saved_y_pred[row_ids[valid_saved_mask]]
            match_count += int((expected == y_pred[valid_saved_mask]).sum())
            compared_prediction_rows += int(valid_saved_mask.sum())
        row_id += n

        table_dict = {"row_id": row_ids}
        for idx, name in enumerate(feature_names):
            table_dict[name] = feat[:, idx]
        table = pa.Table.from_pydict(table_dict, schema=schema)
        parquet_writer = append_table(parquet_writer, table, x_test_path)

        append_csv_rows(
            row_ids_path,
            [
                [
                    int(rid),
                    int(chunk["manifest_entry_index"]),
                    int(chunk["slice_index"]),
                    chunk["source_file"],
                    int(rif),
                ]
                for rid, rif in zip(row_ids.tolist(), chunk["row_in_file"].tolist())
            ],
        )
        append_csv_rows(
            y_test_path,
            [[int(rid), int(y), ID_TO_NAME[int(y)]] for rid, y in zip(row_ids.tolist(), y_true.tolist())],
        )
        append_csv_rows(
            pred_path,
            [[int(rid), int(y), ID_TO_NAME[int(y)]] for rid, y in zip(row_ids.tolist(), y_pred.tolist())],
        )
        append_csv_rows(
            probs_path,
            [
                [int(rid), *[float(v) for v in row]]
                for rid, row in zip(row_ids.tolist(), probs.tolist())
            ],
        )

        y_true_chunks.append(y_true)
        y_pred_chunks.append(y_pred)
        available_row_ids.append(row_ids)

    if parquet_writer is not None:
        parquet_writer.close()

    y_true_all = np.concatenate(y_true_chunks, axis=0)
    y_pred_all = np.concatenate(y_pred_chunks, axis=0)
    row_ids_all = np.concatenate(available_row_ids, axis=0)

    confusion_df_from_arrays(y_true_all, y_pred_all).to_csv(best_dir / "confusion_matrix_test.csv")
    report_df_from_arrays(y_true_all, y_pred_all).to_csv(
        best_dir / "classification_report_test_reconstructed_subset.csv", index=False
    )
    classification_report_df_from_metrics(
        metrics["test_meta"],
        metrics["results_5c"][best["experiment"]]["test"],
    ).to_csv(best_dir / "classification_report_test.csv", index=False)

    export_meta = {
        "full_test_rows_expected": int(metrics["test_meta"]["num_samples"]),
        "manifest_stream_rows_seen": int(row_id),
        "exported_test_rows": int(len(y_true_all)),
        "missing_test_rows": int(missing_rows),
        "missing_test_chunks": int(missing_chunks),
        "saved_metrics_pred_match_fraction": float(match_count / max(compared_prediction_rows, 1)),
        "saved_metrics_pred_match_count": int(match_count),
        "saved_metrics_pred_total_compared": int(compared_prediction_rows),
        "saved_metrics_pred_uncompared_rows": int(max(row_id - compared_prediction_rows, 0)),
        "selected_experiment": best["experiment"],
        "row_id_is_partial_with_gaps": bool(missing_rows > 0),
    }
    save_json(best_dir / "export_metadata.json", export_meta)

    manifest = {
        "selected_best_run": selected_json,
        "package_status": {
            "run_selection_complete": True,
            "base_lightgbm_artifacts_complete": True,
            "overlap_artifacts_complete": False,
            "shap_artifacts_complete": False,
            "validation_complete": False,
        },
        "notes": [
            "The selected experiment is weighted_thresholded, which uses lgbm_weighted_5c.txt plus thresholds_weighted_5c.json.",
            "One test NPZ shard is currently unreadable, so row-level exports are reconstructed for all readable rows and preserve original global row_id positions with gaps for missing rows.",
            "classification_report_test.csv comes from the saved full-run metrics; confusion_matrix_test.csv reflects the reconstructed readable subset only.",
            "SHAP-compatible values are computed later with LightGBM pred_contrib on a balanced analysis subset, not on the full 813270-row test set, to keep artifact size practical.",
        ],
        "export_metadata": export_meta,
    }
    save_json(ANALYSIS_ROOT / "manifest.json", manifest)

    readme = f"""# Analysis Artifacts

This package contains extracted artifacts for the selected best LightGBM 5-class run.

## Selected run

- run_key: `{best['run_key']}`
- window / horizon: `{int(best['window_hours'])}` / `{int(best['horizon_hours'])}`
- selected experiment: `{best['experiment']}`
- model file: `{relpath_str(model_src)}`

## Row alignment

- `row_id` is a stable sequential identifier in the exact streamed test-manifest order.
- `X_test.parquet`, `y_test.csv`, `test_row_ids.csv`, `test_predictions.csv`, and `test_class_probabilities.csv` align on `row_id`.
- `test_row_ids.csv` also exposes `source_file` and `row_in_file`.
- If a shard is unreadable, the missing row IDs are listed in `lightgbm_best_run/missing_test_rows.csv`, and the readable-row exports keep the original global `row_id` values.

## What to use later

- PCA / t-SNE inputs: `overlap_analysis/balanced_analysis_subset.parquet`, `pca_ready_features.parquet`, `tsne_ready_features.parquet`
- SHAP inputs: `shap_analysis/shap_ready_X_test.parquet`
- Best LightGBM model: `lightgbm_best_run/model_artifact{model_src.suffix}`

## Important note

The selected run is the **weighted_thresholded** 5-class setup. The underlying booster is the weighted 5-class model, and the reported predictions also require `lightgbm_best_run/model_thresholds.json`.

## Current limitation

- `classification_report_test.csv` reflects the saved full run metrics.
- `confusion_matrix_test.csv` currently reflects the reconstructed readable subset only, because one source test shard is corrupted in the local project copy.
"""
    write_markdown(ANALYSIS_ROOT / "README.md", readme)


if __name__ == "__main__":
    main()
