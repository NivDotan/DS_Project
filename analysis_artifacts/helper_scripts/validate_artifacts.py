from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
BEST_RUN_DIR = ANALYSIS_ROOT / "lightgbm_best_run"
OVERLAP_DIR = ANALYSIS_ROOT / "overlap_analysis"
SHAP_DIR = ANALYSIS_ROOT / "shap_analysis"
EXPECTED_LABEL_MAPPING = {"quiet": 0, "B": 1, "C": 2, "M": 3, "X": 4}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    manifest_path = ANALYSIS_ROOT / "manifest.json"
    manifest = load_json(manifest_path)

    feature_names = load_json(BEST_RUN_DIR / "feature_names.json")["feature_names"]
    label_mapping = load_json(BEST_RUN_DIR / "label_mapping.json")
    export_meta = load_json(BEST_RUN_DIR / "export_metadata.json")

    x_rows = pq.ParquetFile(BEST_RUN_DIR / "X_test.parquet").metadata.num_rows
    y_rows = len(pd.read_csv(BEST_RUN_DIR / "y_test.csv"))
    pred_rows = len(pd.read_csv(BEST_RUN_DIR / "test_predictions.csv"))
    prob_rows = len(pd.read_csv(BEST_RUN_DIR / "test_class_probabilities.csv"))
    missing_rows = len(pd.read_csv(BEST_RUN_DIR / "missing_test_rows.csv"))

    subset_rows = pq.ParquetFile(OVERLAP_DIR / "balanced_analysis_subset.parquet").metadata.num_rows
    pca_rows = pq.ParquetFile(OVERLAP_DIR / "pca_ready_features.parquet").metadata.num_rows
    tsne_rows = pq.ParquetFile(OVERLAP_DIR / "tsne_ready_features.parquet").metadata.num_rows
    shap_ready_rows = pq.ParquetFile(SHAP_DIR / "shap_ready_X_test.parquet").metadata.num_rows

    shap_full = np.load(SHAP_DIR / "shap_values_full.npz", allow_pickle=True)
    shap_B = np.load(SHAP_DIR / "shap_values_B.npz", allow_pickle=True)
    shap_C = np.load(SHAP_DIR / "shap_values_C.npz", allow_pickle=True)

    subset_ids = set(pd.read_parquet(OVERLAP_DIR / "balanced_analysis_subset.parquet", columns=["row_id"])["row_id"].tolist())
    y_ids = set(pd.read_csv(BEST_RUN_DIR / "y_test.csv", usecols=["row_id"])["row_id"].tolist())

    results = {
        "x_test_rows": int(x_rows),
        "y_test_rows": int(y_rows),
        "prediction_rows": int(pred_rows),
        "probability_rows": int(prob_rows),
        "missing_row_manifest_rows": int(missing_rows),
        "feature_count_from_schema": int(pq.ParquetFile(BEST_RUN_DIR / "X_test.parquet").schema_arrow.names.__len__() - 1),
        "feature_count_from_feature_names_json": int(len(feature_names)),
        "label_mapping_matches_expected": label_mapping == EXPECTED_LABEL_MAPPING,
        "row_count_match_base_exports": bool(x_rows == y_rows == pred_rows == prob_rows),
        "subset_rows": int(subset_rows),
        "pca_rows": int(pca_rows),
        "tsne_rows": int(tsne_rows),
        "subset_rows_match_ready_files": bool(subset_rows == pca_rows == tsne_rows),
        "subset_row_ids_are_subset_of_y_test": subset_ids.issubset(y_ids),
        "shap_ready_rows": int(shap_ready_rows),
        "shap_full_rows": int(shap_full["row_id"].shape[0]),
        "shap_B_rows": int(shap_B["row_id"].shape[0]),
        "shap_C_rows": int(shap_C["row_id"].shape[0]),
        "shap_ready_matches_full": bool(shap_ready_rows == shap_full["row_id"].shape[0]),
        "export_metadata": export_meta,
    }

    missing_files = []
    for rel in [
        "run_selection/lightgbm_run_ranking.csv",
        "run_selection/selected_best_lightgbm_run.json",
        "lightgbm_best_run/model_artifact.txt",
        "lightgbm_best_run/model_metadata.json",
        "lightgbm_best_run/X_test.parquet",
        "lightgbm_best_run/y_test.csv",
        "lightgbm_best_run/test_predictions.csv",
        "lightgbm_best_run/test_class_probabilities.csv",
        "overlap_analysis/balanced_analysis_subset.parquet",
        "shap_analysis/shap_ready_X_test.parquet",
        "shap_analysis/shap_values_full.npz",
    ]:
        path = ANALYSIS_ROOT / rel
        if not path.exists():
            missing_files.append(rel)
    results["missing_expected_files"] = missing_files

    manifest["package_status"]["overlap_artifacts_complete"] = True
    manifest["package_status"]["shap_artifacts_complete"] = True
    manifest["package_status"]["validation_complete"] = True
    manifest["validation"] = results
    save_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
