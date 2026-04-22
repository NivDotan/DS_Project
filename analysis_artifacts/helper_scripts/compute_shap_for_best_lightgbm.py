from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
BEST_RUN_DIR = ANALYSIS_ROOT / "lightgbm_best_run"
OVERLAP_DIR = ANALYSIS_ROOT / "overlap_analysis"
SHAP_DIR = ANALYSIS_ROOT / "shap_analysis"
CLASS_NAMES = ["quiet", "B", "C", "M", "X"]
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def rank_importance(features: list[str], shap_values: np.ndarray) -> pd.DataFrame:
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    df = pd.DataFrame({"feature": features, "mean_abs_shap": mean_abs})
    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df[["feature", "mean_abs_shap", "rank"]]


def main() -> None:
    SHAP_DIR.mkdir(parents=True, exist_ok=True)

    selected = json.loads((ANALYSIS_ROOT / "run_selection" / "selected_best_lightgbm_run.json").read_text(encoding="utf-8"))
    feature_names = json.loads((BEST_RUN_DIR / "feature_names.json").read_text(encoding="utf-8"))["feature_names"]
    subset = pd.read_parquet(OVERLAP_DIR / "balanced_analysis_subset.parquet")

    shap_ready = subset[["row_id", "y_true", "y_true_name", "y_pred", "y_pred_name", *feature_names]].copy()
    shap_ready.to_parquet(SHAP_DIR / "shap_ready_X_test.parquet", index=False)

    X = shap_ready[feature_names].to_numpy(dtype=np.float32, copy=False)
    model_path = ANALYSIS_ROOT.parent / Path(selected["exact_model_file_path"])
    booster = lgb.Booster(model_file=str(model_path))
    contrib = booster.predict(X, pred_contrib=True)

    n_rows = X.shape[0]
    n_features = len(feature_names)
    contrib = np.asarray(contrib, dtype=np.float32).reshape(n_rows, len(CLASS_NAMES), n_features + 1)
    shap_values = contrib[:, :, :-1]
    expected_values = contrib[:, :, -1]
    expected_per_class = expected_values.mean(axis=0)
    expected_drift = np.max(np.abs(expected_values - expected_per_class[None, :]), axis=0)

    np.savez_compressed(
        SHAP_DIR / "shap_values_full.npz",
        row_id=shap_ready["row_id"].to_numpy(dtype=np.int64),
        feature_names=np.array(feature_names, dtype=object),
        class_names=np.array(CLASS_NAMES, dtype=object),
        shap_values=shap_values.astype(np.float32),
    )

    save_json(
        SHAP_DIR / "shap_expected_value.json",
        {
            "class_names": CLASS_NAMES,
            "expected_value_by_class": {name: float(expected_per_class[idx]) for idx, name in enumerate(CLASS_NAMES)},
            "max_rowwise_drift_by_class": {name: float(expected_drift[idx]) for idx, name in enumerate(CLASS_NAMES)},
            "source_note": "Expected values are derived from LightGBM pred_contrib output on the exported SHAP subset.",
        },
    )

    mask_B = shap_ready["y_true_name"].eq("B").to_numpy()
    mask_C = shap_ready["y_true_name"].eq("C").to_numpy()
    shap_B = shap_values[mask_B, CLASS_TO_INDEX["B"], :]
    shap_C = shap_values[mask_C, CLASS_TO_INDEX["C"], :]
    row_ids_B = shap_ready.loc[mask_B, "row_id"].to_numpy(dtype=np.int64)
    row_ids_C = shap_ready.loc[mask_C, "row_id"].to_numpy(dtype=np.int64)

    np.savez_compressed(
        SHAP_DIR / "shap_values_B.npz",
        row_id=row_ids_B,
        feature_names=np.array(feature_names, dtype=object),
        shap_values=shap_B.astype(np.float32),
    )
    np.savez_compressed(
        SHAP_DIR / "shap_values_C.npz",
        row_id=row_ids_C,
        feature_names=np.array(feature_names, dtype=object),
        shap_values=shap_C.astype(np.float32),
    )

    imp_B = rank_importance(feature_names, shap_B)
    imp_C = rank_importance(feature_names, shap_C)
    imp_B.to_csv(SHAP_DIR / "feature_importance_B.csv", index=False)
    imp_C.to_csv(SHAP_DIR / "feature_importance_C.csv", index=False)

    top10_B = imp_B.head(10).copy()
    top10_C = imp_C.head(10).copy()
    top10_B.to_csv(SHAP_DIR / "top10_features_B.csv", index=False)
    top10_C.to_csv(SHAP_DIR / "top10_features_C.csv", index=False)

    topB = top10_B.set_index("feature")
    topC = top10_C.set_index("feature")
    union_features = sorted(set(topB.index) | set(topC.index))
    intersection_size = len(set(topB.index) & set(topC.index))
    union_size = len(union_features)
    jaccard = float(intersection_size / union_size) if union_size else 0.0

    overlap_rows = []
    for feature in union_features:
        overlap_rows.append(
            {
                "feature": feature,
                "in_top10_B": feature in topB.index,
                "in_top10_C": feature in topC.index,
                "rank_B": int(topB.loc[feature, "rank"]) if feature in topB.index else np.nan,
                "rank_C": int(topC.loc[feature, "rank"]) if feature in topC.index else np.nan,
                "mean_abs_shap_B": float(topB.loc[feature, "mean_abs_shap"]) if feature in topB.index else np.nan,
                "mean_abs_shap_C": float(topC.loc[feature, "mean_abs_shap"]) if feature in topC.index else np.nan,
                "intersection_size": intersection_size,
                "jaccard_overlap": jaccard,
            }
        )
    pd.DataFrame(overlap_rows).to_csv(SHAP_DIR / "feature_overlap_B_vs_C.csv", index=False)


if __name__ == "__main__":
    main()
