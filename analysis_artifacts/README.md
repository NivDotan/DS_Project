# Analysis Artifacts

This package contains extracted artifacts for the selected best LightGBM 5-class run.

## Selected run

- run_key: `W72_H24`
- window / horizon: `72` / `24`
- selected experiment: `weighted_thresholded`
- model file: `runs_lightgbm_night\20260411_124138\W72_H24\lgbm_weighted_5c.txt`

## Row alignment

- `row_id` is a stable sequential identifier in the exact streamed test-manifest order.
- `X_test.parquet`, `y_test.csv`, `test_row_ids.csv`, `test_predictions.csv`, and `test_class_probabilities.csv` align on `row_id`.
- `test_row_ids.csv` also exposes `source_file` and `row_in_file`.
- If a shard is unreadable, the missing row IDs are listed in `lightgbm_best_run/missing_test_rows.csv`, and the readable-row exports keep the original global `row_id` values.

## What to use later

- PCA / t-SNE inputs: `overlap_analysis/balanced_analysis_subset.parquet`, `pca_ready_features.parquet`, `tsne_ready_features.parquet`
- SHAP inputs: `shap_analysis/shap_ready_X_test.parquet`
- Best LightGBM model: `lightgbm_best_run/model_artifact.txt`

## Important note

The selected run is the **weighted_thresholded** 5-class setup. The underlying booster is the weighted 5-class model, and the reported predictions also require `lightgbm_best_run/model_thresholds.json`.

## Current limitation

- `classification_report_test.csv` reflects the saved full run metrics.
- `confusion_matrix_test.csv` currently reflects the reconstructed readable subset only, because one source test shard is corrupted in the local project copy.
- The local manifest stream exposes `827746` row positions, while the saved LightGBM test metrics expect `813270` rows. The exported row-level tables are internally aligned to each other, but they should be treated as a reconstruction layer for interpretability work rather than a perfect replay of the original reported test-order evaluation.

## Helper scripts

- `helper_scripts/export_best_lightgbm_artifacts.py` exports the selected run and row-level test artifacts.
- `helper_scripts/build_overlap_subset.py` builds the balanced PCA / t-SNE subset.
- `helper_scripts/compute_shap_for_best_lightgbm.py` computes LightGBM `pred_contrib` SHAP-style artifacts on the balanced subset.
- `helper_scripts/validate_artifacts.py` checks row alignment, feature counts, label mapping, and SHAP/subset consistency, then updates `manifest.json`.
