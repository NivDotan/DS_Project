from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
BEST_RUN_DIR = ANALYSIS_ROOT / "lightgbm_best_run"
OVERLAP_DIR = ANALYSIS_ROOT / "overlap_analysis"
SEED = 42


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    OVERLAP_DIR.mkdir(parents=True, exist_ok=True)

    y_df = pd.read_csv(BEST_RUN_DIR / "y_test.csv")
    pred_df = pd.read_csv(BEST_RUN_DIR / "test_predictions.csv")
    feature_names = json.loads((BEST_RUN_DIR / "feature_names.json").read_text(encoding="utf-8"))["feature_names"]

    meta = y_df.merge(pred_df, on="row_id", how="inner", validate="one_to_one")
    counts_before = meta["y_true_name"].value_counts().sort_index().to_dict()

    quiet_df = meta[meta["y_true_name"] == "quiet"].copy()
    nonquiet_df = meta[meta["y_true_name"] != "quiet"].copy()
    quiet_cap = min(len(quiet_df), len(nonquiet_df))
    quiet_keep = quiet_df.sample(n=quiet_cap, random_state=SEED) if quiet_cap < len(quiet_df) else quiet_df
    selected = (
        pd.concat([nonquiet_df, quiet_keep], axis=0)
        .sort_values("row_id")
        .reset_index(drop=True)
    )
    selected_ids = set(int(x) for x in selected["row_id"].tolist())
    counts_after = selected["y_true_name"].value_counts().sort_index().to_dict()

    out_path = OVERLAP_DIR / "balanced_analysis_subset.parquet"
    writer: pq.ParquetWriter | None = None
    parquet_file = pq.ParquetFile(BEST_RUN_DIR / "X_test.parquet")

    for batch in parquet_file.iter_batches(batch_size=25000):
        df = batch.to_pandas()
        df = df[df["row_id"].isin(selected_ids)]
        if df.empty:
            continue
        df = df.merge(selected, on="row_id", how="inner", validate="one_to_one")
        ordered = ["row_id", "y_true", "y_true_name", *feature_names, "y_pred", "y_pred_name"]
        table = pa.Table.from_pandas(df[ordered], preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")
        writer.write_table(table)

    if writer is not None:
        writer.close()

    subset_df = pd.read_parquet(out_path)
    ready_cols = ["row_id", "y_true", "y_true_name", *feature_names]
    subset_df[ready_cols].to_parquet(OVERLAP_DIR / "pca_ready_features.parquet", index=False)
    subset_df[ready_cols].to_parquet(OVERLAP_DIR / "tsne_ready_features.parquet", index=False)

    save_json(
        OVERLAP_DIR / "balanced_analysis_subset_metadata.json",
        {
            "sampling_logic": {
                "quiet": f"random sample with seed={SEED}, capped at total non-quiet count",
                "B": "keep all available rows",
                "C": "keep all available rows",
                "M": "keep all available rows",
                "X": "keep all available rows",
            },
            "counts_before_sampling": counts_before,
            "counts_after_sampling": counts_after,
            "random_seed": SEED,
            "source_note": "Built from the exported row-aligned LightGBM test artifacts in analysis_artifacts/lightgbm_best_run.",
        },
    )


if __name__ == "__main__":
    main()
