from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd

from project_paths import PROJECT_ROOT


STAGE2_2424_RUNS = {
    "TCN": PROJECT_ROOT / "runs_pytorch_stage2_fix" / "W24_H24_stage3a_reject_only",
    "BiLSTM": PROJECT_ROOT
    / "runs_pytorch_stage2_fix"
    / "W24_H24_stage3a_reject_only_bilstm"
    / "W24_H24_stage3a_reject_only",
    "Transformer": PROJECT_ROOT
    / "runs_pytorch_stage2_fix"
    / "stage3a_transformer_v2"
    / "W24_H24_stage3a_reject_only",
    "MiniRocket": PROJECT_ROOT
    / "runs_pytorch_stage2_fix"
    / "stage3a_minirocket_v1"
    / "W24_H24_stage3a_reject_only",
}


STAGE2_W72_RUNS = {
    "W72_H24": PROJECT_ROOT / "runs_pytorch_stage23_local" / "W72_H24_stage3a_reject_only",
    "W72_H48": PROJECT_ROOT / "runs_pytorch_stage23_local" / "W72_H48_stage3a_reject_only",
    "W72_H72": PROJECT_ROOT / "runs_pytorch_stage23_local" / "W72_H72_stage3a_reject_only",
}


METRIC_COLUMNS = [
    "t_flare",
    "t_reject",
    "flare_recall",
    "quiet_to_flare_fpr",
    "pred_quiet_rate",
    "flare_recall_given_gate",
    "quiet_fpr_given_gate",
    "gate_rate",
]


def _safe_read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _best_row_from_sweep(df: pd.DataFrame) -> pd.Series:
    # Prefer higher flare recall, then lower quiet->flare FPR, then higher quiet filtered.
    ranked = df.sort_values(
        ["flare_recall", "quiet_to_flare_fpr", "pred_quiet_rate"],
        ascending=[False, True, False],
    )
    return ranked.iloc[0]


def _match_test_row(test_df: pd.DataFrame | None, t_flare: float, t_reject: float) -> pd.Series | None:
    if test_df is None:
        return None
    mask = (
        test_df["t_flare"].round(6).eq(round(float(t_flare), 6))
        & test_df["t_reject"].round(6).eq(round(float(t_reject), 6))
    )
    matched = test_df.loc[mask]
    if matched.empty:
        return None
    return matched.iloc[0]


def _summarize_stage2_run(name: str, run_dir: Path) -> dict:
    metrics_path = run_dir / "stage3a_reject_metrics.json"
    val_path = run_dir / "stage3a_reject_sweep_val.csv"
    test_path = run_dir / "stage3a_reject_sweep_test.csv"

    metrics_obj = _safe_read_json(metrics_path)
    val_df = _safe_read_csv(val_path)
    test_df = _safe_read_csv(test_path)

    row = {
        "run": name,
        "run_dir": str(run_dir),
        "metrics_json_exists": metrics_path.exists(),
        "val_sweep_exists": val_path.exists(),
        "test_sweep_exists": test_path.exists(),
        "selection_source": None,
    }

    if metrics_obj is not None:
        best = metrics_obj["best_val"]
        row["selection_source"] = "metrics_json.best_val"
        for col in METRIC_COLUMNS:
            row[f"val_{col}"] = best.get(col)
        test_row = _match_test_row(test_df, best["t_flare"], best["t_reject"])
    elif val_df is not None and not val_df.empty:
        best = _best_row_from_sweep(val_df)
        row["selection_source"] = "derived_from_val_sweep"
        for col in METRIC_COLUMNS:
            row[f"val_{col}"] = best.get(col)
        test_row = _match_test_row(test_df, best["t_flare"], best["t_reject"])
    else:
        row["selection_source"] = "no_stage2_summary_files"
        test_row = None

    if test_row is not None:
        for col in METRIC_COLUMNS:
            row[f"test_{col}"] = test_row.get(col)
    else:
        for col in METRIC_COLUMNS:
            row[f"test_{col}"] = None

    return row


def get_stage2_results_2424() -> pd.DataFrame:
    rows = [_summarize_stage2_run(name, run_dir) for name, run_dir in STAGE2_2424_RUNS.items()]
    df = pd.DataFrame(rows)
    order = ["TCN", "BiLSTM", "Transformer", "MiniRocket"]
    df["run"] = pd.Categorical(df["run"], categories=order, ordered=True)
    return df.sort_values("run").reset_index(drop=True)


def get_stage2_results_w72() -> pd.DataFrame:
    rows = [_summarize_stage2_run(name, run_dir) for name, run_dir in STAGE2_W72_RUNS.items()]
    df = pd.DataFrame(rows)
    order = ["W72_H24", "W72_H48", "W72_H72"]
    df["run"] = pd.Categorical(df["run"], categories=order, ordered=True)
    return df.sort_values("run").reset_index(drop=True)


def get_stage2_sweeps_2424() -> Dict[str, dict]:
    out = {}
    for name, run_dir in STAGE2_2424_RUNS.items():
        out[name] = {
            "val": _safe_read_csv(run_dir / "stage3a_reject_sweep_val.csv"),
            "test": _safe_read_csv(run_dir / "stage3a_reject_sweep_test.csv"),
        }
    return out


def get_stage2_sweeps_w72() -> Dict[str, dict]:
    out = {}
    for name, run_dir in STAGE2_W72_RUNS.items():
        out[name] = {
            "val": _safe_read_csv(run_dir / "stage3a_reject_sweep_val.csv"),
            "test": _safe_read_csv(run_dir / "stage3a_reject_sweep_test.csv"),
        }
    return out


def plot_stage2_tradeoff(
    sweeps: Dict[str, dict],
    split: str = "val",
    title: str | None = None,
    annotate_t_reject: bool = True,
) -> None:
    plt.figure(figsize=(8, 6))
    for name, parts in sweeps.items():
        df = parts.get(split)
        if df is None or df.empty:
            continue
        plt.plot(
            df["quiet_to_flare_fpr"],
            df["flare_recall"],
            marker="o",
            linewidth=1.8,
            label=name,
        )
        if annotate_t_reject:
            for _, row in df.iterrows():
                plt.annotate(
                    f"{row['t_reject']:.2f}",
                    (row["quiet_to_flare_fpr"], row["flare_recall"]),
                    fontsize=8,
                    xytext=(4, 4),
                    textcoords="offset points",
                )

    plt.xlabel("Quiet -> Flare FPR")
    plt.ylabel("Flare Recall")
    plt.title(title or f"Stage 2 Tradeoff ({split.upper()})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()


def compact_stage2_view(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "run",
        "selection_source",
        "val_t_flare",
        "val_t_reject",
        "val_flare_recall",
        "val_quiet_to_flare_fpr",
        "val_pred_quiet_rate",
        "val_gate_rate",
        "test_flare_recall",
        "test_quiet_to_flare_fpr",
        "test_pred_quiet_rate",
        "test_gate_rate",
    ]
    keep = [c for c in cols if c in df.columns]
    return df.loc[:, keep].copy()
