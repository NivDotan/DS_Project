from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from project_paths import PROJECT_ROOT

STAGE1_ROOT = PROJECT_ROOT / "runs_pytorch_stage1_dataset_sweep_safe"

# These are the operating thresholds you used for the W72 family later on.
RUNS: dict[str, dict[str, float | str]] = {
    "W72_H72": {
        "chosen_plain_t": 0.20,
        "stage1_dir": str(STAGE1_ROOT / "W72_H72_two_stage_bal_other_stage1_safe"),
    },
    "W72_H48": {
        "chosen_plain_t": 0.20,
        "stage1_dir": str(STAGE1_ROOT / "W72_H48_two_stage_bal_other_stage1_safe"),
    },
    "W72_H24": {
        "chosen_plain_t": 0.075,
        "stage1_dir": str(STAGE1_ROOT / "W72_H24_two_stage_bal_other_stage1_safe"),
    },
}


def first_close_row(df: pd.DataFrame, col: str, value: float, atol: float = 1e-12) -> pd.Series:
    mask = np.isclose(df[col].astype(float), float(value), atol=atol)
    if not mask.any():
        raise ValueError(f"No row found in column '{col}' close to value {value}")
    return df.loc[mask].iloc[0]


def _load_run(run_key: str):
    cfg = RUNS[run_key]
    stage1_dir = Path(str(cfg["stage1_dir"]))
    chosen_plain_t = float(cfg["chosen_plain_t"])

    val_thr = pd.read_csv(stage1_dir / "stage1_threshold_sweep_val_natural.csv")
    test_thr = pd.read_csv(stage1_dir / "stage1_threshold_sweep_test_natural.csv")

    row_thr_val = first_close_row(val_thr, "t", chosen_plain_t)
    row_thr_test = first_close_row(test_thr, "t", chosen_plain_t)
    return chosen_plain_t, val_thr, test_thr, row_thr_val, row_thr_test


def stage1_summary_table_for_run(run_key: str) -> pd.DataFrame:
    chosen_plain_t, _, _, row_thr_val, row_thr_test = _load_run(run_key)

    stage1_summary_table = pd.DataFrame([
        {
            "Model": run_key,
            "Split": "Validation",
            "Gate Type": "Plain threshold",
            "Threshold": f"t = {chosen_plain_t:.3f}",
            "Quiet filtered": row_thr_val["quiet_filtered"],
            "B kept": row_thr_val["B_pass"],
            "C kept": row_thr_val["C_pass"],
            "M kept": row_thr_val["M_pass"],
            "X kept": row_thr_val["X_pass"],
        },
        {
            "Model": run_key,
            "Split": "Test",
            "Gate Type": "Plain threshold",
            "Threshold": f"t = {chosen_plain_t:.3f}",
            "Quiet filtered": row_thr_test["quiet_filtered"],
            "B kept": row_thr_test["B_pass"],
            "C kept": row_thr_test["C_pass"],
            "M kept": row_thr_test["M_pass"],
            "X kept": row_thr_test["X_pass"],
        },
    ])

    stage1_summary_table_display = stage1_summary_table.copy()
    for col in ["Quiet filtered", "B kept", "C kept", "M kept", "X kept"]:
        stage1_summary_table_display[col] = stage1_summary_table_display[col].map(lambda x: f"{float(x):.4f}")
    return stage1_summary_table_display


def plot_stage1_plain_threshold_test(run_key: str) -> None:
    chosen_plain_t, _, test_thr, _, _ = _load_run(run_key)
    plot_df = test_thr[["t", "quiet_filtered", "M_pass", "X_pass", "C_pass", "B_pass"]].copy()
    plot_df = plot_df.sort_values("t")

    plt.figure(figsize=(8, 5))
    plt.plot(plot_df["t"], plot_df["quiet_filtered"], marker="o", label="Quiet filtered")
    plt.plot(plot_df["t"], plot_df["M_pass"], marker="o", label="M kept")
    plt.plot(plot_df["t"], plot_df["X_pass"], marker="o", label="X kept")
    plt.plot(plot_df["t"], plot_df["C_pass"], marker="o", label="C kept")
    plt.plot(plot_df["t"], plot_df["B_pass"], marker="o", label="B kept")
    plt.axvline(chosen_plain_t, linestyle="--", label=f"Chosen plain threshold ({chosen_plain_t:.3f})")
    plt.xlabel("Stage1 threshold")
    plt.ylabel("Rate")
    plt.title(f"{run_key} Stage1 plain-threshold trade-off on test data")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_stage1_plain_threshold_val(run_key: str) -> None:
    chosen_plain_t, val_thr, _, _, _ = _load_run(run_key)
    plot_df = val_thr[["t", "quiet_filtered", "M_pass", "X_pass", "C_pass", "B_pass"]].copy()
    plot_df = plot_df.sort_values("t")

    plt.figure(figsize=(8, 5))
    plt.plot(plot_df["t"], plot_df["quiet_filtered"], marker="o", label="Quiet filtered")
    plt.plot(plot_df["t"], plot_df["M_pass"], marker="o", label="M kept")
    plt.plot(plot_df["t"], plot_df["X_pass"], marker="o", label="X kept")
    plt.plot(plot_df["t"], plot_df["C_pass"], marker="o", label="C kept")
    plt.plot(plot_df["t"], plot_df["B_pass"], marker="o", label="B kept")
    plt.axvline(chosen_plain_t, linestyle="--", label=f"Chosen plain threshold ({chosen_plain_t:.3f})")
    plt.xlabel("Stage1 threshold")
    plt.ylabel("Rate")
    plt.title(f"{run_key} Stage1 plain-threshold trade-off on validation data")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def print_stage1_chosen_gate(run_key: str) -> None:
    chosen_plain_t, _, _, _, row_thr_test = _load_run(run_key)
    print("Chosen plain threshold gate:")
    print(f"  t = {chosen_plain_t:.3f}")
    print(f"  quiet filtered  = {row_thr_test['quiet_filtered']:.4f}")
    print(f"  C kept          = {row_thr_test['C_pass']:.4f}")
    print(f"  B kept          = {row_thr_test['B_pass']:.4f}")
    print(f"  M kept          = {row_thr_test['M_pass']:.4f}")
    print(f"  X kept          = {row_thr_test['X_pass']:.4f}")


def render_stage1_section_for_run(run_key: str) -> pd.DataFrame:
    #display_df = stage1_summary_table_for_run(run_key)
    print(f"\n=== {run_key} ===")
    #print(display_df.to_string(index=False))
    plot_stage1_plain_threshold_test(run_key)
    plot_stage1_plain_threshold_val(run_key)
    print_stage1_chosen_gate(run_key)
    #return display_df


def render_stage1_section_for_all_w72() -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for run_key in RUNS:
        out[run_key] = render_stage1_section_for_run(run_key)
    return out
