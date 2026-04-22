
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _first_close_row(df: pd.DataFrame, col: str, value: float, atol: float = 1e-12) -> pd.Series:
    mask = np.isclose(df[col].astype(float), float(value), atol=atol)
    if not mask.any():
        raise ValueError(f"No row found in column '{col}' close to value {value}")
    return df.loc[mask].iloc[0]

def render_stage1_plain_run(val_csv, test_csv, run_label: str, chosen_t: float):
    val_thr = pd.read_csv(val_csv)
    test_thr = pd.read_csv(test_csv)

    row_val = _first_close_row(val_thr, "t", chosen_t)
    row_test = _first_close_row(test_thr, "t", chosen_t)

    # Validation plot
    plot_df = val_thr[["t", "quiet_filtered", "B_pass", "C_pass", "M_pass", "X_pass"]].copy().sort_values("t")
    plt.figure(figsize=(8, 5))
    plt.plot(plot_df["t"], plot_df["quiet_filtered"], marker="o", label="Quiet filtered")
    plt.plot(plot_df["t"], plot_df["B_pass"], marker="o", label="B kept")
    plt.plot(plot_df["t"], plot_df["C_pass"], marker="o", label="C kept")
    plt.plot(plot_df["t"], plot_df["M_pass"], marker="o", label="M kept")
    plt.plot(plot_df["t"], plot_df["X_pass"], marker="o", label="X kept")
    plt.axvline(chosen_t, linestyle="--", label=f"Chosen threshold ({chosen_t:.3f})")
    plt.xlabel("Stage1 threshold")
    plt.ylabel("Rate")
    plt.title(f"{run_label} — Stage1 plain-threshold trade-off (validation)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Test plot
    plot_df = test_thr[["t", "quiet_filtered", "B_pass", "C_pass", "M_pass", "X_pass"]].copy().sort_values("t")
    plt.figure(figsize=(8, 5))
    plt.plot(plot_df["t"], plot_df["quiet_filtered"], marker="o", label="Quiet filtered")
    plt.plot(plot_df["t"], plot_df["B_pass"], marker="o", label="B kept")
    plt.plot(plot_df["t"], plot_df["C_pass"], marker="o", label="C kept")
    plt.plot(plot_df["t"], plot_df["M_pass"], marker="o", label="M kept")
    plt.plot(plot_df["t"], plot_df["X_pass"], marker="o", label="X kept")
    plt.axvline(chosen_t, linestyle="--", label=f"Chosen threshold ({chosen_t:.3f})")
    plt.xlabel("Stage1 threshold")
    plt.ylabel("Rate")
    plt.title(f"{run_label} — Stage1 plain-threshold trade-off (test)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("Chosen plain threshold gate:")
    print(f"  t = {chosen_t:.3f}")
    print(f"  quiet filtered  = {row_test['quiet_filtered']:.4f}")
    print(f"  C kept          = {row_test['C_pass']:.4f}")
    print(f"  B kept          = {row_test['B_pass']:.4f}")
    print(f"  M kept          = {row_test['M_pass']:.4f}")
    print(f"  X kept          = {row_test['X_pass']:.4f}")

    return {"val_row": row_val, "test_row": row_test}
