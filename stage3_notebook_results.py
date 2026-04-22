from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd

from project_paths import PROJECT_ROOT


# One representative Stage3 run per family for W24/H24.
STAGE3_2424_REPRESENTATIVE_RUNS = {
    "TCN": PROJECT_ROOT / "runs_pytorch_stage2_fix" / "W24_H24_stage3b_stable",
    "BiLSTM": PROJECT_ROOT / "runs_pytorch_stage2_fix" / "m_fix_bilstm_v1" / "W24_H24_stage3b_stable",
    "Transformer": PROJECT_ROOT / "runs_pytorch_stage2_fix" / "stage3b_transformer_best_from_029_025" / "W24_H24_stage3b_stable_minirocket_v1",
    "MiniRocket": PROJECT_ROOT / "runs_pytorch_stage2_fix" / "stage3b_minirocket_phaseB_only_v1" / "W24_H24_stage3b_stable_minirocket_v1",
}


STAGE3_W72_RUNS = {
    "W72_H24": PROJECT_ROOT / "runs_pytorch_stage23_local" / "m_fix_bilstm_v1" / "W72_H24_stage3b_stable",
    "W72_H48": PROJECT_ROOT / "runs_pytorch_stage23_local" / "m_fix_bilstm_v1" / "W72_H48_stage3b_stable",
    "W72_H72": PROJECT_ROOT / "runs_pytorch_stage23_local" / "m_fix_bilstm_v1" / "W72_H72_stage3b_stable",
}


def _summary_path(run_dir: Path) -> Path:
    return run_dir / "stage3b_stable_summary.json"


def _load_summary(run_dir: Path) -> dict | None:
    p = _summary_path(run_dir)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _extract_best_row(label: str, run_dir: Path) -> dict:
    summary = _load_summary(run_dir)
    row = {
        "run": label,
        "run_dir": str(run_dir),
        "summary_json_exists": summary is not None,
    }
    if summary is None:
        return row

    best = summary.get("best", {})
    val = best.get("val", {})
    test = best.get("test", {})
    locked = summary.get("locked_gate", {})
    row.update(
        {
            "locked_t_flare": locked.get("t_flare"),
            "locked_t_reject_choices": tuple(locked.get("t_reject_choices", [])),
            "best_t_flare": best.get("t_flare"),
            "best_t_reject": best.get("t_reject"),
            "best_t_conf_bcmx": best.get("t_conf_bcmx"),
            "ckpt_path": best.get("ckpt_path"),
            "val_severe_recall": val.get("severe_recall"),
            "val_quiet_to_flare_fpr": val.get("quiet_to_flare_fpr"),
            "val_severe_support": val.get("severe_support"),
            "val_c_recall": val.get("c_recall"),
            "test_severe_recall": test.get("severe_recall"),
            "test_quiet_to_flare_fpr": test.get("quiet_to_flare_fpr"),
            "test_severe_support": test.get("severe_support"),
            "test_c_recall": test.get("c_recall"),
        }
    )
    return row


def get_stage3_results_2424() -> pd.DataFrame:
    rows = [_extract_best_row(label, run_dir) for label, run_dir in STAGE3_2424_REPRESENTATIVE_RUNS.items()]
    df = pd.DataFrame(rows)
    order = ["TCN", "BiLSTM", "Transformer", "MiniRocket"]
    df["run"] = pd.Categorical(df["run"], categories=order, ordered=True)
    return df.sort_values("run").reset_index(drop=True)


def get_stage3_results_w72() -> pd.DataFrame:
    rows = [_extract_best_row(label, run_dir) for label, run_dir in STAGE3_W72_RUNS.items()]
    df = pd.DataFrame(rows)
    order = ["W72_H24", "W72_H48", "W72_H72"]
    df["run"] = pd.Categorical(df["run"], categories=order, ordered=True)
    return df.sort_values("run").reset_index(drop=True)


def get_stage3_full_inventory_2424() -> pd.DataFrame:
    rows = []
    for p in PROJECT_ROOT.joinpath("runs_pytorch_stage2_fix").rglob("stage3b_stable_summary.json"):
        rows.append(_extract_best_row(str(p.parent.relative_to(PROJECT_ROOT / "runs_pytorch_stage2_fix")), p.parent))
    return pd.DataFrame(rows).sort_values(["test_severe_recall", "test_quiet_to_flare_fpr"], ascending=[False, True]).reset_index(drop=True)


def get_stage3_conf_sweeps(run_dir: Path, split: str = "val", best_only: bool = True) -> pd.DataFrame:
    summary = _load_summary(run_dir)
    if summary is None:
        return pd.DataFrame()

    rows = []
    best_t_reject = summary.get("best", {}).get("t_reject")
    for run in summary.get("runs", []):
        if best_only and run.get("t_reject") != best_t_reject:
            continue
        conf_rows = run.get(f"{split}_conf_sweep", [])
        for item in conf_rows:
            rows.append(
                {
                    "t_flare": run.get("t_flare"),
                    "t_reject": run.get("t_reject"),
                    "t_conf_bcmx": item.get("t_conf_bcmx"),
                    "severe_recall": item.get("severe_recall"),
                    "quiet_to_flare_fpr": item.get("quiet_to_flare_fpr"),
                    "severe_support": item.get("severe_support"),
                    "c_support": item.get("c_support"),
                    "c_recall": item.get("c_recall"),
                }
            )
    return pd.DataFrame(rows)


def get_stage3_sweeps_2424(split: str = "val", best_only: bool = True) -> Dict[str, pd.DataFrame]:
    return {label: get_stage3_conf_sweeps(run_dir, split=split, best_only=best_only) for label, run_dir in STAGE3_2424_REPRESENTATIVE_RUNS.items()}


def get_stage3_sweeps_w72(split: str = "val", best_only: bool = True) -> Dict[str, pd.DataFrame]:
    return {label: get_stage3_conf_sweeps(run_dir, split=split, best_only=best_only) for label, run_dir in STAGE3_W72_RUNS.items()}


def compact_stage3_view(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "run",
        "best_t_flare",
        "best_t_reject",
        "best_t_conf_bcmx",
        "val_severe_recall",
        "val_quiet_to_flare_fpr",
        "val_c_recall",
        "test_severe_recall",
        "test_quiet_to_flare_fpr",
        "test_c_recall",
        "ckpt_path",
    ]
    keep = [c for c in cols if c in df.columns]
    return df.loc[:, keep].copy()


def plot_stage3_best_tradeoff(df: pd.DataFrame, split: str = "test", title: str | None = None) -> None:
    x_col = f"{split}_quiet_to_flare_fpr"
    y_col = f"{split}_severe_recall"
    data = df.dropna(subset=[x_col, y_col]).copy()

    plt.figure(figsize=(8, 6))
    plt.scatter(data[x_col], data[y_col], s=80)
    for _, row in data.iterrows():
        plt.annotate(str(row["run"]), (row[x_col], row[y_col]), fontsize=9, xytext=(5, 5), textcoords="offset points")
    plt.xlabel("Quiet -> Flare FPR")
    plt.ylabel("Severe Recall")
    plt.title(title or f"Stage 3 Best Operating Points ({split.upper()})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_stage3_conf_tradeoff(sweeps: Dict[str, pd.DataFrame], title: str | None = None) -> None:
    plt.figure(figsize=(8, 6))
    for label, df in sweeps.items():
        if df is None or df.empty:
            continue
        df = df.sort_values("t_conf_bcmx")
        plt.plot(df["quiet_to_flare_fpr"], df["severe_recall"], marker="o", linewidth=1.8, label=label)
        for _, row in df.iterrows():
            plt.annotate(
                f"{row['t_conf_bcmx']:.2f}",
                (row["quiet_to_flare_fpr"], row["severe_recall"]),
                fontsize=8,
                xytext=(4, 4),
                textcoords="offset points",
            )
    plt.xlabel("Quiet -> Flare FPR")
    plt.ylabel("Severe Recall")
    plt.title(title or "Stage 3 Confidence Sweep Tradeoff")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()


def export_stage3_inventory(out_dir: Path | None = None) -> dict[str, Path]:
    out_dir = out_dir or (PROJECT_ROOT / "stage3_run_inventory")
    out_dir.mkdir(parents=True, exist_ok=True)

    p1 = out_dir / "stage3_2424_representative.csv"
    p2 = out_dir / "stage3_2424_representative.json"
    p3 = out_dir / "stage3_2424_full_inventory.csv"
    p4 = out_dir / "stage3_2424_full_inventory.json"
    p5 = out_dir / "stage3_w72_inventory.csv"
    p6 = out_dir / "stage3_w72_inventory.json"

    df_2424 = get_stage3_results_2424()
    df_2424_full = get_stage3_full_inventory_2424()
    df_w72 = get_stage3_results_w72()

    df_2424.to_csv(p1, index=False)
    p2.write_text(df_2424.to_json(orient="records", indent=2), encoding="utf-8")
    df_2424_full.to_csv(p3, index=False)
    p4.write_text(df_2424_full.to_json(orient="records", indent=2), encoding="utf-8")
    df_w72.to_csv(p5, index=False)
    p6.write_text(df_w72.to_json(orient="records", indent=2), encoding="utf-8")

    return {
        "stage3_2424_representative_csv": p1,
        "stage3_2424_representative_json": p2,
        "stage3_2424_full_inventory_csv": p3,
        "stage3_2424_full_inventory_json": p4,
        "stage3_w72_inventory_csv": p5,
        "stage3_w72_inventory_json": p6,
    }


if __name__ == "__main__":
    paths = export_stage3_inventory()
    for k, v in paths.items():
        print(f"{k}: {v}")
