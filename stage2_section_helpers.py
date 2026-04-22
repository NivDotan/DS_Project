
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def config_table_2424(cfg_bilstm, cfg_tx, cfg_mr) -> pd.DataFrame:
    rows = []
    for name, cfg in [
        ("W24_H24_BiLSTM", cfg_bilstm),
        ("W24_H24_Transformer", cfg_tx),
        ("W24_H24_MiniRocket", cfg_mr),
    ]:
        row = {
            "run": name,
            "root4": getattr(cfg, "root4", None),
            "W": getattr(cfg, "W", None),
            "H": getattr(cfg, "H", None),
            "out_root": getattr(cfg, "out_root", None),
            "s1_ckpt_path": getattr(cfg, "s1_ckpt_path", None),
            "hidden": getattr(cfg, "hidden", None),
            "epochs": getattr(cfg, "s3a_epochs", None),
            "t_flare_train": getattr(cfg, "t_flare_train", None),
            "max_train_files": getattr(cfg, "max_train_files", None),
            "max_val_files": getattr(cfg, "max_val_files", None),
            "per_file": getattr(cfg, "per_file", None),
            "neg_ratio": getattr(cfg, "s3a_neg_ratio", None),
        }
        rows.append(row)
    return pd.DataFrame(rows)

def config_table_w72(configs: dict) -> pd.DataFrame:
    rows = []
    for run_key, c in configs.items():
        rows.append({
            "run_key": run_key,
            "window_h": c.get("window_h"),
            "horizon_h": c.get("horizon_h"),
            "dataset_root": str(c.get("dataset_root")),
            "stage1_run_dir": str(c.get("stage1_run_dir")),
            "stage2_run_dir": str(c.get("stage2_run_dir")),
            "chosen_t_flare": c.get("chosen_t_flare"),
            "status": c.get("status"),
        })
    return pd.DataFrame(rows)

def _plot_tradeoff_df(df: pd.DataFrame, title: str):
    if df is None or len(df) == 0:
        print(f"No data for {title}")
        return
    plt.figure(figsize=(8, 5))
    if "t_reject" in df.columns:
        x = df["t_reject"]
        xlabel = "t_reject"
    elif "t" in df.columns:
        x = df["t"]
        xlabel = "threshold"
    else:
        x = np.arange(len(df))
        xlabel = "index"
    if "flare_recall" in df.columns:
        plt.plot(x, df["flare_recall"], marker="o", label="Flare recall")
    if "quiet_to_flare_fpr" in df.columns:
        plt.plot(x, df["quiet_to_flare_fpr"], marker="o", label="Quiet->flare FPR")
    if "pred_quiet_rate" in df.columns:
        plt.plot(x, df["pred_quiet_rate"], marker="o", label="Pred quiet rate")
    plt.xlabel(xlabel)
    plt.ylabel("Rate")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_val_test_pair(parts: dict, run_label: str):
    val = parts.get("val")
    test = parts.get("test")
    _plot_tradeoff_df(val, f"{run_label} / VAL")
    _plot_tradeoff_df(test, f"{run_label} / TEST")

def build_w72_chosen_rows(w72_configs: dict, compact_view: pd.DataFrame, sweeps_w72: dict) -> pd.DataFrame:
    out = []
    compact_idx = {}
    if compact_view is not None and len(compact_view) > 0 and "run" in compact_view.columns:
        compact_idx = {str(r["run"]): r for _, r in compact_view.iterrows()}
    for run_key, cfg in w72_configs.items():
        chosen_t_flare = cfg.get("chosen_t_flare")
        compact_row = compact_idx.get(run_key)
        chosen_t_reject = None
        if compact_row is not None and pd.notna(compact_row.get("val_t_reject")):
            chosen_t_reject = float(compact_row.get("val_t_reject"))
        parts = sweeps_w72.get(run_key, {})
        for split in ["val", "test"]:
            df = parts.get(split)
            if df is None or len(df) == 0:
                out.append({
                    "run_key": run_key,
                    "split": split.upper(),
                    "t_flare": chosen_t_flare,
                    "t_reject": chosen_t_reject,
                    "flare_recall": np.nan,
                    "quiet_to_flare_fpr": np.nan,
                    "pred_quiet_rate": np.nan,
                    "flare_recall_given_gate": np.nan,
                    "quiet_fpr_given_gate": np.nan,
                    "gate_rate": np.nan,
                })
                continue
            sub = df.copy()
            if chosen_t_flare is not None and "t_flare" in sub.columns:
                sub = sub[np.isclose(sub["t_flare"].astype(float), float(chosen_t_flare))]
            chosen = None
            if chosen_t_reject is not None and "t_reject" in sub.columns:
                tmp = sub[np.isclose(sub["t_reject"].astype(float), float(chosen_t_reject))]
                if len(tmp):
                    chosen = tmp.iloc[0]
            if chosen is None:
                chosen = sub.iloc[0] if len(sub) else df.iloc[0]
            out.append({
                "run_key": run_key,
                "split": split.upper(),
                "t_flare": chosen.get("t_flare"),
                "t_reject": chosen.get("t_reject"),
                "flare_recall": chosen.get("flare_recall"),
                "quiet_to_flare_fpr": chosen.get("quiet_to_flare_fpr"),
                "pred_quiet_rate": chosen.get("pred_quiet_rate"),
                "flare_recall_given_gate": chosen.get("flare_recall_given_gate"),
                "quiet_fpr_given_gate": chosen.get("quiet_fpr_given_gate"),
                "gate_rate": chosen.get("gate_rate"),
            })
    return pd.DataFrame(out)

def short_stage2_text_2424():
    return (
        "The 24/24 experiments were used as the first controlled comparison between the main Stage 2 gate architectures. "
        "The BiLSTM run is the most permissive and tends to preserve more flare windows, while MiniRocket is the most conservative and keeps the quiet leakage lower. "
        "The Transformer sits between them, and together these runs define the recall-versus-cleanliness trade-off before scaling the gate to the longer 72-hour windows."
    )

def short_stage2_text_w72():
    return (
        "The 72-hour runs answer a different question: whether a longer temporal history improves the gate enough to justify the extra input context. "
        "In practice, these runs are used to choose an operating point for each (window, horizon) setup, based on how much flare recall is preserved relative to the quiet-to-flare false-positive rate."
    )

def short_stage2_dataset_text(run_key: str):
    mapping = {
        "W72_H24": "This setup keeps the long 72-hour context while predicting the 24-hour horizon. It tests whether long historical context helps the gate separate quiet from flare more effectively in the short-horizon case.",
        "W72_H48": "This setup keeps the same long context but shifts the prediction to a 48-hour horizon. The main question here is whether the gate remains stable when the target is moved farther away in time.",
        "W72_H72": "This is the longest context and the longest horizon in the current Stage 2 sweep. It is useful mainly as a robustness check for whether the gate can still preserve flare signal under the widest forecasting setup.",
    }
    return mapping.get(run_key, "")
