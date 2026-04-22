# ==============================================================================
# ClassBuffer helper
# Source: Handling Class Imbalance.ipynb cell 18
# ==============================================================================
from sklearn.metrics import confusion_matrix, classification_report
from collections import deque
import random

# ==== add above run_stage3b_stable ====
class ClassBuffer:
    def __init__(self, max_per_class=2048, add_per_class_per_batch=32):
        self.max_per_class = int(max_per_class)
        self.add_per_class_per_batch = int(add_per_class_per_batch)
        self.buf = {
            IDX_B: deque(maxlen=self.max_per_class),
            IDX_C: deque(maxlen=self.max_per_class),
            IDX_M: deque(maxlen=self.max_per_class),
            IDX_X: deque(maxlen=self.max_per_class),
        }

    def add(self, Xf: torch.Tensor, yf: torch.Tensor):
        # Xf:[N,T,F], yf:[N] (flare-only)
        for cls in [IDX_B, IDX_C, IDX_M, IDX_X]:
            idx = torch.where(yf == cls)[0]
            if idx.numel() == 0:
                continue
            k = min(self.add_per_class_per_batch, int(idx.numel()))
            take = idx[torch.randperm(idx.numel(), device=idx.device)[:k]]
            for i in take.tolist():
                self.buf[cls].append(Xf[i].detach().cpu().clone())

    def can_sample(self, need):
        # with replacement sampling => only need at least 1 sample/class
        for cls, k in need.items():
            if int(k) > 0 and len(self.buf[cls]) == 0:
                return False
        return True

    def sample(self, need: dict):
        xs, ys = [], []
        for cls, k in need.items():
            for _ in range(int(k)):
                xs.append(random.choice(self.buf[cls]))  # replacement
                ys.append(cls)
        X = torch.stack(xs, dim=0)                     # cpu
        y = torch.tensor(ys, dtype=torch.long)         # cpu
        p = torch.randperm(X.size(0))
        return X[p], y[p]

    def sizes(self):
        return {k: len(v) for k, v in self.buf.items()}


# ==============================================================================
# Generic Stage3B stable runner
# Source: Handling Class Imbalance.ipynb cell 19
# ==============================================================================
# ===== Stage3B from frozen Stage3A (locked t_flare=0.04, t_reject in [0.20,0.25]) =====
from dataclasses import dataclass, asdict
from typing import Optional, Tuple
import os, json, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# assumes already defined in notebook:
# IDX_Q, IDX_B, IDX_C, IDX_M, IDX_X, classes, name2id
# NPZFileDataset, make_fixed_loader, collate_take_random_samples_fixedT, set_seed
# SimpleTCN, Stage3ABCMXTCN
# ---------- 1) Use ONE evaluator for baseline + sweeps ----------

def y5_to_bcsev_torch(y5: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(y5)
    out[y5 == IDX_B] = 0
    out[y5 == IDX_C] = 1
    out[(y5 == IDX_M) | (y5 == IDX_X)] = 2
    return out

def logits4_to_bcsev(logits4: torch.Tensor) -> torch.Tensor:
    # logits4 order: B,C,M,X
    l_b = logits4[:, 0:1]
    l_c = logits4[:, 1:2]
    l_s = torch.logsumexp(logits4[:, 2:4], dim=1, keepdim=True)  # merge M/X
    return torch.cat([l_b, l_c, l_s], dim=1)

def pick_best_conf_with_floor(conf_rows, baseline_severe_recall, floor_ratio=0.95):
    floor = float(floor_ratio) * float(baseline_severe_recall)

    filtered = [r for r in conf_rows if float(r["severe_recall"]) >= floor]
    target = filtered if len(filtered) > 0 else conf_rows

    # maximize severe recall, then minimize quiet->flare fpr
    target = sorted(target, key=lambda r: (-float(r["severe_recall"]), float(r["quiet_to_flare_fpr"])))
    best = target[0]
    best["severe_floor_used"] = floor
    best["floor_satisfied"] = len(filtered) > 0
    return best

def sweep_conf_bcmx_on_val(model_s1, model_s3, dl_val, device, t_flare, t_reject, t_conf_grid):
    rows = []
    for tc in t_conf_grid:
        y_val, pred_val = predict_final_5class(
            model_s1, model_s3, dl_val, device,
            t_flare=t_flare, t_reject=t_reject, t_conf_bcmx=tc
        )
        m = eval_5class_metrics(y_val, pred_val)
        rows.append({
            "t_conf_bcmx": float(tc),
            "severe_recall": float(m["severe_recall"]),
            "quiet_to_flare_fpr": float(m["quiet_to_flare_fpr"]),
        })
    return rows

def quota_sample_bcmx(ys5, total=128, quota=(0.40,0.30,0.25,0.05), x_min=0):
    device = ys5.device
    pools = [
        torch.where(ys5 == IDX_B)[0],
        torch.where(ys5 == IDX_C)[0],
        torch.where(ys5 == IDX_M)[0],
        torch.where(ys5 == IDX_X)[0],
    ]

    q = torch.tensor(quota, dtype=torch.float32, device=device)
    q = q / q.sum()

    target = torch.floor(q * float(total)).long()
    while int(target.sum().item()) < int(total):
        j = int(torch.argmax(q - target.float() / max(float(total), 1.0)).item())
        target[j] += 1

    # no X forcing if empty
    if pools[3].numel() == 0:
        target[3] = 0
    else:
        target[3] = max(int(target[3].item()), int(x_min))

    chosen = []
    for k, pool in zip(target.tolist(), pools):
        if k <= 0 or pool.numel() == 0:
            continue
        if pool.numel() >= k:
            take = pool[torch.randperm(pool.numel(), device=device)[:k]]
        else:
            take = pool[torch.randint(0, pool.numel(), (k,), device=device)]
        chosen.append(take)

    if not chosen:
        return None

    idx = torch.cat(chosen, dim=0)

    # IMPORTANT: pad if short (missing classes)
    if idx.numel() < total:
        flare_all = torch.where(torch.isin(ys5, torch.tensor([IDX_B,IDX_C,IDX_M,IDX_X], device=device)))[0]
        k = total - idx.numel()
        extra = flare_all[torch.randint(0, flare_all.numel(), (k,), device=device)]
        idx = torch.cat([idx, extra], dim=0)

    idx = idx[torch.randperm(idx.numel(), device=device)]
    return idx

def _extract_state_any(ckpt):
    for k in ("model_state", "model_state_dict", "model"):
        if k in ckpt:
            return ckpt[k]
    return ckpt

def y5_to_bcmx_index_torch(y5: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(y5)
    out[y5 == IDX_B] = 0
    out[y5 == IDX_C] = 1
    out[y5 == IDX_M] = 2
    out[y5 == IDX_X] = 3
    return out

ID_FROM_BCMX = np.array([IDX_B, IDX_C, IDX_M, IDX_X], dtype=np.int64)


def quota_sample_bcmx_min(
    ys5: torch.Tensor,
    total: int = 128,
    quota=(0.40, 0.30, 0.20, 0.10),  # B,C,M,X
    m_min: int = 1,
    x_min: int = 1,):
    device = ys5.device
    pools = {
        IDX_B: torch.where(ys5 == IDX_B)[0],
        IDX_C: torch.where(ys5 == IDX_C)[0],
        IDX_M: torch.where(ys5 == IDX_M)[0],
        IDX_X: torch.where(ys5 == IDX_X)[0],
    }

    q = torch.tensor(quota, dtype=torch.float32, device=device)
    q = q / q.sum()
    target = torch.floor(q * int(total)).long()
    while int(target.sum().item()) < int(total):
        target[torch.argmax(q - target.float() / max(float(total), 1.0))] += 1

    # enforce minimums only if class exists
    # index order: B,C,M,X -> 0,1,2,3
    if pools[IDX_M].numel() > 0:
        target[2] = max(int(target[2].item()), int(m_min))
    if pools[IDX_X].numel() > 0:
        target[3] = max(int(target[3].item()), int(x_min))

    # if class missing, force target to 0
    if pools[IDX_M].numel() == 0:
        target[2] = 0
    if pools[IDX_X].numel() == 0:
        target[3] = 0

    # adjust overflow if minimums increased total
    while int(target.sum().item()) > int(total):
        for j in [0, 1, 2, 3]:
            min_j = 0
            if j == 2 and pools[IDX_M].numel() > 0:
                min_j = int(m_min)
            if j == 3 and pools[IDX_X].numel() > 0:
                min_j = int(x_min)
            if int(target[j].item()) > min_j:
                target[j] -= 1
                if int(target.sum().item()) <= int(total):
                    break

    idx_take = []
    for j, cls in enumerate([IDX_B, IDX_C, IDX_M, IDX_X]):
        pool = pools[cls]
        k = int(target[j].item())
        if k <= 0 or pool.numel() == 0:
            continue
        if pool.numel() >= k:
            sel = pool[torch.randperm(pool.numel(), device=device)[:k]]
        else:
            sel = pool[torch.randint(0, pool.numel(), (k,), device=device)]
        idx_take.append(sel)

    if not idx_take:
        return None

    idx = torch.cat(idx_take, dim=0)
    idx = idx[torch.randperm(idx.numel(), device=device)]
    return idx

@torch.no_grad()
def _frozen_gate_survivor_mask(model_s1_frozen, model_s3a_frozen, X, t_flare, t_reject):
    p1 = torch.softmax(model_s1_frozen(X), dim=1)[:, 1]
    logits_rej, _ = model_s3a_frozen(X)
    p2 = torch.softmax(logits_rej, dim=1)[:, 1]
    return (p1 >= float(t_flare)) & (p2 >= float(t_reject))

@torch.no_grad()
def predict_final_5class_with_conf(model_s1_frozen, model_s3a_frozen, model_s3_train, loader, device, t_flare, t_reject, t_conf_bcmx):
    y_all, pred_all = [], []
    model_s1_frozen.eval()
    model_s3a_frozen.eval()
    model_s3_train.eval()

    for X, y5 in loader:
        X = X.to(device, non_blocking=True)
        y_np = y5.numpy().astype(np.int64)

        surv = _frozen_gate_survivor_mask(model_s1_frozen, model_s3a_frozen, X, t_flare, t_reject)
        pred = np.full(len(y_np), IDX_Q, dtype=np.int64)

        if surv.any():
            _, logits4 = model_s3_train(X[surv])
            probs4 = torch.softmax(logits4, dim=1).detach().cpu().numpy()
            conf = np.max(probs4, axis=1)
            arg4 = np.argmax(probs4, axis=1)

            keep = conf >= float(t_conf_bcmx)
            idx_surv = torch.where(surv)[0].detach().cpu().numpy()
            idx_keep = idx_surv[keep]
            pred[idx_keep] = ID_FROM_BCMX[arg4[keep]]

        y_all.append(y_np)
        pred_all.append(pred)

    return np.concatenate(y_all), np.concatenate(pred_all)

def eval_5class_summary(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[IDX_Q, IDX_B, IDX_C, IDX_M, IDX_X])

    quiet_true = (y_true == IDX_Q)
    severe_true = np.isin(y_true, [IDX_M, IDX_X])
    severe_pred = np.isin(y_pred, [IDX_M, IDX_X])

    severe_support = int(np.sum(severe_true))
    severe_recall = float(np.sum(severe_true & severe_pred)) / max(float(severe_support), 1.0)
    quiet_to_flare_fpr = float(np.sum(quiet_true & (y_pred != IDX_Q))) / max(float(np.sum(quiet_true)), 1.0)

    # non-negotiable extra visibility
    c_support = int(np.sum(y_true == IDX_C))
    c_recall = float(np.sum((y_true == IDX_C) & (y_pred == IDX_C))) / max(float(c_support), 1.0)

    return {
        "severe_recall": severe_recall,
        "quiet_to_flare_fpr": quiet_to_flare_fpr,
        "severe_support": severe_support,
        "c_support": c_support,
        "c_recall": c_recall,
        "confusion_matrix": cm.tolist(),
    }

# ===== add ABOVE run_stage3b_stable =====
def eval_stage3a_gate(y5, p_flare, p_reject, t_flare, t_reject):
    gate = (p_flare >= float(t_flare))
    final_flare = gate & (p_reject >= float(t_reject))
    quiet_true = (y5 == IDX_Q)
    flare_true = (y5 != IDX_Q)
    severe_true = np.isin(y5, [IDX_M, IDX_X])
    return {
        "flare_recall": float(np.sum(flare_true & final_flare) / max(np.sum(flare_true), 1)),
        "severe_recall": float(np.sum(severe_true & final_flare) / max(np.sum(severe_true), 1)),
        "quiet_to_flare_fpr": float(np.sum(quiet_true & final_flare) / max(np.sum(quiet_true), 1)),
        "pred_quiet_rate": float((~final_flare).mean()),
        "gate_rate": float(gate.mean()),
    }

def baseline_severe_recall_from_npz(npz_path, t_flare, t_reject):
    z = np.load(npz_path)
    m = eval_stage3a_gate(
        y5=z["y5"], p_flare=z["p_flare"], p_reject=z["p_reject"],
        t_flare=t_flare, t_reject=t_reject
    )
    return float(m["severe_recall"])


def pick_best_conf(
    df_conf: pd.DataFrame,
    severe_target: float,
    quiet_fpr_max: float,
    baseline_severe_recall: Optional[float] = None,
    baseline_floor_ratio: float = 0.95,):
    df = df_conf.copy()
    if baseline_severe_recall is not None:
        floor = float(baseline_floor_ratio) * float(baseline_severe_recall)
        df_floor = df[df["severe_recall"] >= floor]
        if len(df_floor) > 0:
            df = df_floor

    feasible = df[
        (df["severe_recall"] >= float(severe_target)) &
        (df["quiet_to_flare_fpr"] <= float(quiet_fpr_max))
    ]
    target = feasible if len(feasible) else df
    best = target.sort_values(
        ["severe_recall", "quiet_to_flare_fpr"],
        ascending=[False, True]
    ).iloc[0]
    return best.to_dict(), bool(len(feasible) > 0)

@torch.no_grad()
def eval_phaseA_bcmx_flare_only(model_s3, loader, device):
    model_s3.eval()
    y_true4_all, y_pred4_all = [], []

    for X, y5 in loader:
        flare = (y5 != IDX_Q)
        if flare.sum().item() == 0:
            continue

        Xf = X[flare].to(device, non_blocking=True)
        y5f = y5[flare].to(device, non_blocking=True)

        y4_true = y5_to_bcmx_index_torch(y5f).detach().cpu().numpy()
        _, logits4 = model_s3(Xf)
        y4_pred = torch.argmax(logits4, dim=1).detach().cpu().numpy()

        y_true4_all.append(y4_true)
        y_pred4_all.append(y4_pred)

    y_true4 = np.concatenate(y_true4_all)
    y_pred4 = np.concatenate(y_pred4_all)

    cm = confusion_matrix(y_true4, y_pred4, labels=[0,1,2,3])  # B,C,M,X
    rec = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)

    out = {
        "cm_bcmx": cm.tolist(),
        "recall_B": float(rec[0]),
        "recall_C": float(rec[1]),
        "recall_M": float(rec[2]),
        "recall_X": float(rec[3]),
        "report": classification_report(
            y_true4, y_pred4, labels=[0,1,2,3], target_names=["B","C","M","X"], zero_division=0, output_dict=True
        ),
    }
    return out

@torch.no_grad()
def predict_final_5class(model_s1, model_s3, loader, device, t_flare, t_reject, t_conf_bcmx=0.0):
    model_s1.eval(); model_s3.eval()
    y_all, pred_all = [], []

    for X, y5 in loader:
        X = X.to(device, non_blocking=True)
        y_np = y5.numpy().astype(np.int64)

        p1 = torch.softmax(model_s1(X), dim=1)[:, 1].detach().cpu().numpy()  # Stage1 p(flare)
        logits_rej, logits4 = model_s3(X)
        p2 = torch.softmax(logits_rej, dim=1)[:, 1].detach().cpu().numpy()   # Stage3A p(flare)
        probs4 = torch.softmax(logits4, dim=1).detach().cpu().numpy()         # Stage3B BCMX probs

        pass12 = (p1 >= float(t_flare)) & (p2 >= float(t_reject))

        pred = np.full(len(y_np), IDX_Q, dtype=np.int64)
        if np.any(pass12):
            p4 = probs4[pass12]
            conf = np.max(p4, axis=1)
            arg4 = np.argmax(p4, axis=1)
            keep_conf = conf >= float(t_conf_bcmx)

            idx_pass12 = np.where(pass12)[0]
            idx_keep = idx_pass12[keep_conf]
            pred[idx_keep] = ID_FROM_BCMX[arg4[keep_conf]]

        y_all.append(y_np)
        pred_all.append(pred)

    return np.concatenate(y_all), np.concatenate(pred_all)

def eval_5class_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[IDX_Q, IDX_B, IDX_C, IDX_M, IDX_X])

    severe_true = np.isin(y_true, [IDX_M, IDX_X])
    severe_pred = np.isin(y_pred, [IDX_M, IDX_X])
    quiet_true = (y_true == IDX_Q)

    severe_recall = float(np.sum(severe_true & severe_pred)) / max(float(np.sum(severe_true)), 1.0)
    quiet_to_flare_fpr = float(np.sum(quiet_true & (y_pred != IDX_Q))) / max(float(np.sum(quiet_true)), 1.0)

    rep = classification_report(
        y_true, y_pred,
        labels=[IDX_Q, IDX_B, IDX_C, IDX_M, IDX_X],
        target_names=["quiet", "B", "C", "M", "X"],
        output_dict=True,
        zero_division=0
    )
    return {
        "severe_recall": severe_recall,
        "quiet_to_flare_fpr": quiet_to_flare_fpr,
        "confusion_matrix": cm.tolist(),
        "classification_report": rep,
    }

@dataclass
class Stage3BFromRejectCfg:
    root4: str
    W: int
    H: int
    out_root: str
    s1_ckpt_path: str
    s3a_ckpt_path: str

    max_train_files: Optional[int] = None
    max_val_files: Optional[int] = None
    max_test_files: Optional[int] = None

    batch_files: int = 8
    per_file: int = 256
    fixed_batch_size: int = 512
    hidden: int = 64
    lr: float = 2e-4
    seed: int = 123
    clamp_abs: float = 1e3

    s3b_epochs: int = 8
    # keep your current class weights (do not retune now)
    s3b_alpha: Tuple[float, float, float, float] = (0.06205621, 0.1368635, 0.17784386, 3.62323644)

    # locked thresholds
    t_flare: float = 0.04
    t_reject_choices: Tuple[float, ...] = (0.20, 0.25)

    use_fixed_eval: bool = False

    s3b_finetune_last_conv: bool = True
    s3b_quota_total: int = 128
    s3b_quota: Tuple[float, float, float, float] = (0.40, 0.30, 0.25, 0.05)
    s3b_x_min: int = 0
    debug_survivors_once_per_epoch: bool = True

@dataclass
class Stage3BStableCfg:
    root4: str
    W: int
    H: int
    out_root: str
    s1_ckpt_path: str
    s3a_ckpt_path: str

    max_train_files: Optional[int] = None
    max_val_files: Optional[int] = None
    max_test_files: Optional[int] = None

    batch_files: int = 16#8
    per_file: int = 512#256
    fixed_batch_size: int = 512
    hidden: int = 64
    seed: int = 123
    clamp_abs: float = 1e3

    # locked gate (Step 1)
    t_flare: float = 0.04
    t_reject_choices: Tuple[float, ...] = (0.20, 0.25)

    # Phase A (Step 2)
    phaseA_epochs: int = 6

    # Phase B (Step 3)
    phaseB_epochs: int = 6



    # Step 4: confidence reject sweep
    severe_target: float = 0.90
    quiet_fpr_target: float = 0.10

    # Step 5: fixed eval preferred
    use_fixed_eval: bool = True


    s3a_val_npz: Optional[str] = None
    baseline_floor_ratio: float = 0.95

        # ===================== PATCH 1: cfg =====================
    # Add/replace these fields in Stage3BStableCfg

    phaseA_quota_total: int = 512
    phaseB_quota_total: int = 256

    phaseA_quota: Tuple[float, float, float, float] = (0.25, 0.25, 0.30, 0.20)
    phaseB_quota: Tuple[float, float, float, float] = (0.25, 0.25, 0.30, 0.20)

    phaseA_m_min: int = 8
    phaseA_x_min: int = 8
    phaseB_m_min: int = 8
    phaseB_x_min: int = 8

    phaseA_lr_backbone: float = 5e-5
    phaseA_lr_head: float = 5e-4
    phaseB_lr_backbone: float = 5e-5
    phaseB_lr_head: float = 2e-4

    bcmx_ce_weights: Tuple[float, float, float, float] = (1.0, 1.0, 2.0, 8.0)  # B,C,M,X
    quiet_entropy_lambda: float = 0.05  # 0.02-0.10
    t_conf_grid: Tuple[float, ...] = tuple(np.round(np.arange(0.0, 1.0, 0.02), 2))
    save_phaseA_each_epoch: bool = True
    save_phaseA_every: int = 1
    save_phaseA_last: bool = True

    phaseA_buffer_max_per_class: int = 2048
    phaseA_buffer_add_per_class: int = 32
    phaseA_need: Tuple[int, int, int, int] = (64, 64, 64, 4)  # B,C,M,X
    phaseA_steps_per_epoch: int = 200

    train_bcsev: bool = True


def run_stage3b_after_reject(cfg: Stage3BFromRejectCfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(cfg.seed)
    print("device:", device)

    man_dir = os.path.join(cfg.root4, "manifests_by_group")
    man_train = os.path.join(man_dir, f"manifest_train_W{cfg.W}h_H{cfg.H}h.json")
    man_val   = os.path.join(man_dir, f"manifest_val_W{cfg.W}h_H{cfg.H}h.json")
    man_test  = os.path.join(man_dir, f"manifest_test_W{cfg.W}h_H{cfg.H}h.json")

    out_dir = os.path.join(cfg.out_root, f"W{cfg.W}_H{cfg.H}_stage3b_after_reject")
    os.makedirs(out_dir, exist_ok=True)

    ds_train = NPZFileDataset(man_train, name2id, max_files=cfg.max_train_files, allow_pickle=True)
    ds_val   = NPZFileDataset(man_val, name2id, max_files=cfg.max_val_files, allow_pickle=True)

    X0, _ = ds_train[0]
    T_fixed = int(X0.shape[1])
    F = int(X0.shape[2])

    dl_train = DataLoader(
        ds_train,
        batch_size=cfg.batch_files,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=lambda b: collate_take_random_samples_fixedT(
            b, per_file=cfg.per_file, T_fixed=T_fixed, clamp_abs=cfg.clamp_abs
        ),
    )

    if cfg.use_fixed_eval:
        base_fixed = os.path.join(cfg.out_root, f"W{cfg.W}_H{cfg.H}_two_stage_bal")
        val_fixed = os.path.join(base_fixed, f"val_fixed_bal_W{cfg.W}_H{cfg.H}.npz")
        test_fixed = os.path.join(base_fixed, f"test_fixed_bal_W{cfg.W}_H{cfg.H}.npz")
        dl_val = make_fixed_loader(val_fixed, batch_size=cfg.fixed_batch_size, num_workers=0) if os.path.exists(val_fixed) else None
        dl_test = make_fixed_loader(test_fixed, batch_size=cfg.fixed_batch_size, num_workers=0) if os.path.exists(test_fixed) else None
    else:
        dl_val = DataLoader(
            ds_val, batch_size=cfg.batch_files, shuffle=False, num_workers=0, pin_memory=True,
            collate_fn=lambda b: collate_take_random_samples_fixedT(
                b, per_file=cfg.per_file, T_fixed=T_fixed, clamp_abs=cfg.clamp_abs
            ),
        )
        if os.path.exists(man_test):
            ds_test = NPZFileDataset(man_test, name2id, max_files=cfg.max_test_files, allow_pickle=True)
            dl_test = DataLoader(
                ds_test, batch_size=cfg.batch_files, shuffle=False, num_workers=0, pin_memory=True,
                collate_fn=lambda b: collate_take_random_samples_fixedT(
                    b, per_file=cfg.per_file, T_fixed=T_fixed, clamp_abs=cfg.clamp_abs
                ),
            )
        else:
            dl_test = None

    if dl_val is None:
        raise RuntimeError("Validation loader is missing.")

    # frozen Stage1
    ckpt1 = torch.load(cfg.s1_ckpt_path, map_location=device)
    model_s1 = SimpleTCN(num_features=F, num_classes=2, hidden=cfg.hidden).to(device)
    model_s1.load_state_dict(_extract_state_any(ckpt1))
    model_s1.eval()

    # Stage3A+B model initialized from stage3a checkpoint
    base_ckpt = torch.load(cfg.s3a_ckpt_path, map_location=device)

        # ---- Phase A: pretrain BCMX on all flares (once) ----
    model_pre = Stage3ABCMXTCN(num_features=F, hidden=cfg.hidden).to(device)
    model_pre.load_state_dict(_extract_state_any(base_ckpt))

    # freeze reject head; train backbone + BCMX head
    for p in model_pre.head_reject.parameters():
        p.requires_grad = False
    for p in model_pre.net.parameters():
        p.requires_grad = True
    for p in model_pre.head_bcmx.parameters():
        p.requires_grad = True

    optA = torch.optim.Adam([
        {"params": model_pre.net.parameters(), "lr": 1e-4},
        {"params": model_pre.head_bcmx.parameters(), "lr": 1e-3},
    ])
    lossA = nn.CrossEntropyLoss()  # no class weights in Phase A

    for ep in range(1, 6):
        model_pre.train()
        total, n = 0.0, 0

        for X, y5 in dl_train:
            X = X.to(device, non_blocking=True)
            y5 = y5.to(device, non_blocking=True)

            flare = (y5 != IDX_Q)
            if flare.sum().item() == 0:
                continue
            Xf, yf = X[flare], y5[flare]

            idxq = quota_sample_bcmx_min(
                ys5=yf,
                total=cfg.phaseA_quota_total,   # NOT cfg.s3b_quota_total
                quota=cfg.phaseA_quota,
                m_min=cfg.phaseA_m_min,
                x_min=cfg.phaseA_x_min,
            )
            if idxq is None:
                continue
            Xq, yq = Xf[idxq], yf[idxq]

            _, logits4 = model_pre(Xq)
            y4 = y5_to_bcmx_index_torch(yq)
            loss = lossA(logits4, y4)

            if not torch.isfinite(loss):
                continue

            optA.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_pre.parameters(), 1.0)
            optA.step()

            total += float(loss.item()) * Xq.size(0)
            n += Xq.size(0)

        print(f"[PhaseA Ep {ep}] loss={total/max(n,1):.4f}")

    base_ckpt_for_B = {"model_state": model_pre.state_dict()}

    all_runs = []
    for t_reject in cfg.t_reject_choices:
        print(f"\n=== Train Stage3B with locked t_flare={cfg.t_flare}, t_reject={t_reject} ===")
        # Stage3A+B model initialized from stage3a checkpoint
        model_s3 = Stage3ABCMXTCN(num_features=F, hidden=cfg.hidden).to(device)
        #model_s3.load_state_dict(_extract_state_any(base_ckpt))
        model_s3.load_state_dict(_extract_state_any(base_ckpt_for_B))


        # freeze everything first
        for p in model_s3.parameters():
            p.requires_grad = False

        # keep reject head frozen
        for p in model_s3.head_reject.parameters():
            p.requires_grad = False

        # unfreeze BCMX head
        for p in model_s3.head_bcmx.parameters():
            p.requires_grad = True

        # optional: unfreeze last conv block for better BCMX separation
        if cfg.s3b_finetune_last_conv:
            for p in model_s3.net[2].parameters():  # second Conv1d
                p.requires_grad = True

        #loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(cfg.s3b_alpha, dtype=torch.float32, device=device))
        loss_fn = nn.CrossEntropyLoss()
        opt = torch.optim.Adam([p for p in model_s3.parameters() if p.requires_grad], lr=cfg.lr)




        # train Stage3B ONLY on survivors and ONLY non-quiet labels
        for ep in range(1, cfg.s3b_epochs + 1):
            model_s3.train()
            model_s1.eval()
            model_s3.head_reject.eval()
            if not cfg.s3b_finetune_last_conv:
                model_s3.net.eval()

            total, n = 0.0, 0

            # epoch-level survivor counts (before quota)
            b_e = c_e = m_e = x_e = 0
            # sampled counts (after quota, what actually trained)
            b_tr = c_tr = m_tr = x_tr = 0

            for X, y5 in dl_train:
                X = X.to(device, non_blocking=True)
                y5 = y5.to(device, non_blocking=True)

                with torch.no_grad():
                    p1 = torch.softmax(model_s1(X), dim=1)[:, 1]
                    logits_rej, _ = model_s3(X)
                    p2 = torch.softmax(logits_rej, dim=1)[:, 1]

                surv = (p1 >= float(cfg.t_flare)) & (p2 >= float(t_reject)) & (y5 != IDX_Q)
                if surv.sum().item() == 0:
                    continue

                Xs = X[surv]
                ys = y5[surv]

                # survivors distribution across epoch
                b_e += int((ys == IDX_B).sum().item())
                c_e += int((ys == IDX_C).sum().item())
                m_e += int((ys == IDX_M).sum().item())
                x_e += int((ys == IDX_X).sum().item())

                idxq = quota_sample_bcmx(
                    ys,
                    total=cfg.s3b_quota_total,
                    quota=cfg.s3b_quota,
                    x_min=cfg.s3b_x_min,
                )
                if idxq is None:
                    continue

                Xq = Xs[idxq]
                yq = ys[idxq]

                # trained distribution across epoch
                b_tr += int((yq == IDX_B).sum().item())
                c_tr += int((yq == IDX_C).sum().item())
                m_tr += int((yq == IDX_M).sum().item())
                x_tr += int((yq == IDX_X).sum().item())

                _, logits4 = model_s3(Xq)
                y4 = y5_to_bcmx_index_torch(yq)
                loss = loss_fn(logits4, y4)
                if not torch.isfinite(loss):
                    continue

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_s3.parameters(), 1.0)
                opt.step()

                total += float(loss.item()) * Xq.size(0)
                n += Xq.size(0)

            print(f"[S3B Ep {ep}] loss={total/max(n,1):.4f}")
            print(f"[Ep {ep}] survivors total B/C/M/X = {b_e}/{c_e}/{m_e}/{x_e}")
            print(f"[Ep {ep}] trained   total B/C/M/X = {b_tr}/{c_tr}/{m_tr}/{x_tr}")

        stage3a_json = r"runs_pytorch_stage2_fix\W24_H24_stage3a_reject_only\stage3a_reject_metrics.json"
        
        with open(stage3a_json, "r", encoding="utf-8") as f:
            s3a = json.load(f)

        # baseline severe recall from Stage3A chosen point
        baseline_severe_recall = float(s3a["best_val"]["flare_recall"])

        print("baseline_severe_recall:", baseline_severe_recall)

        conf_rows = sweep_conf_bcmx_on_val(
            model_s1, model_s3, dl_val, device,
            t_flare=cfg.t_flare, t_reject=t_reject,
            t_conf_grid=[0.0, 0.3, 0.4, 0.5, 0.6],
        )

        best_conf_row = pick_best_conf_with_floor(
            conf_rows,
            baseline_severe_recall=baseline_severe_recall,
            floor_ratio=0.95
        )
        best_conf = float(best_conf_row["t_conf_bcmx"])

        # eval
        y_val, pred_val = predict_final_5class(model_s1, model_s3, dl_val, device, cfg.t_flare, t_reject, t_conf_bcmx=best_conf)
        val_metrics = eval_5class_metrics(y_val, pred_val)

        test_metrics = None
        if dl_test is not None:
            y_test, pred_test = predict_final_5class(model_s1, model_s3, dl_test, device, cfg.t_flare, t_reject, t_conf_bcmx=best_conf)
            test_metrics = eval_5class_metrics(y_test, pred_test)

        run = {
            "t_flare": float(cfg.t_flare),
            "t_reject": float(t_reject),
            "val": {
                "severe_recall": val_metrics["severe_recall"],
                "quiet_to_flare_fpr": val_metrics["quiet_to_flare_fpr"],
            },
            "test": None if test_metrics is None else {
                "severe_recall": test_metrics["severe_recall"],
                "quiet_to_flare_fpr": test_metrics["quiet_to_flare_fpr"],
            },
            "val_full": val_metrics,
            "test_full": test_metrics,
        }
        all_runs.append(run)

        # save model for this threshold
        torch.save(
            {"model_state": model_s3.state_dict(), "extra": {"cfg": asdict(cfg), "t_reject": float(t_reject)}},
            os.path.join(out_dir, f"stage3b_after_reject_trej_{t_reject:.2f}.pt")
        )

    # choose best by val severe_recall then lower quiet_to_flare_fpr
    all_runs_sorted = sorted(
        all_runs,
        key=lambda r: (-r["val"]["severe_recall"], r["val"]["quiet_to_flare_fpr"])
    )
    best = all_runs_sorted[0]

    out = {
        "out_dir": out_dir,
        "locked_thresholds": {"t_flare": float(cfg.t_flare), "t_reject_choices": list(cfg.t_reject_choices)},
        "best": {
            "t_flare": best["t_flare"],
            "t_reject": best["t_reject"],
            "val": best["val"],
            "test": best["test"],
        },
        "runs": [
            {
                "t_flare": r["t_flare"],
                "t_reject": r["t_reject"],
                "val": r["val"],
                "test": r["test"],
            } for r in all_runs_sorted
        ],
    }

    with open(os.path.join(out_dir, "stage3b_after_reject_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # optional full metrics dump
    with open(os.path.join(out_dir, "stage3b_after_reject_full_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(all_runs_sorted, f, indent=2)

    print("\nBest locked setup:", out["best"])
    print("Saved:", os.path.join(out_dir, "stage3b_after_reject_summary.json"))
    return out

def run_stage3b_stable(cfg: Stage3BStableCfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(cfg.seed)
    print("device:", device)

    man_dir = os.path.join(cfg.root4, "manifests_by_group")
    man_train = os.path.join(man_dir, f"manifest_train_W{cfg.W}h_H{cfg.H}h.json")
    man_val   = os.path.join(man_dir, f"manifest_val_W{cfg.W}h_H{cfg.H}h.json")
    man_test  = os.path.join(man_dir, f"manifest_test_W{cfg.W}h_H{cfg.H}h.json")

    out_dir = os.path.join(cfg.out_root, f"W{cfg.W}_H{cfg.H}_stage3b_stable")
    os.makedirs(out_dir, exist_ok=True)

    ds_train = NPZFileDataset(man_train, name2id, max_files=cfg.max_train_files, allow_pickle=True)
    ds_val   = NPZFileDataset(man_val, name2id, max_files=cfg.max_val_files, allow_pickle=True)

    X0, _ = ds_train[0]
    T_fixed = int(X0.shape[1])
    F = int(X0.shape[2])

    dl_train = DataLoader(
        ds_train, batch_size=cfg.batch_files, shuffle=True, num_workers=0, pin_memory=True,
        collate_fn=lambda b: collate_take_random_samples_fixedT(
            b, per_file=cfg.per_file, T_fixed=T_fixed, clamp_abs=cfg.clamp_abs
        ),
    )

    if cfg.use_fixed_eval:
        base_fixed = os.path.join(cfg.out_root, f"W{cfg.W}_H{cfg.H}_two_stage_bal")
        val_fixed = os.path.join(base_fixed, f"val_fixed_bal_W{cfg.W}_H{cfg.H}.npz")
        test_fixed = os.path.join(base_fixed, f"test_fixed_bal_W{cfg.W}_H{cfg.H}.npz")
        dl_val = make_fixed_loader(val_fixed, batch_size=cfg.fixed_batch_size, num_workers=0) if os.path.exists(val_fixed) else None
        dl_test = make_fixed_loader(test_fixed, batch_size=cfg.fixed_batch_size, num_workers=0) if os.path.exists(test_fixed) else None
    else:
        dl_val = DataLoader(
            ds_val, batch_size=cfg.batch_files, shuffle=False, num_workers=0, pin_memory=True,
            collate_fn=lambda b: collate_take_random_samples_fixedT(
                b, per_file=cfg.per_file, T_fixed=T_fixed, clamp_abs=cfg.clamp_abs
            ),
        )
        if os.path.exists(man_test):
            ds_test = NPZFileDataset(man_test, name2id, max_files=cfg.max_test_files, allow_pickle=True)
            dl_test = DataLoader(
                ds_test, batch_size=cfg.batch_files, shuffle=False, num_workers=0, pin_memory=True,
                collate_fn=lambda b: collate_take_random_samples_fixedT(
                    b, per_file=cfg.per_file, T_fixed=T_fixed, clamp_abs=cfg.clamp_abs
                ),
            )
        else:
            dl_test = None

    if dl_val is None:
        raise RuntimeError("Validation loader missing.")

    # ---- frozen gates ----
    ckpt1 = torch.load(cfg.s1_ckpt_path, map_location=device)
    model_s1_frozen = SimpleTCN(num_features=F, num_classes=2, hidden=cfg.hidden).to(device)
    model_s1_frozen.load_state_dict(_extract_state_any(ckpt1))
    model_s1_frozen.eval()
    for p in model_s1_frozen.parameters():
        p.requires_grad = False

    ckpt3a = torch.load(cfg.s3a_ckpt_path, map_location=device)
    model_s3a_frozen = Stage3ABCMXTCN(num_features=F, hidden=cfg.hidden).to(device)
    model_s3a_frozen.load_state_dict(_extract_state_any(ckpt3a))
    model_s3a_frozen.eval()
    for p in model_s3a_frozen.parameters():
        p.requires_grad = False

    # baseline severe per t_reject from frozen gate output
    s3a_val_npz = cfg.s3a_val_npz or os.path.join(os.path.dirname(cfg.s3a_ckpt_path), "val_arrays_stage3a.npz")
    baseline_sev_by_treject = {
        float(tr): baseline_severe_recall_from_npz(s3a_val_npz, t_flare=cfg.t_flare, t_reject=float(tr))
        for tr in cfg.t_reject_choices
    }
    print("baseline severe recall by t_reject:", baseline_sev_by_treject)

    # weighted CE (Fix #2)
    w = torch.tensor(cfg.bcmx_ce_weights, dtype=torch.float32, device=device)

    # ---- Phase A: pretrain BCMX on all flares ----
    model_pre = Stage3ABCMXTCN(num_features=F, hidden=cfg.hidden).to(device)
    model_pre.load_state_dict(_extract_state_any(ckpt3a))
    for p in model_pre.head_reject.parameters():
        p.requires_grad = False
    for p in model_pre.net.parameters():
        p.requires_grad = True
    for p in model_pre.head_bcmx.parameters():
        p.requires_grad = True

    optA = torch.optim.Adam([
        {"params": model_pre.net.parameters(), "lr": cfg.phaseA_lr_backbone},
        {"params": model_pre.head_bcmx.parameters(), "lr": cfg.phaseA_lr_head},
    ])
    #w_bcmx = torch.tensor([1.0, 1.5, 1.5, 4.0], dtype=torch.float32, device=device)  # B,C,M,X

    lossA = nn.CrossEntropyLoss(weight=w)

    lossA_4 = nn.CrossEntropyLoss(weight=w)
    lossB_4 = nn.CrossEntropyLoss(weight=w)

    lossA_3 = nn.CrossEntropyLoss(weight=w)
    lossB_3 = nn.CrossEntropyLoss(weight=w)

    buf = ClassBuffer(
        max_per_class=cfg.phaseA_buffer_max_per_class,
        add_per_class_per_batch=cfg.phaseA_buffer_add_per_class,
    )
    need = {
        IDX_B: int(cfg.phaseA_need[0]),
        IDX_C: int(cfg.phaseA_need[1]),
        IDX_M: int(cfg.phaseA_need[2]),
        IDX_X: int(cfg.phaseA_need[3]),
    }


    for ep in range(1, cfg.phaseA_epochs + 1):
        ep_t0 = time.perf_counter()
        model_pre.train()
        total, n, steps = 0.0, 0, 0
        b_e = c_e = m_e = x_e = 0
        seen_in_stream_x = 0

        for X, y5 in dl_train:
            X = X.to(device, non_blocking=True)
            y5 = y5.to(device, non_blocking=True)

            flare = (y5 != IDX_Q)
            if flare.sum().item() == 0:
                continue

            Xf, yf = X[flare], y5[flare]
            seen_in_stream_x += int((yf == IDX_X).sum().item())
            buf.add(Xf, yf)

            if not buf.can_sample(need):
                continue

            Xq_cpu, yq_cpu = buf.sample(need)
            Xq = Xq_cpu.to(device, non_blocking=True)
            yq = yq_cpu.to(device, non_blocking=True)

            _, logits4 = model_pre(Xq)
            if cfg.train_bcsev:
                logits3 = logits4_to_bcsev(logits4)
                y3 = y5_to_bcsev_torch(yq)
                loss = lossA_3(logits3, y3)
            else:
                y4 = y5_to_bcmx_index_torch(yq)
                loss = lossA_4(logits4, y4)
            if not torch.isfinite(loss):
                continue

            optA.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_pre.parameters(), 1.0)
            optA.step()

            b_e += int((yq == IDX_B).sum().item())
            c_e += int((yq == IDX_C).sum().item())
            m_e += int((yq == IDX_M).sum().item())
            x_e += int((yq == IDX_X).sum().item())

            total += float(loss.item()) * Xq.size(0)
            n += Xq.size(0)
            steps += 1
            if steps >= cfg.phaseA_steps_per_epoch:
                break

        s = buf.sizes()
        ep_sec = time.perf_counter() - ep_t0
        print(
            f"[PhaseA Ep {ep}] loss={total/max(n,1):.4f} | steps={steps} | "
            f"train B/C/M/X={b_e}/{c_e}/{m_e}/{x_e} | "
            f"buf B/C/M/X={s[IDX_B]}/{s[IDX_C]}/{s[IDX_M]}/{s[IDX_X]} | "
            f"stream_X_seen={seen_in_stream_x} | {ep_sec:.1f}s"
        )

        if steps == 0:
            raise RuntimeError("PhaseA made 0 optimizer steps. Relax sampling or increase flare-rich batching.")

        if cfg.save_phaseA_each_epoch and (ep % cfg.save_phaseA_every == 0):
            p_ep = os.path.join(out_dir, f"stage3b_phaseA_ep_{ep:02d}.pt")
            torch.save(
                {
                    "model_state": model_pre.state_dict(),
                    "optimizer_state": optA.state_dict(),
                    "extra": {"cfg": asdict(cfg), "phase": "A", "epoch": ep},
                },
                p_ep,
            )
            print(f"[PhaseA Ep {ep}] saved: {p_ep} | exists={os.path.exists(p_ep)}")

    if cfg.save_phaseA_last:
        p_last = os.path.join(out_dir, "stage3b_phaseA_last.pt")
        torch.save(
            {
                "model_state": model_pre.state_dict(),
                "optimizer_state": optA.state_dict(),
                "extra": {"cfg": asdict(cfg), "phase": "A", "epoch": cfg.phaseA_epochs},
            },
            p_last,
        )
        print(f"[PhaseA] final saved: {p_last} | exists={os.path.exists(p_last)}")

    phaseA_val = eval_phaseA_bcmx_flare_only(model_pre, dl_val, device)
    print("PhaseA flare-only recalls:",
        phaseA_val["recall_B"], phaseA_val["recall_C"], phaseA_val["recall_M"], phaseA_val["recall_X"])

    base_state_for_B = {"model_state": model_pre.state_dict()}

    # ---- Phase B per t_reject + conf sweep ----
    all_runs = []
    for t_reject in cfg.t_reject_choices:
        print(f"\n=== PhaseB t_flare={cfg.t_flare}, t_reject={t_reject} ===")
        baseline_sev = baseline_sev_by_treject[float(t_reject)]

        model_s3_train = Stage3ABCMXTCN(num_features=F, hidden=cfg.hidden).to(device)
        model_s3_train.load_state_dict(_extract_state_any(base_state_for_B))

        for p in model_s3_train.head_reject.parameters():
            p.requires_grad = False
        for p in model_s3_train.net.parameters():
            p.requires_grad = True
        for p in model_s3_train.head_bcmx.parameters():
            p.requires_grad = True

        optB = torch.optim.Adam([
            {"params": model_s3_train.net.parameters(), "lr": cfg.phaseB_lr_backbone},
            {"params": model_s3_train.head_bcmx.parameters(), "lr": cfg.phaseB_lr_head},
        ])
        lossB = nn.CrossEntropyLoss(weight=w)
        #lossB = nn.CrossEntropyLoss(weight=w)

        for ep in range(1, cfg.phaseB_epochs + 1):
            ep_t0 = time.perf_counter()
            model_s3_train.train()
            total, n = 0.0, 0
            b_e = c_e = m_e = x_e = 0

            for X, y5 in dl_train:
                X = X.to(device, non_blocking=True)
                y5 = y5.to(device, non_blocking=True)

                with torch.no_grad():
                    surv = _frozen_gate_survivor_mask(model_s1_frozen, model_s3a_frozen, X, cfg.t_flare, t_reject)

                if surv.sum().item() == 0:
                    continue

                Xs_all, ys_all = X[surv], y5[surv]  # include quiet survivors
                is_quiet = (ys_all == IDX_Q)
                is_flare = ~is_quiet

                loss = None

                # CE on flare survivors + quota (Fix #3)
                if is_flare.any():
                    Xf, yf = Xs_all[is_flare], ys_all[is_flare]
                    idxq = quota_sample_bcmx_min(
                        ys5=yf,
                        total=cfg.phaseB_quota_total,
                        quota=cfg.phaseB_quota,
                        m_min=cfg.phaseB_m_min,
                        x_min=cfg.phaseB_x_min,
                    )
                    if idxq is not None:
                        Xq, yq = Xf[idxq], yf[idxq]
                        b_e += int((yq == IDX_B).sum().item())
                        c_e += int((yq == IDX_C).sum().item())
                        m_e += int((yq == IDX_M).sum().item())
                        x_e += int((yq == IDX_X).sum().item())

                        _, logits4 = model_s3_train(Xq)
                        if cfg.train_bcsev:
                            logits3 = logits4_to_bcsev(logits4)
                            y3 = y5_to_bcsev_torch(yq)
                            loss_ce = lossB_3(logits3, y3)
                        else:
                            y4 = y5_to_bcmx_index_torch(yq)
                            loss_ce = lossB_4(logits4, y4)
                        loss = loss_ce if loss is None else (loss + loss_ce)

                # entropy maximize on quiet survivors (Fix #4)
                if is_quiet.any():
                    _, logits4_q = model_s3_train(Xs_all[is_quiet])
                    p = torch.softmax(logits4_q, dim=1)
                    loss_ent = -(p * torch.log(p.clamp_min(1e-8))).sum(dim=1).mean()
                    ent_term = -cfg.quiet_entropy_lambda * loss_ent
                    loss = ent_term if loss is None else (loss + ent_term)

                if (loss is None) or (not torch.isfinite(loss)):
                    continue

                optB.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_s3_train.parameters(), 1.0)
                optB.step()

                total += float(loss.item()) * Xs_all.size(0)
                n += Xs_all.size(0)
            ep_sec = time.perf_counter() - ep_t0
            print(f"[PhaseB Ep {ep}] loss={total/max(n,1):.4f} | seen B/C/M/X={b_e}/{c_e}/{m_e}/{x_e} | {ep_sec:.1f}s")
            print(f"[PhaseB Ep {ep}] loss={total/max(n,1):.4f} | seen B/C/M/X={b_e}/{c_e}/{m_e}/{x_e}")

        # conf sweep
        conf_rows = []
        for tc in cfg.t_conf_grid:
            y_val, pred_val = predict_final_5class_with_conf(
                model_s1_frozen, model_s3a_frozen, model_s3_train,
                dl_val, device, cfg.t_flare, t_reject, tc
            )
            m = eval_5class_summary(y_val, pred_val)
            conf_rows.append({
                "t_conf_bcmx": float(tc),
                "severe_recall": float(m["severe_recall"]),
                "quiet_to_flare_fpr": float(m["quiet_to_flare_fpr"]),
                "severe_support": int(m.get("severe_support", 0)),
                "c_support": int(m.get("c_support", 0)),
                "c_recall": float(m.get("c_recall", 0.0)),
            })

        df_conf = pd.DataFrame(conf_rows)
        best_conf, feasible = pick_best_conf(
            df_conf,
            severe_target=cfg.severe_target,
            quiet_fpr_max=cfg.quiet_fpr_target,
            baseline_severe_recall=baseline_sev,
            baseline_floor_ratio=cfg.baseline_floor_ratio,
        )
        t_conf_best = float(best_conf["t_conf_bcmx"])

        # save per t_reject model (Fix #5)
        ckpt_path = os.path.join(out_dir, f"stage3b_stable_trej_{float(t_reject):.2f}.pt")
        torch.save(
            {
                "model_state": model_s3_train.state_dict(),
                "extra": {"cfg": asdict(cfg), "t_reject": float(t_reject), "t_conf_best": t_conf_best},
            },
            ckpt_path,
        )

        # final val/test at chosen conf
        y_val, pred_val = predict_final_5class_with_conf(
            model_s1_frozen, model_s3a_frozen, model_s3_train,
            dl_val, device, cfg.t_flare, t_reject, t_conf_best
        )
        val_metrics = eval_5class_summary(y_val, pred_val)

        test_metrics = None
        if dl_test is not None:
            y_test, pred_test = predict_final_5class_with_conf(
                model_s1_frozen, model_s3a_frozen, model_s3_train,
                dl_test, device, cfg.t_flare, t_reject, t_conf_best
            )
            test_metrics = eval_5class_summary(y_test, pred_test)

        all_runs.append({
            "t_flare": float(cfg.t_flare),
            "t_reject": float(t_reject),
            "baseline_severe_recall": float(baseline_sev),
            "best_t_conf_bcmx": t_conf_best,
            "best_conf_feasible": bool(feasible),
            "ckpt_path": ckpt_path,
            "val": val_metrics,
            "test": test_metrics,
            "val_conf_sweep": conf_rows,
        })

    all_runs_sorted = sorted(all_runs, key=lambda r: (-r["val"]["severe_recall"], r["val"]["quiet_to_flare_fpr"]))
    best = all_runs_sorted[0]

    out = {
        "out_dir": out_dir,
        "locked_gate": {"t_flare": cfg.t_flare, "t_reject_choices": list(cfg.t_reject_choices)},
        "baseline_sev_by_treject": baseline_sev_by_treject,
        "best": {
            "t_flare": best["t_flare"],
            "t_reject": best["t_reject"],
            "t_conf_bcmx": best["best_t_conf_bcmx"],
            "ckpt_path": best["ckpt_path"],
            "val": {
                "severe_recall": best["val"]["severe_recall"],
                "quiet_to_flare_fpr": best["val"]["quiet_to_flare_fpr"],
                "severe_support": best["val"].get("severe_support", 0),
                "c_recall": best["val"].get("c_recall", 0.0),
            },
            "test": None if best["test"] is None else {
                "severe_recall": best["test"]["severe_recall"],
                "quiet_to_flare_fpr": best["test"]["quiet_to_flare_fpr"],
                "severe_support": best["test"].get("severe_support", 0),
                "c_recall": best["test"].get("c_recall", 0.0),
            },
        },
        "runs": all_runs_sorted,
    }

    with open(os.path.join(out_dir, "stage3b_stable_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    with open(os.path.join(out_dir, "stage3b_stable_full_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(all_runs_sorted, f, indent=2)

    print("Saved:", os.path.join(out_dir, "stage3b_stable_summary.json"))
    print("Saved:", os.path.join(out_dir, "stage3b_stable_full_metrics.json"))
    print("Best:", out["best"])
    return out

# Example run:
#cfg_locked = Stage3BFromRejectCfg(
#    root4=r"section3_windows_forecast_24_fixed_section4_fe_ext_compact_TRUE",
#    W=24, H=24,
#    out_root=r"runs_pytorch_stage2_fix",
#    s1_ckpt_path=r"runs_pytorch_stage2_fix\W24_H24_two_stage_bal_other_stage1\stage1_last.pt",
#    s3a_ckpt_path=r"runs_pytorch_stage2_fix\W24_H24_stage3a_reject_only\stage3a_reject_last.pt",
#    s3b_epochs=8,
#    t_flare=0.04,
#    t_reject_choices=(0.20, 0.25),
#    use_fixed_eval=False,
#)
#res_locked = run_stage3b_after_reject(cfg_locked)
#print(res_locked["best"])

# Example:
#cfg_stable = Stage3BStableCfg(
#    root4=r"section3_windows_forecast_24_fixed_section4_fe_ext_compact_TRUE",
#    W=24, H=24,
#    out_root=r"runs_pytorch_stage2_fix",
#    s1_ckpt_path=r"runs_pytorch_stage2_fix\W24_H24_two_stage_bal_other_stage1\stage1_last.pt",
#    s3a_ckpt_path=r"runs_pytorch_stage2_fix\W24_H24_stage3a_reject_only\stage3a_reject_last.pt",
#    use_fixed_eval=True,
#)



# Good Run
# ===================== PATCH 4: night run config =====================
#cfg_stable = Stage3BStableCfg(
#    root4=r"section3_windows_forecast_24_fixed_section4_fe_ext_compact_TRUE",
#    W=24, H=24,
#    out_root=r"runs_pytorch_stage2_fix",
#    s1_ckpt_path=r"runs_pytorch_stage2_fix\W24_H24_two_stage_bal_other_stage1\stage1_last.pt",
#    s3a_ckpt_path=r"runs_pytorch_stage2_fix\W24_H24_stage3a_reject_only\stage3a_reject_last.pt",
#    use_fixed_eval=True,
#)
#
#cfg_stable.phaseA_epochs = 6
#cfg_stable.phaseB_epochs = 12
#cfg_stable.phaseA_quota_total = 256
#cfg_stable.phaseB_quota_total = 256
#
#cfg_stable.phaseA_quota = (0.25, 0.25, 0.30, 0.20)
#cfg_stable.phaseB_quota = (0.25, 0.25, 0.30, 0.20)
#
#cfg_stable.phaseA_m_min = 1
#cfg_stable.phaseA_x_min = 0
#cfg_stable.phaseB_m_min = 1
#cfg_stable.phaseB_x_min = 0
#
#cfg_stable.phaseA_lr_backbone = 5e-5
#cfg_stable.phaseA_lr_head = 5e-4
#cfg_stable.phaseB_lr_backbone = 5e-5
#cfg_stable.phaseB_lr_head = 2e-4
#
#cfg_stable.bcmx_ce_weights = (1.0, 1.0, 2.0)#(1.0, 1.5, 1.5, 4.0)
#cfg_stable.quiet_entropy_lambda = 0.05
#cfg_stable.t_conf_grid = tuple(np.round(np.arange(0.0, 1.0, 0.02), 2))
##res = run_stage3b_stable(cfg_stable)
## for Phase A quality checks, use natural loaders


# ===================== M-FIX RUN (BiLSTM backbone) =====================
# Put this in CELL 19 (after run_stage3b_stable is defined)

import os
import numpy as np
import torch
import torch.nn as nn


# ===================== Stage3A BiLSTM (replace class + run) =====================
import os
import torch
import torch.nn as nn

# Keep same class name so existing run_stage3a_reject_only() code uses it
class Stage3ABCMXTCN(nn.Module):
    def __init__(self, num_features: int, hidden: int = 64, layers: int = 1, bidirectional: bool = True):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden,
            num_layers=max(1, int(layers)),
            batch_first=True,               # [N,T,F]
            bidirectional=bool(bidirectional),
            dropout=0.0 if int(layers) <= 1 else 0.1,
        )
        out_dim = hidden * (2 if bool(bidirectional) else 1)
        self.head_reject = nn.Linear(out_dim, 2)  # quiet/flare
        self.head_bcmx   = nn.Linear(out_dim, 4)  # B,C,M,X

    def forward(self, x):
        h_seq, _ = self.rnn(x)             # [N,T,H*dir]
        h = h_seq.mean(dim=1)              # temporal pooling
        return self.head_reject(h), self.head_bcmx(h)

# Train Stage3A reject-only with BiLSTM
cfg_s3a_bilstm = Stage3ARejectCfg(
    root4=r"section3_windows_forecast_24_fixed_section4_fe_ext_compact_TRUE",
    W=24, H=24,
    out_root=r"runs_pytorch_stage2_fix\W24_H24_stage3a_reject_only_bilstm",
    s1_ckpt_path=r"runs_pytorch_stage2_fix\W24_H24_two_stage_bal_other_stage1\stage1_last.pt",
)

# optional tuning
cfg_s3a_bilstm.hidden = 64
cfg_s3a_bilstm.s3a_epochs = 8
cfg_s3a_bilstm.t_flare_train = 0.04
cfg_s3a_bilstm.use_fixed_eval = False

#res_s3a_bilstm = run_stage3a_reject_only(cfg_s3a_bilstm)
#print(res_s3a_bilstm)



# 1) Override Stage3 class name used by run_stage3b_stable
# Keep the same class name so existing code path in cell 18 picks it up.
class Stage3ABCMXTCN(nn.Module):
    """
    BiLSTM version (name kept for compatibility with existing runner code).
    Returns:
      - head_reject logits: [N,2]
      - head_bcmx logits:   [N,4]
    """
    def __init__(self, num_features: int, hidden: int = 64, layers: int = 1, bidirectional: bool = True):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden,
            num_layers=max(1, int(layers)),
            batch_first=True,          # input [N,T,F]
            bidirectional=bool(bidirectional),
            dropout=0.0 if int(layers) <= 1 else 0.1,
        )
        out_dim = hidden * (2 if bool(bidirectional) else 1)
        self.head_reject = nn.Linear(out_dim, 2)
        self.head_bcmx = nn.Linear(out_dim, 4)

    def forward(self, x):
        # x: [N,T,F]
        h_seq, _ = self.rnn(x)
        h = h_seq.mean(dim=1)  # temporal pooling
        return self.head_reject(h), self.head_bcmx(h)


# 2) Your config (same logic as your current cell, with new output folder)
cfg_m_fix = Stage3BStableCfg(
    root4=r"section3_windows_forecast_24_fixed_section4_fe_ext_compact_TRUE",
    W=24, H=24,
    out_root=r"runs_pytorch_stage2_fix",
    s1_ckpt_path=r"runs_pytorch_stage2_fix\W24_H24_two_stage_bal_other_stage1\stage1_last.pt",
    # IMPORTANT: this must be a BiLSTM Stage3A checkpoint (NOT TCN)
    s3a_ckpt_path=r"runs_pytorch_stage2_fix\W24_H24_stage3a_reject_only\stage3a_reject_last.pt",
)

cfg_m_fix.train_bcsev = False
cfg_m_fix.bcmx_ce_weights = (1.0, 2.0, 20.0, 40.0)

cfg_m_fix.t_flare = 0.02
cfg_m_fix.t_reject_choices = (0.15, 0.20)

cfg_m_fix.phaseA_epochs = 10
cfg_m_fix.phaseA_steps_per_epoch = 300
cfg_m_fix.phaseA_need = (32, 32, 128, 16)
cfg_m_fix.phaseA_buffer_max_per_class = 4096
cfg_m_fix.phaseA_buffer_add_per_class = 64
cfg_m_fix.phaseA_lr_backbone = 1e-4
cfg_m_fix.phaseA_lr_head = 1e-3

cfg_m_fix.phaseB_epochs = 10
cfg_m_fix.phaseB_quota_total = 256
cfg_m_fix.phaseB_quota = (0.10, 0.15, 0.50, 0.25)
cfg_m_fix.phaseB_m_min = 16
cfg_m_fix.phaseB_x_min = 4
cfg_m_fix.phaseB_lr_backbone = 1e-4
cfg_m_fix.phaseB_lr_head = 5e-4

cfg_m_fix.quiet_entropy_lambda = 0.05
cfg_m_fix.t_conf_grid = tuple(np.round(np.arange(0.0, 0.6, 0.02), 2))
cfg_m_fix.severe_target = 0.70
cfg_m_fix.quiet_fpr_target = 0.15
cfg_m_fix.baseline_floor_ratio = 0.80
cfg_m_fix.save_phaseA_each_epoch = True
cfg_m_fix.save_phaseA_every = 2
cfg_m_fix.use_fixed_eval = False

# 3) Run to NEW folder
#_original_run = run_stage3b_stable

def run_stage3b_stable_m_fix_bilstm(cfg):
    old_out_root = cfg.out_root
    cfg.out_root = os.path.join(old_out_root, "m_fix_bilstm_v1")  # new folder
    os.makedirs(cfg.out_root, exist_ok=True)
    try:
        return _original_run(cfg)
    finally:
        cfg.out_root = old_out_root

#print("Ready. Output folder:", r"runs_pytorch_stage2_fix\m_fix_bilstm_v1")
#res_m_fix_bilstm = run_stage3b_stable_m_fix_bilstm(cfg_m_fix)
#print(res_m_fix_bilstm["best"])


# ==============================================================================
# TCN m_fix baseline run config
# Source: Handling Class Imbalance.ipynb cell 21
# ==============================================================================
# ===================== M-FIX RUN: Stage3B with full BCMX + heavy M weight =====================
# Drop this cell into your notebook AFTER the cell that defines run_stage3b_stable
# It uses your existing infrastructure — no new classes needed.
#
# KEY CHANGES vs your last run:
# 1. train_bcsev = False  (train full B,C,M,X — NOT collapsed B,C,Sev)
# 2. bcmx_ce_weights = (1.0, 2.0, 20.0, 40.0)  — M gets 20x, X gets 40x
# 3. phaseA_need = (32, 32, 128, 16)  — M gets 60% of buffer samples
# 4. t_flare = 0.02  — let all M through the gate
# 5. More epochs, more steps
# ==============================================================================================

import numpy as np

cfg_m_fix = Stage3BStableCfg(
    root4=r"section3_windows_forecast_24_fixed_section4_fe_ext_compact_TRUE",
    W=24, H=24,
    out_root=r"runs_pytorch_stage2_fix",
    s1_ckpt_path=r"runs_pytorch_stage2_fix\W24_H24_two_stage_bal_other_stage1\stage1_last.pt",
    s3a_ckpt_path=r"runs_pytorch_stage2_fix\W24_H24_stage3a_reject_only\stage3a_reject_last.pt",
    #use_fixed_eval=True,
)

# ---- CRITICAL: train full BCMX, NOT collapsed BCSev ----
cfg_m_fix.train_bcsev = False  # <-- THIS IS THE MAIN FIX

# ---- M-heavy class weights for [B, C, M, X] ----
cfg_m_fix.bcmx_ce_weights = (1.0, 2.0, 20.0, 40.0)

# ---- Gate: lower to let all M through ----
cfg_m_fix.t_flare = 0.02
cfg_m_fix.t_reject_choices = (0.15, 0.20)  # slightly lower reject too

# ---- Phase A: M-heavy buffer sampling ----
cfg_m_fix.phaseA_epochs = 10
cfg_m_fix.phaseA_steps_per_epoch = 300
cfg_m_fix.phaseA_need = (32, 32, 128, 16)       # M gets 128/208 = 61% of samples
cfg_m_fix.phaseA_buffer_max_per_class = 4096
cfg_m_fix.phaseA_buffer_add_per_class = 64
cfg_m_fix.phaseA_lr_backbone = 1e-4              # slightly higher than before
cfg_m_fix.phaseA_lr_head = 1e-3

# ---- Phase B: M-heavy quota ----
cfg_m_fix.phaseB_epochs = 10
cfg_m_fix.phaseB_quota_total = 256
cfg_m_fix.phaseB_quota = (0.10, 0.15, 0.50, 0.25)  # M gets 50%
cfg_m_fix.phaseB_m_min = 16
cfg_m_fix.phaseB_x_min = 4
cfg_m_fix.phaseB_lr_backbone = 1e-4
cfg_m_fix.phaseB_lr_head = 5e-4

# ---- Entropy regularization on quiet survivors ----
cfg_m_fix.quiet_entropy_lambda = 0.05

# ---- Confidence sweep grid ----
cfg_m_fix.t_conf_grid = tuple(np.round(np.arange(0.0, 0.6, 0.02), 2))

# ---- Eval targets ----
cfg_m_fix.severe_target = 0.70   # lower than 0.90 since we're rebalancing
cfg_m_fix.quiet_fpr_target = 0.15
cfg_m_fix.baseline_floor_ratio = 0.80  # more lenient floor

# ---- Save everything ----
cfg_m_fix.save_phaseA_each_epoch = True
cfg_m_fix.save_phaseA_every = 2
cfg_m_fix.use_fixed_eval = False
# ---- Override output dir so it doesn't overwrite your previous run ----
# We monkey-patch out_root to write to a new folder
import os
_original_run = run_stage3b_stable

def run_stage3b_stable_m_fix(cfg):
    """Wrapper that redirects output to m_fix subfolder."""
    # Override the output directory inside the function
    old_out_root = cfg.out_root
    cfg.out_root = os.path.join(old_out_root, "m_fix_v1")
    os.makedirs(cfg.out_root, exist_ok=True)
    try:
        return _original_run(cfg)
    finally:
        cfg.out_root = old_out_root

print("="*60)
print("M-FIX CONFIG SUMMARY")
print("="*60)
print(f"  train_bcsev:       {cfg_m_fix.train_bcsev}  (CHANGED: was True)")
print(f"  bcmx_ce_weights:   {cfg_m_fix.bcmx_ce_weights}  (CHANGED: was (1.0, 1.0, 2.0))")
print(f"  t_flare:           {cfg_m_fix.t_flare}  (CHANGED: was 0.04)")
print(f"  phaseA_need:       {cfg_m_fix.phaseA_need}  (CHANGED: M gets 61%)")
print(f"  phaseB_quota:      {cfg_m_fix.phaseB_quota}  (CHANGED: M gets 50%)")
print(f"  phaseA_epochs:     {cfg_m_fix.phaseA_epochs}")
print(f"  phaseB_epochs:     {cfg_m_fix.phaseB_epochs}")
print(f"  Output dir:        runs_pytorch_stage2_fix/m_fix_v1/")
print("="*60)
print("\nReady to run. Execute: res_m_fix = run_stage3b_stable_m_fix(cfg_m_fix)")
print("Expected runtime: ~20-40 min depending on GPU")

# UNCOMMENT TO RUN:
res_m_fix = run_stage3b_stable_m_fix(cfg_m_fix)
print(res_m_fix["best"])
