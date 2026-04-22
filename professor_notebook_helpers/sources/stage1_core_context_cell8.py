import os, json, time
import numpy as np
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, f1_score, confusion_matrix, recall_score, precision_score

# ---------------------------
# Labels
# ---------------------------
classes = ["quiet","B","C","M","X"]
name2id = {c:i for i,c in enumerate(classes)}
id2name = {i:c for c,i in name2id.items()}
K = len(classes)

IDX_Q = name2id["quiet"]
IDX_B = name2id["B"]
IDX_C = name2id["C"]
IDX_M = name2id["M"]
IDX_X = name2id["X"]

import random, os
import numpy as np
import torch

def set_seed(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = False  # מהיר יותר
    torch.backends.cudnn.benchmark = True       # מהיר יותר

def collate_stage2_quota_fixedT(
    batch,
    per_file: int,
    T_fixed: int,
    clamp_abs: float = None,
    quiet_id: int = 0,
    B_id: int = 1,
    C_id: int = 2,
    M_id: int = 3,
    X_id: int = 4,
    flare_frac: float = 0.80,
    flare_quota = (0.30, 0.30, 0.30, 0.10),
    x_min_per_file: int = 0,
    x_max_per_file: int = 8,
    pad_value: float = 0.0,
):
    X_out, y_out = [], []
    n_flare = int(round(per_file * flare_frac))
    n_rest  = per_file - n_flare

    if not isinstance(flare_quota, dict):
        if len(flare_quota) != 4:
            raise ValueError("flare_quota must have 4 values for B,C,M,X")
        flare_quota = {
            B_id: float(flare_quota[0]),
            C_id: float(flare_quota[1]),
            M_id: float(flare_quota[2]),
            X_id: float(flare_quota[3]),
        }


    flare_ids = list(flare_quota.keys())
    weights = np.array([flare_quota[i] for i in flare_ids], dtype=np.float32)
    weights = weights / max(weights.sum(), 1e-12)

    for (X_file, y_file) in batch:
        y_np = y_file if isinstance(y_file, np.ndarray) else y_file.cpu().numpy()
        N = len(y_np)
        if N == 0:
            continue

        idx_by = {cid: np.flatnonzero(y_np == cid) for cid in [quiet_id, B_id, C_id, M_id, X_id]}
        chosen = []

        counts = {cid: int(round(n_flare * w)) for cid, w in zip(flare_ids, weights)}
        while sum(counts.values()) > n_flare:
            k = max(counts, key=lambda c: counts[c])
            counts[k] -= 1
        while sum(counts.values()) < n_flare:
            k = max([B_id, C_id, M_id], key=lambda c: len(idx_by[c]))
            counts[k] += 1

        if X_id in counts:
            counts[X_id] = min(counts[X_id], x_max_per_file)
            if counts[X_id] < x_min_per_file and len(idx_by[X_id]) > 0:
                counts[X_id] = min(x_min_per_file, len(idx_by[X_id]))

        for cid, k in counts.items():
            pool = idx_by[cid]
            if len(pool) == 0 or k <= 0:
                continue
            take = np.random.choice(pool, size=min(k, len(pool)), replace=(len(pool) < k))
            chosen.append(take)

        if n_rest > 0:
            pool = idx_by[quiet_id]
            if len(pool) == 0:
                pool = np.arange(N)
            take = np.random.choice(pool, size=min(n_rest, len(pool)), replace=(len(pool) < n_rest))
            chosen.append(take)

        if len(chosen):
            chosen = np.concatenate(chosen)
        else:
            chosen = np.random.choice(np.arange(N), size=min(per_file, N), replace=(N < per_file))

        X_sel = X_file[chosen]
        y_sel = y_np[chosen]

        # ✅ crop/pad ל-T_fixed
        T_here = X_sel.shape[1]
        if T_here > T_fixed:
            X_sel = X_sel[:, :T_fixed, :]
            nan_count = np.isnan(X_sel).sum()
            if nan_count > 0:
                print(f"[COLLATE] NaNs in X_sel before clip:", nan_count)
            X_sel = np.nan_to_num(X_sel, nan=0.0, posinf=0.0, neginf=0.0)

        elif T_here < T_fixed:
            pad = np.full((X_sel.shape[0], T_fixed - T_here, X_sel.shape[2]), pad_value, dtype=X_sel.dtype)
            X_sel = np.concatenate([X_sel, pad], axis=1)

        # ✅ הקריטי: ניקוי NaN/Inf
        X_sel = np.nan_to_num(X_sel, nan=0.0, posinf=0.0, neginf=0.0)

        if clamp_abs is not None:
            X_sel = np.clip(X_sel, -clamp_abs, clamp_abs)

        X_out.append(torch.as_tensor(X_sel, dtype=torch.float32))
        y_out.append(torch.as_tensor(y_sel, dtype=torch.long))

    X_out = torch.cat(X_out, dim=0)
    y_out = torch.cat(y_out, dim=0)
    return X_out, y_out

def gate_checker(y_true, y_pred_on, y_pred_off, labels, focus=("B","C","M","X")):
    # Macro-F1
    macro_f1_on  = f1_score(y_true, y_pred_on,  labels=labels, average="macro", zero_division=0)
    macro_f1_off = f1_score(y_true, y_pred_off, labels=labels, average="macro", zero_division=0)

    # Per-class recall
    rec_on  = recall_score(y_true, y_pred_on,  labels=labels, average=None, zero_division=0)
    rec_off = recall_score(y_true, y_pred_off, labels=labels, average=None, zero_division=0)
    rec_on  = dict(zip(labels, rec_on))
    rec_off = dict(zip(labels, rec_off))

    # Confusion matrices
    cm_on  = confusion_matrix(y_true, y_pred_on,  labels=labels)
    cm_off = confusion_matrix(y_true, y_pred_off, labels=labels)

    # Decision rules
    macro_drop = macro_f1_on < macro_f1_off
    recall_drop = any(rec_on[c] < rec_off[c] for c in focus)

    hurts = macro_drop or recall_drop
    decision = "keep gate" if not hurts else "soften/remove gate"

    return {
        "macro_f1_on": macro_f1_on,
        "macro_f1_off": macro_f1_off,
        "recall_on": rec_on,
        "recall_off": rec_off,
        "cm_on": cm_on,
        "cm_off": cm_off,
        "hurts": hurts,
        "decision": decision,
    }

def tune_sev_thresholds_with_constraints(p_flare, probs2, y_true, t_flare, grid,
                                         min_prec_B=0.02, min_rec_M=0.35, min_rec_X=0.35):
    # labels: [quiet, B, C, M, X]
    best = None
    best_info = None

    # fallback best (no constraints)
    best_any = -1e9
    best_any_info = None

    for tB in grid:
        for tC in grid:
            for tM in grid:
                for tX in grid:
                    t_sev = [tB, tC, tM, tX]

                    pred = two_stage_predict_with_quiet(
                        p_flare, probs2, t_flare=t_flare, t_sev=t_sev
                    )

                    mf1 = f1_score(y_true, pred, average="macro", labels=[0,1,2,3,4], zero_division=0)

                    # always track best_any
                    if mf1 > best_any:
                        best_any = mf1
                        best_any_info = {"t_sev": t_sev, "macro_f1": mf1, "constraint_met": False}

                    # check constraints
                    prec_B = precision_score(y_true, pred, labels=[1], average=None, zero_division=0)[0]
                    rec_M  = recall_score(y_true, pred, labels=[3], average=None, zero_division=0)[0]
                    rec_X  = recall_score(y_true, pred, labels=[4], average=None, zero_division=0)[0]

                    if prec_B < min_prec_B or rec_M < min_rec_M or rec_X < min_rec_X:
                        continue

                    # constraint-met candidate
                    if best is None or mf1 > best:
                        best = mf1
                        best_info = {
                            "t_sev": t_sev,
                            "macro_f1": mf1,
                            "prec_B": prec_B,
                            "rec_M": rec_M,
                            "rec_X": rec_X,
                            "constraint_met": True
                        }

    # if no constraint-met result, return best_any
    return best_info if best_info is not None else best_any_info

def two_stage_predict_with_quiet(p_flare, probs2, t_flare, t_sev):
    # probs2 shape: [N,5] -> [quiet,B,C,M,X]
    pred = np.zeros(len(p_flare), dtype=np.int64)

    for i in range(len(p_flare)):
        if p_flare[i] < t_flare:
            pred[i] = IDX_Q
            continue

        p = probs2[i]  # 5-class probs
        # thresholds for B/C/M/X in order
        tB, tC, tM, tX = t_sev

        # if quiet is highest or all flares below thresholds -> quiet
        if p[0] >= 0.90 and p[0] >= max(p[1], p[2], p[3], p[4]):
            pred[i] = IDX_Q
            continue

        # apply thresholds
        cand = [
            (1, p[1], tB),
            (2, p[2], tC),
            (3, p[3], tM),
            (4, p[4], tX),
        ]
        cand = [c for c in cand if c[1] >= c[2]]

        if not cand:
            pred[i] = IDX_Q
        else:
            pred[i] = max(cand, key=lambda x: x[1])[0]

    return pred

def sweep_stage1_thresholds(p_flare, y_true, thresholds, min_precision=None):
    y_bin = (y_true != IDX_Q).astype(np.int64)
    best = None
    rows = []

    for t in thresholds:
        pred = (p_flare >= t).astype(np.int64)
        prec = precision_score(y_bin, pred, zero_division=0)
        rec  = recall_score(y_bin, pred, zero_division=0)
        f1   = f1_score(y_bin, pred, average="binary", zero_division=0)

        rows.append((t, prec, rec, f1))

        # choose "best" by recall (optionally with precision constraint)
        if min_precision is None or prec >= min_precision:
            if best is None or rec > best[2]:
                best = (t, prec, rec, f1)

    # print table
    print("\nThreshold sweep (VAL_NATURAL)")
    print("t\tprec\trec\tf1")
    for t, prec, rec, f1 in rows:
        print(f"{t:.3f}\t{prec:.3f}\t{rec:.3f}\t{f1:.3f}")

    print("\nBest (max recall", f"with prec>={min_precision}" if min_precision else "", "):")
    print(best)
    return rows, best

def reset_collate_stage2_hardquiet_stats():
    collate_stage2_quota_fixedT_hardquiet._stats = {
        "batches": 0,
        "batches_with_hardquiet": 0,
        "quiet_scores_buffer": [],      # sampled buffer for mean/median
        "last_batch_label_dist": {},
        "label_dist_history": [],       # list[dict], capped
    }

def get_collate_stage2_hardquiet_stats():
    if not hasattr(collate_stage2_quota_fixedT_hardquiet, "_stats"):
        reset_collate_stage2_hardquiet_stats()
    s = collate_stage2_quota_fixedT_hardquiet._stats
    qs = np.asarray(s["quiet_scores_buffer"], dtype=np.float32)
    out = dict(s)
    out["quiet_score_mean"] = float(qs.mean()) if qs.size else None
    out["quiet_score_median"] = float(np.median(qs)) if qs.size else None
    return out

def collate_stage2_quota_fixedT_hardquiet(
    batch,
    per_file: int,
    T_fixed: int,
    clamp_abs: float = None,
    quiet_id: int = 0,
    B_id: int = 1,
    C_id: int = 2,
    M_id: int = 3,
    X_id: int = 4,
    flare_frac: float = 0.80,
    flare_quota=(0.30, 0.30, 0.30, 0.10),  # B,C,M,X
    x_min_per_file: int = 0,
    x_max_per_file: int = 8,
    pad_value: float = 0.0,
    model_s1=None,
    device="cuda",
    hardquiet_mult: int = 8,
    hardquiet_mode: str = "topk",          # "topk" | "threshold"
    t_flare_train: float = None,
    log_every: int = 100,
):
    if model_s1 is None:
        raise ValueError("model_s1 must be provided for hard-negative quiet mining.")
    if hardquiet_mode not in ("topk", "threshold"):
        raise ValueError("hardquiet_mode must be 'topk' or 'threshold'.")
    if hardquiet_mode == "threshold" and t_flare_train is None:
        raise ValueError("t_flare_train is required when hardquiet_mode='threshold'.")

    if not hasattr(collate_stage2_quota_fixedT_hardquiet, "_stats"):
        reset_collate_stage2_hardquiet_stats()
    stats = collate_stage2_quota_fixedT_hardquiet._stats
    stats["batches"] += 1

    # flare quota normalization
    if not isinstance(flare_quota, dict):
        if len(flare_quota) != 4:
            raise ValueError("flare_quota must have 4 values for B,C,M,X")
        flare_quota = {
            B_id: float(flare_quota[0]),
            C_id: float(flare_quota[1]),
            M_id: float(flare_quota[2]),
            X_id: float(flare_quota[3]),
        }

    flare_ids = [B_id, C_id, M_id, X_id]
    q = np.array([max(float(flare_quota[i]), 0.0) for i in flare_ids], dtype=np.float32)
    q = q / max(float(q.sum()), 1e-12)

    n_flare = int(round(per_file * flare_frac))
    n_rest = per_file - n_flare

    def _fix_windows(X_arr):
        # crop/pad
        T_here = int(X_arr.shape[1])
        if T_here > T_fixed:
            X_arr = X_arr[:, :T_fixed, :]
        elif T_here < T_fixed:
            pad = np.full((X_arr.shape[0], T_fixed - T_here, X_arr.shape[2]), pad_value, dtype=X_arr.dtype)
            X_arr = np.concatenate([X_arr, pad], axis=1)
        # cleanup
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)
        if clamp_abs is not None:
            X_arr = np.clip(X_arr, -clamp_abs, clamp_abs)
        return X_arr

    X_out, y_out = [], []
    batch_quiet_scores = []
    used_hardquiet_this_batch = False

    for (X_file, y_file) in batch:
        y_np = y_file if isinstance(y_file, np.ndarray) else y_file.cpu().numpy()
        N = len(y_np)
        if N == 0:
            continue

        idx_by = {cid: np.flatnonzero(y_np == cid) for cid in [quiet_id, B_id, C_id, M_id, X_id]}
        chosen = []

        # 1) initial flare counts from quota
        counts = {cid: int(round(n_flare * w)) for cid, w in zip(flare_ids, q)}

        # force exact sum == n_flare
        while sum(counts.values()) > n_flare:
            k = max(counts, key=lambda c: counts[c])
            counts[k] -= 1
        while sum(counts.values()) < n_flare:
            k = max([B_id, C_id, M_id], key=lambda c: len(idx_by[c]))
            counts[k] += 1

        # 2) enforce X min/max while keeping exact n_flare
        x_old = counts[X_id]
        x_new = min(x_old, x_max_per_file)
        if len(idx_by[X_id]) > 0:
            x_new = max(x_new, min(x_min_per_file, n_flare))
        counts[X_id] = x_new

        delta = x_new - x_old
        if delta > 0:
            # remove from non-X donors first
            donors = [B_id, C_id, M_id]
            while delta > 0:
                donor = max(donors, key=lambda c: counts[c])
                if counts[donor] <= 0:
                    break
                counts[donor] -= 1
                delta -= 1
            # if still not possible, reduce X back
            if delta > 0:
                counts[X_id] -= delta
        elif delta < 0:
            # distribute freed slots to non-X classes with largest available pools
            freed = -delta
            receivers = [B_id, C_id, M_id]
            for _ in range(freed):
                rcv = max(receivers, key=lambda c: len(idx_by[c]))
                counts[rcv] += 1

        # final exactness guard
        while sum(counts.values()) > n_flare:
            k = max(counts, key=lambda c: counts[c])
            counts[k] -= 1
        while sum(counts.values()) < n_flare:
            k = max([B_id, C_id, M_id], key=lambda c: len(idx_by[c]))
            counts[k] += 1

        # 3) sample flare windows (with replacement if needed)
        for cid, k in counts.items():
            if k <= 0:
                continue
            pool = idx_by[cid]
            if len(pool) == 0:
                continue
            take = np.random.choice(pool, size=k, replace=(len(pool) < k))
            chosen.append(take)

        # 4) hard-negative quiet mining for rest
        quiet_pool = idx_by[quiet_id]
        n_rest_eff = n_rest if len(quiet_pool) > 0 else 0

        if n_rest_eff > 0:
            cand_size = min(len(quiet_pool), max(n_rest_eff, hardquiet_mult * n_rest_eff))
            cand_idx = np.random.choice(quiet_pool, size=cand_size, replace=False)

            X_cand = _fix_windows(X_file[cand_idx]).astype(np.float32, copy=False)
            X_cand_t = torch.as_tensor(X_cand, dtype=torch.float32, device=device)

            prev_mode = model_s1.training
            model_s1.eval()
            with torch.no_grad():
                logits = model_s1(X_cand_t)
                p_flare = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            if prev_mode:
                model_s1.train()

            order = np.argsort(-p_flare)  # descending by hardness

            if hardquiet_mode == "threshold":
                above = np.flatnonzero(p_flare >= float(t_flare_train))
                if above.size >= n_rest_eff:
                    above_order = above[np.argsort(-p_flare[above])]
                    pick_local = above_order[:n_rest_eff]
                else:
                    pick_local = above.tolist()
                    need = n_rest_eff - len(pick_local)
                    remaining = [i for i in order.tolist() if i not in set(pick_local)]
                    pick_local.extend(remaining[:need])
                    pick_local = np.asarray(pick_local, dtype=np.int64)
            else:
                pick_local = order[:n_rest_eff]

            selected_quiet_idx = cand_idx[pick_local]
            selected_quiet_scores = p_flare[pick_local]

            # if still short (rare), fill via replacement from selected
            if selected_quiet_idx.size < n_rest_eff and selected_quiet_idx.size > 0:
                extra_n = n_rest_eff - selected_quiet_idx.size
                extra_pos = np.random.choice(np.arange(selected_quiet_idx.size), size=extra_n, replace=True)
                selected_quiet_idx = np.concatenate([selected_quiet_idx, selected_quiet_idx[extra_pos]])
                selected_quiet_scores = np.concatenate([selected_quiet_scores, selected_quiet_scores[extra_pos]])

            if selected_quiet_idx.size > 0:
                chosen.append(selected_quiet_idx)
                batch_quiet_scores.extend(selected_quiet_scores.tolist())
                used_hardquiet_this_batch = True

        if len(chosen) == 0:
            continue

        chosen_idx = np.concatenate(chosen)
        if chosen_idx.size > 1:
            np.random.shuffle(chosen_idx)

        X_sel = _fix_windows(X_file[chosen_idx]).astype(np.float32, copy=False)
        y_sel = y_np[chosen_idx].astype(np.int64, copy=False)

        X_out.append(torch.as_tensor(X_sel, dtype=torch.float32))
        y_out.append(torch.as_tensor(y_sel, dtype=torch.long))

    if len(X_out) == 0:
        raise ValueError("collate_stage2_quota_fixedT_hardquiet produced empty batch.")

    X_out = torch.cat(X_out, dim=0)
    y_out = torch.cat(y_out, dim=0)

    # lightweight logging counters
    if used_hardquiet_this_batch:
        stats["batches_with_hardquiet"] += 1

    if batch_quiet_scores:
        # keep bounded buffer for mean/median reporting
        buf = stats["quiet_scores_buffer"]
        buf.extend(batch_quiet_scores)
        if len(buf) > 20000:
            stats["quiet_scores_buffer"] = buf[-20000:]

    binc = torch.bincount(y_out, minlength=5).cpu().tolist()
    label_dist = {"quiet": int(binc[0]), "B": int(binc[1]), "C": int(binc[2]), "M": int(binc[3]), "X": int(binc[4])}
    stats["last_batch_label_dist"] = label_dist
    stats["label_dist_history"].append(label_dist)
    if len(stats["label_dist_history"]) > 200:
        stats["label_dist_history"] = stats["label_dist_history"][-200:]

    if log_every and stats["batches"] % log_every == 0:
        qs = np.asarray(stats["quiet_scores_buffer"], dtype=np.float32)
        q_mean = float(qs.mean()) if qs.size else None
        q_med = float(np.median(qs)) if qs.size else None
        print(
            f"[hardquiet] batches={stats['batches']} "
            f"hardquiet_batches={stats['batches_with_hardquiet']} "
            f"quiet_p_mean={q_mean} quiet_p_median={q_med} "
            f"last_label_dist={label_dist}"
        )

    return X_out, y_out

# ============================================================
# Manifest helpers
# ============================================================
def load_manifest_obj(manifest_path: str) -> Dict[str, Any]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)

def iter_manifest_npz_paths(manifest_path: str) -> List[str]:
    m = load_manifest_obj(manifest_path)
    root_dir = m.get("root_dir", "")
    out = []
    for ent in m["entries"]:
        rel = ent["file"]
        p = os.path.normpath(rel)
        if not os.path.isabs(p):
            p = os.path.normpath(os.path.join(root_dir, p))
        out.append(p)
    return out


def load_checkpoint(path: str, model, optimizer=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)

    # support both formats:
    # ckpt["model"] or ckpt["model_state_dict"]
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    elif "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        raise KeyError("Checkpoint missing model state dict keys.")

    if optimizer is not None:
        if "optimizer" in ckpt and ckpt["optimizer"] is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        elif "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt

# ============================================================
# Dataset: file-level
# ============================================================
class NPZFileDataset(Dataset):
    """
    One item = one npz file:
      returns (X_file, y_int_file)
      X_file: (N,T,F)
      y_int_file: (N,)
    Also stores file_label_counts + file_tags (by scan).
    """
    def __init__(self, manifest_path: str, name2id: Dict[str,int],
                 max_files: Optional[int]=None, allow_pickle=True):
        self.manifest_path = manifest_path
        self.name2id = name2id
        self.allow_pickle = allow_pickle

        self.files = iter_manifest_npz_paths(manifest_path)
        if max_files is not None:
            self.files = self.files[:max_files]

        self.file_tags = []          # 'X','M','C','B','quiet_only'
        self.file_label_counts = []  # Counter[str] per file

        for fp in self.files:
            z = np.load(fp, allow_pickle=allow_pickle)
            y = z["y"] if "y" in z.files else z["label"]
            y_str = np.array([str(v) for v in y], dtype=object)
            c = Counter(y_str.tolist())
            self.file_label_counts.append(c)

            if c.get("X", 0) > 0:
                self.file_tags.append("X")
            elif c.get("M", 0) > 0:
                self.file_tags.append("M")
            elif c.get("C", 0) > 0:
                self.file_tags.append("C")
            elif c.get("B", 0) > 0:
                self.file_tags.append("B")
            else:
                self.file_tags.append("quiet_only")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        fp = self.files[idx]
        z = np.load(fp, allow_pickle=self.allow_pickle)
        X = z["X"]
        y = z["y"] if "y" in z.files else z["label"]

        if getattr(y, "dtype", None) == object or isinstance(y[0], (str, np.str_)):
            y_int = np.array([self.name2id[str(v)] for v in y], dtype=np.int64)
        else:
            y_int = y.astype(np.int64, copy=False)
        return X, y_int

def summarize_scan(ds: NPZFileDataset, title: str) -> Counter:
    total = Counter()
    fx=fm=fc=fb=fq=0
    for c, tag in zip(ds.file_label_counts, ds.file_tags):
        total.update(c)
        if tag=="X": fx+=1
        elif tag=="M": fm+=1
        elif tag=="C": fc+=1
        elif tag=="B": fb+=1
        else: fq+=1
    print(f"[{title}] total label counts across scanned files:", dict(total))
    print(f"[{title}] files containing X: {fx}/{len(ds)} | M: {fm}/{len(ds)} | C: {fc}/{len(ds)} | B: {fb}/{len(ds)} | quiet_only: {fq}/{len(ds)}")
    return total

# ============================================================
# Collate (train only): sample per file + fix T + clean
# ============================================================
def collate_take_random_samples_fixedT(
    batch,
    per_file: int = 128,
    T_fixed: Optional[int] = None,
    pad_value: float = 0.0,
    clamp_abs: Optional[float] = 1e3,
):
    Xs, ys = [], []
    if T_fixed is None:
        X0, _ = batch[0]
        T_fixed = int(X0.shape[1])

    for X_file, y_file in batch:
        n = len(y_file)
        if n <= 0:
            continue
        take = min(per_file, n)
        idx = np.random.choice(n, size=take, replace=False)

        X_take = X_file[idx]
        y_take = y_file[idx]

        T_here = X_take.shape[1]
        if T_here > T_fixed:
            X_take = X_take[:, :T_fixed, :]
        elif T_here < T_fixed:
            pad = np.full((take, T_fixed - T_here, X_take.shape[2]), pad_value, dtype=X_take.dtype)
            X_take = np.concatenate([X_take, pad], axis=1)

        X_take = np.nan_to_num(X_take, nan=0.0, posinf=0.0, neginf=0.0)
        if clamp_abs is not None:
            X_take = np.clip(X_take, -clamp_abs, clamp_abs)

        Xs.append(X_take)
        ys.append(y_take)

    X = torch.from_numpy(np.concatenate(Xs, axis=0)).float()
    y = torch.from_numpy(np.concatenate(ys, axis=0)).long()
    return X, y

# ============================================================
# File-level sampler (train)
# ============================================================
def make_file_weighted_sampler(
    ds: NPZFileDataset,
    wX: float = 100.0,
    wM: float = 20.0,
    wC: float = 5.0,
    wB: float = 3.0,
    wQ: float = 1.0,
    num_samples: Optional[int] = None
) -> WeightedRandomSampler:
    tag2w = {"X": wX, "M": wM, "C": wC, "B": wB, "quiet_only": wQ}
    weights = np.array([tag2w.get(t, 1.0) for t in ds.file_tags], dtype=np.float64)
    weights = torch.tensor(weights, dtype=torch.double)
    if num_samples is None:
        num_samples = len(ds)
    return WeightedRandomSampler(weights=weights, num_samples=num_samples, replacement=True)

# ============================================================
# Models
# ============================================================
class SimpleTCN(nn.Module):
    def __init__(self, num_features: int, num_classes: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(num_features, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)          # (B,F,T)
        h = self.net(x).squeeze(-1)    # (B,hidden)
        return self.fc(h)

# ============================================================
# Utils: fixed datasets loaders
# ============================================================
class FixedNPZDataset(Dataset):
    def __init__(self, npz_path: str):
        z = np.load(npz_path, allow_pickle=False)
        self.X = np.asarray(z["X"], dtype=np.float32)
        self.y = np.asarray(z["y"], dtype=np.int64)

    def __len__(self):
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

def make_fixed_loader(npz_path: str, batch_size: int = 1024, num_workers: int = 0) -> DataLoader:
    ds = FixedNPZDataset(npz_path)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

# ============================================================
# ✅ Balanced fixed creator (guarantees per-class minimums)
# ============================================================
def _crop_pad_clean(X_take: np.ndarray, T_fixed: int, pad_value: float, clamp_abs: Optional[float]):
    take = X_take.shape[0]
    T_here = int(X_take.shape[1])
    if T_here > T_fixed:
        X_take = X_take[:, :T_fixed, :]
    elif T_here < T_fixed:
        pad = np.full((take, T_fixed - T_here, X_take.shape[2]), pad_value, dtype=X_take.dtype)
        X_take = np.concatenate([X_take, pad], axis=1)
    X_take = np.nan_to_num(X_take, nan=0.0, posinf=0.0, neginf=0.0)
    if clamp_abs is not None:
        X_take = np.clip(X_take, -clamp_abs, clamp_abs)
    return X_take

def make_fixed_balanced(
    ds: NPZFileDataset,
    out_path: str,
    T_fixed: int,
    per_class_min: Dict[str, int],
    seed: int = 123,
    clamp_abs: Optional[float] = 1e3,
    pad_value: float = 0.0,
    max_files_per_class: Optional[int] = None,
) -> str:
    """
    Creates fixed NPZ with guaranteed minimum samples per class (if available in ds).
    Does NOT modify originals. Writes out_path only.
    per_class_min keys: "X","M","C","B","quiet"
    """
    rng = np.random.default_rng(seed)

    # Pre-index candidate files per class using scan results
    idx_by = {k: [] for k in ["X","M","C","B","quiet"]}
    for i, c in enumerate(ds.file_label_counts):
        if c.get("X", 0) > 0: idx_by["X"].append(i)
        if c.get("M", 0) > 0: idx_by["M"].append(i)
        if c.get("C", 0) > 0: idx_by["C"].append(i)
        if c.get("B", 0) > 0: idx_by["B"].append(i)
        # quiet candidates: any file that has quiet (usually almost all), but prioritize quiet_only later
        if c.get("quiet", 0) > 0: idx_by["quiet"].append(i)

    X_list, y_list = [], []

    # Order matters: X -> M -> C -> B -> quiet
    order = ["X","M","C","B","quiet"]

    for cls in order:
        need = int(per_class_min.get(cls, 0))
        if need <= 0:
            continue

        cand = idx_by[cls]
        if len(cand) == 0:
            print(f"[make_fixed_balanced] WARNING: no candidate files for class={cls}")
            continue

        if max_files_per_class is not None:
            cand = list(cand)
            rng.shuffle(cand)
            cand = cand[:max_files_per_class]

        # Shuffle candidates so fixed is deterministic but not biased to first files
        cand = np.array(cand, dtype=np.int64)
        rng.shuffle(cand)

        got = 0
        for fi in cand:
            if got >= need:
                break

            X_file, y_file = ds[int(fi)]
            y_file = np.asarray(y_file, dtype=np.int64)

            if cls == "quiet":
                mask = (y_file == IDX_Q)
            else:
                mask = (y_file == name2id[cls])

            idxs = np.where(mask)[0]
            if idxs.size == 0:
                continue

            take = min(need - got, idxs.size)
            choose = rng.choice(idxs, size=take, replace=False)

            X_take = X_file[choose]
            y_take = y_file[choose]

            X_take = _crop_pad_clean(X_take, T_fixed=T_fixed, pad_value=pad_value, clamp_abs=clamp_abs)

            X_list.append(X_take.astype(np.float32, copy=False))
            y_list.append(y_take.astype(np.int64, copy=False))
            got += int(take)

        if got < need:
            print(f"[make_fixed_balanced] WARNING: class={cls} requested {need} but collected {got} (limited by data)")

    if not X_list:
        raise ValueError("make_fixed_balanced produced 0 samples. Check dataset / per_class_min.")

    X = np.concatenate(X_list, axis=0).astype(np.float32, copy=False)
    y = np.concatenate(y_list, axis=0).astype(np.int64, copy=False)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, X=X, y=y, T_fixed=int(T_fixed), seed=int(seed), per_class_min=json.dumps(per_class_min))
    return out_path

# ============================================================
# Metrics / saving
# ============================================================
def per_class_recall(cm: np.ndarray) -> Dict[str, float]:
    rec = {}
    for i in range(K):
        support = cm[i, :].sum()
        tp = cm[i, i]
        rec[classes[i]] = float(tp / support) if support > 0 else 0.0
    return rec


def compute_alpha_from_counts(
    counts: Counter,
    classes_subset: List[str],
    mode: str = "sqrt_inv",
    power: float = 0.5,
) -> np.ndarray:
    arr = np.array([float(counts.get(c, 0)) for c in classes_subset], dtype=np.float64)
    arr[arr == 0] = 1.0

    if mode == "balanced":
        w = arr.sum() / (len(arr) * arr)
    elif mode == "inv":
        w = 1.0 / arr
    elif mode == "pow_inv":
        w = 1.0 / np.power(arr, power)
    elif mode == "sqrt_inv":
        w = 1.0 / np.sqrt(arr)
    else:
        raise ValueError(f"unknown mode: {mode}")

    w = w * (len(arr) / w.sum())
    return w.astype(np.float32)

def save_checkpoint(out_dir: str, tag: str, model: nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    extra: Optional[Dict[str, Any]] = None) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{tag}.pt")
    payload = {"model_state": model.state_dict(), "extra": extra or {}}
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)
    return path

def evaluate_and_save(out_dir: str, split_name: str, y_true: np.ndarray, probs: np.ndarray,
                      tag: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)

    y_true = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float32)
    pred = probs.argmax(axis=1).astype(np.int64)

    cm = confusion_matrix(y_true, pred, labels=list(range(K)))
    mf1 = f1_score(y_true, pred, average="macro", labels=list(range(K)))

    rep = classification_report(
        y_true, pred,
        labels=list(range(K)),
        target_names=classes,
        zero_division=0,
        output_dict=True
    )

    metrics = {
        "split": split_name,
        "tag": tag,
        "macro_f1": float(mf1),
        "recall_per_class": per_class_recall(cm),
        "confusion_matrix": cm.tolist(),
        "classification_report": rep,
        "dist": dict(Counter(y_true.tolist())),
        "extra": extra or {},
    }

    np.save(os.path.join(out_dir, f"{split_name}_{tag}_probs.npy"), probs)
    np.save(os.path.join(out_dir, f"{split_name}_{tag}_y_true.npy"), y_true)
    np.save(os.path.join(out_dir, f"{split_name}_{tag}_pred.npy"), pred)

    with open(os.path.join(out_dir, f"{split_name}_{tag}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics

# ============================================================
# Train / predict (generic)
# ============================================================
@torch.no_grad()
def predict_proba(model, loader, device="cuda") -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs_all, y_all = [], []
    for X, y in loader:
        X = X.to(device, non_blocking=True)
        logits = model(X)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        probs_all.append(probs)
        y_all.append(y.numpy())
    return np.concatenate(probs_all, axis=0), np.concatenate(y_all, axis=0)

def train_one_epoch(model, loader, optimizer, loss_fn, device="cuda"):
    model.train()
    total_loss, nseen = 0.0, 0
    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(X)
        loss = loss_fn(logits, y)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * X.size(0)
        nseen += X.size(0)
    return total_loss / max(nseen, 1)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        logp = torch.log_softmax(logits, dim=1)
        p = torch.exp(logp)
        pt = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = -(1 - pt) ** self.gamma * logp.gather(1, targets.unsqueeze(1)).squeeze(1)
        if self.alpha is not None:
            a = self.alpha.to(logits.device).gather(0, targets)
            loss = a * loss
        return loss.mean()
    
class FocalLossBinary(nn.Module):
    def __init__(self, gamma=2.0, pos_weight: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        # logits: (N,2), targets: (N,)
        logp = torch.log_softmax(logits, dim=1)
        p = torch.exp(logp)
        pt = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = -(1 - pt) ** self.gamma * logp.gather(1, targets.unsqueeze(1)).squeeze(1)

        # apply extra weight to positive class
        w = torch.ones_like(loss)
        w[targets == 1] = float(self.pos_weight)
        loss = w * loss
        return loss.mean()

class FocalLossMulticlass(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # tensor shape (K,)

    def forward(self, logits, targets):
        logp = torch.log_softmax(logits, dim=1)
        p = torch.exp(logp)
        pt = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = -(1 - pt) ** self.gamma * logp.gather(1, targets.unsqueeze(1)).squeeze(1)
        if self.alpha is not None:
            a = self.alpha.to(logits.device).gather(0, targets)
            loss = a * loss
        return loss.mean()
    
# ============================================================
# ============================================================
# Two-stage predict + thresholds
# ============================================================
SEV_CLASSES = ["B","C","M","X"]

def predict_severity_with_thresholds(p_sev: np.ndarray, t_sev: Optional[np.ndarray] = None) -> np.ndarray:
    """
    p_sev: (N,4) probs over [B,C,M,X]
    t_sev: (4,) thresholds for [B,C,M,X] (OVR-style). If None -> argmax.
    returns: (N,) in 0..3
    """
    if t_sev is None:
        return p_sev.argmax(axis=1)
    t = np.asarray(t_sev, dtype=np.float32)
    passed = p_sev >= t[None, :]
    pred = np.empty(p_sev.shape[0], dtype=np.int64)
    for i in range(p_sev.shape[0]):
        idx = np.flatnonzero(passed[i])
        if idx.size == 0:
            pred[i] = int(np.argmax(p_sev[i]))
        else:
            pred[i] = int(idx[np.argmax(p_sev[i, idx])])
    return pred


def two_stage_predict(p_flare: np.ndarray, p_sev: np.ndarray, t_flare: float, t_sev: Optional[np.ndarray] = None) -> np.ndarray:
    """
    p_flare: (N,) prob flare=1
    p_sev:   (N,4) probs over [B,C,M,X] for ALL N
    t_sev:   (4,) optional per-class thresholds for [B,C,M,X]
    """
    N = p_flare.shape[0]
    pred = np.full(N, IDX_Q, dtype=np.int64)
    flare_idx = np.where(p_flare >= t_flare)[0]
    if flare_idx.size > 0:
        sev_pred = predict_severity_with_thresholds(p_sev[flare_idx], t_sev=t_sev)
        pred[flare_idx] = sev_pred + 1  # 0..3 -> 1..4
    return pred

# ============================================================
# Gate tuning (macro-F1 target)
# ============================================================
def score_gate(
    y_true: np.ndarray,
    pred: np.ndarray,
    objective: str = "macro_f1",
    recall_weight: float = 0.0,
    recall_classes: Tuple[str, ...] = ("B","C","M","X"),
) -> Tuple[float, Dict[str, Any]]:
    cm = confusion_matrix(y_true, pred, labels=list(range(K)))
    rec = per_class_recall(cm)
    mf1 = f1_score(y_true, pred, average="macro", labels=list(range(K)))

    if objective == "macro_f1":
        mean_rec = float(np.mean([rec.get(c, 0.0) for c in recall_classes])) if recall_classes else 0.0
        obj = float(mf1)
    elif objective == "macro_f1_plus_recall":
        mean_rec = float(np.mean([rec.get(c, 0.0) for c in recall_classes])) if recall_classes else 0.0
        obj = float(mf1 + recall_weight * mean_rec)
    else:
        raise ValueError(f"Unknown objective: {objective}")

    details = {
        "macro_f1": float(mf1),
        "mean_recall_focus": float(mean_rec),
        "recall_focus_classes": list(recall_classes),
        "recall_per_class": {k: float(v) for k, v in rec.items()},
    }
    return obj, details


def tune_t_flare(
    p_flare: np.ndarray,
    p_sev: np.ndarray,
    y_true: np.ndarray,
    grid=(0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5),
    lam: Optional[float] = None,
    objective: str = "macro_f1",
    recall_weight: float = 0.0,
    recall_classes: Tuple[str, ...] = ("B","C","M","X"),
) -> Tuple[float, Dict[str, Any]]:
    # Backward compat: if lam is provided, treat it as recall_weight
    if lam is not None:
        recall_weight = float(lam)
        if objective == "macro_f1":
            objective = "macro_f1_plus_recall"

    best = (-1e9, None, None)
    for t in grid:
        pred = two_stage_predict(p_flare, p_sev, t_flare=float(t))
        obj, info = score_gate(y_true, pred, objective=objective, recall_weight=recall_weight, recall_classes=recall_classes)
        if obj > best[0]:
            best = (obj, float(t), info)

    return best[1], {
        "objective": float(best[0]),
        "details": best[2],
        "grid": list(grid),
        "objective_name": objective,
        "recall_weight": float(recall_weight),
        "recall_focus_classes": list(recall_classes),
    }

def tune_t_flare_quiet(p_flare, p_sev, y_true, grid, objective, recall_weight, recall_classes):
    t_sev_dummy = [0.0, 0.0, 0.0, 0.0]
    best = (-1e9, None, None)
    for t in grid:
        pred = two_stage_predict_with_quiet(p_flare, p_sev, t_flare=float(t), t_sev=t_sev_dummy)
        obj, info = score_gate(y_true, pred, objective, recall_weight, recall_classes)
        if obj > best[0]:
            best = (obj, float(t), info)
    return best[1], {
        "objective": float(best[0]),
        "details": best[2],
        "grid": list(grid),
        "objective_name": objective,
        "recall_weight": float(recall_weight),
        "recall_focus_classes": list(recall_classes),
    }
# ============================================================
# Severity thresholds (OVR-style) to boost macro-F1
# ============================================================
def tune_sev_thresholds(
    p_flare: np.ndarray,
    p_sev: np.ndarray,
    y_true: np.ndarray,
    t_flare: float,
    grid=(0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.8),
    iters: int = 2,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    t = np.full(4, 0.5, dtype=np.float32)
    best_mf1 = -1.0

    for _ in range(int(iters)):
        for c in range(4):
            best_c = -1.0
            best_t = float(t[c])
            for cand in grid:
                t_try = t.copy()
                t_try[c] = float(cand)
                pred = two_stage_predict(p_flare, p_sev, t_flare=t_flare, t_sev=t_try)
                mf1 = f1_score(y_true, pred, average="macro", labels=list(range(K)))
                if mf1 > best_c:
                    best_c = float(mf1)
                    best_t = float(cand)
            t[c] = best_t
            best_mf1 = max(best_mf1, best_c)

    pred = two_stage_predict(p_flare, p_sev, t_flare=t_flare, t_sev=t)
    mf1 = f1_score(y_true, pred, average="macro", labels=list(range(K)))
    cm = confusion_matrix(y_true, pred, labels=list(range(K)))
    info = {
        "macro_f1": float(mf1),
        "recall_per_class": per_class_recall(cm),
        "grid": list(grid),
        "iters": int(iters),
    }
    return t, info


def stage1_pass_rate_by_class(p_flare, y_true, t_flare):
    for cls_id, cls_name in enumerate(classes):
        m = (y_true == cls_id)
        if m.sum() == 0:
            continue
        pass_rate = (p_flare[m] >= t_flare).mean()
        print(f"{cls_name}: pass {pass_rate:.3f} ({(p_flare[m] >= t_flare).sum()}/{m.sum()})")



def pick_t_flare_gate(
    p_flare, y_true, thresholds,
    min_flare_pass=0.94, min_mx_pass=0.98, target_quiet_pass=0.25
):
    best = None
    rows = []

    q = (y_true == IDX_Q)
    flare = (y_true != IDX_Q)
    mx = np.isin(y_true, [IDX_M, IDX_X])
    c_mask = (y_true == IDX_C)

    for t in thresholds:
        passed = (p_flare >= t)
        quiet_pass = passed[q].mean() if q.any() else 0.0
        flare_pass = passed[flare].mean() if flare.any() else 0.0
        mx_pass = passed[mx].mean() if mx.any() else 0.0
        c_pass = passed[c_mask].mean() if c_mask.any() else 0.0

        feasible = (flare_pass >= min_flare_pass) and (mx_pass >= min_mx_pass)
        score = (1.0 - quiet_pass) + 0.25 * flare_pass + 0.10 * c_pass
        rows.append((t, quiet_pass, flare_pass, mx_pass, c_pass, feasible, score))

        if feasible and (best is None or score > best[-1]):
            best = (t, quiet_pass, flare_pass, mx_pass, c_pass, feasible, score)

    if best is None:
        # fallback: maximize flare while penalizing quiet pass
        best = max(rows, key=lambda r: (r[2] - 0.7 * r[1]))

    return best, rows

def choose_t_for_m_goal(
    p_flare: np.ndarray,
    y_true: np.ndarray,
    thresholds,
    m_goal: float = 0.90,
    x_goal: float = 0.98,
    quiet_cap: float = 0.30,
):
    rows = []
    for t in thresholds:
        t = float(t)
        passed = (p_flare >= t)

        q = float(passed[y_true == IDX_Q].mean()) if np.any(y_true == IDX_Q) else 0.0
        b = float(passed[y_true == IDX_B].mean()) if np.any(y_true == IDX_B) else 0.0
        c = float(passed[y_true == IDX_C].mean()) if np.any(y_true == IDX_C) else 0.0
        m = float(passed[y_true == IDX_M].mean()) if np.any(y_true == IDX_M) else 0.0
        x = float(passed[y_true == IDX_X].mean()) if np.any(y_true == IDX_X) else 0.0

        feasible = (m >= m_goal) and (x >= x_goal) and (q <= quiet_cap)
        rows.append({
            "t": t, "quiet_pass": q, "B_pass": b, "C_pass": c, "M_pass": m, "X_pass": x, "feasible": feasible
        })

    feasible_rows = [r for r in rows if r["feasible"]]
    if feasible_rows:
        # prioritize M, then lower quiet, then X, then C
        best = max(feasible_rows, key=lambda r: (r["M_pass"], -r["quiet_pass"], r["X_pass"], r["C_pass"]))
    else:
        # fallback: still prioritize M strongly
        best = max(rows, key=lambda r: (2.0 * r["M_pass"] + 0.5 * r["X_pass"] - 0.8 * r["quiet_pass"]))

    return best, rows

# Config + runner
# ============================================================
@dataclass
class ExperimentCfg:
    root4: str
    W: int
    H: int
    out_root: str

    # limits
    max_train_files: Optional[int] = None
    max_val_files: Optional[int] = None
    max_test_files: Optional[int] = None

    # sampling (file-batch -> sample-batch)
    batch_files: int = 8
    per_file: int = 256
    clamp_abs: float = 1e3

    # Stage1 sampler weights (file-level)
    wX: float = 100.0
    wM: float = 20.0
    wC: float = 5.0
    wB: float = 3.0
    wQ: float = 1.0

    # ✅ Stage2 sampler weights (separate)
    s2_wX: float = 600.0
    s2_wM: float = 120.0
    s2_wC: float = 8.0
    s2_wB: float = 4.0
    s2_wQ: float = 1.0

    # model/train
    lr: float = 1e-3
    hidden: int = 64
    seed: int = 123

    # fixed balanced sets
    fixed_seed: int = 123
    fixed_batch_size: int = 512
    fixed_min_val: Optional[Dict[str, int]] = None
    fixed_min_test: Optional[Dict[str, int]] = None

    # Stage1 (binary)
    s1_epochs: int = 12
    s1_pos_weight: float = 20.0
    s1_use_focal: bool = False
    s1_focal_gamma: float = 2.0

    # Stage2 (severity)
    s2_epochs: int = 10
    s2_use_focal: bool = True
    s2_focal_gamma: float = 2.0
    s2_alpha: Tuple[float, float, float, float] = (0.5, 1.0, 10.0, 30.0)  # [B,C,M,X]
    s2_alpha_mode: str = "pow_inv"
    s2_alpha_pow: float = 0.5

    # Stage2 quota (per-file sampling)
    s2_flare_frac: float = 0.85
    s2_flare_quota: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)  # B,C,M,X
    s2_x_min_per_file: int = 1
    s2_x_max_per_file: int = 12

    # threshold tuning for Stage1 gate
    t_flare_grid: Tuple[float, ...] = (0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3)
    t_flare_lam: float = 3.0

    # Gate selection (macro-F1 target)
    gate_mode: str = "grid_macro_f1"  # grid_macro_f1 | quiet_quantile | no_gate
    gate_quantile: float = 0.95
    gate_objective: str = "macro_f1"  # macro_f1 | macro_f1_plus_recall
    gate_recall_weight: float = 1.0
    gate_recall_classes: Tuple[str, ...] = ("B","C","M","X")

    # Per-class thresholds for Stage2 (OVR-style)
    use_sev_thresholds: bool = True
    sev_t_grid: Tuple[float, ...] = (0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.8)
    sev_t_iters: int = 2

@dataclass
class ExperimentCfgSkipStage1:
    root4: str
    W: int
    H: int
    out_root: str
    out_dir_changes: bool = False

    # limits
    max_train_files: Optional[int] = None
    max_val_files: Optional[int] = None
    max_test_files: Optional[int] = None

    # sampling (file-batch -> sample-batch)
    batch_files: int = 8
    per_file: int = 256
    clamp_abs: float = 1e3

    # Stage1 sampler weights (file-level)
    wX: float = 100.0
    wM: float = 20.0
    wC: float = 5.0
    wB: float = 3.0
    wQ: float = 1.0

    # ✅ Stage2 sampler weights (separate)
    s2_wX: float = 600.0
    s2_wM: float = 120.0
    s2_wC: float = 8.0
    s2_wB: float = 4.0
    s2_wQ: float = 1.0

    # model/train
    lr: float = 1e-3
    hidden: int = 64
    seed: int = 123

    # fixed balanced sets
    fixed_seed: int = 123
    fixed_batch_size: int = 512
    fixed_min_val: Optional[Dict[str, int]] = None
    fixed_min_test: Optional[Dict[str, int]] = None

    # Stage1 (binary)
    s1_epochs: int = 12
    s1_pos_weight: float = 20.0
    s1_use_focal: bool = False
    s1_focal_gamma: float = 2.0

    # Stage2 (severity)
    s2_epochs: int = 10
    s2_use_focal: bool = True
    s2_focal_gamma: float = 2.0
    s2_alpha: Tuple[float, float, float, float] = (0.5, 1.0, 10.0, 30.0)  # [B,C,M,X]
    s2_alpha_mode: str = "pow_inv"
    s2_alpha_pow: float = 0.5

    # Stage2 quota (per-file sampling)
    s2_flare_frac: float = 0.85
    s2_flare_quota: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)  # B,C,M,X
    s2_x_min_per_file: int = 1
    s2_x_max_per_file: int = 12

    # threshold tuning for Stage1 gate
    t_flare_grid: Tuple[float, ...] = (0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3)
    t_flare_lam: float = 3.0

    # Gate selection (macro-F1 target)
    gate_mode: str = "grid_macro_f1"  # grid_macro_f1 | quiet_quantile | no_gate
    gate_quantile: float = 0.95
    gate_objective: str = "macro_f1"  # macro_f1 | macro_f1_plus_recall
    gate_recall_weight: float = 1.0
    gate_recall_classes: Tuple[str, ...] = ("B","C","M","X")

    # Per-class thresholds for Stage2 (OVR-style)
    use_sev_thresholds: bool = True
    sev_t_grid: Tuple[float, ...] = (0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.8)
    sev_t_iters: int = 2

    s1_resume_path: Optional[str] = None   # path to stage1_last.pt
    skip_stage1_train: bool = False        # if True: don't train stage1, must load

    quiet_w: float = 0.2
    manual_t_sev: bool = False

def load_and_eval_stage1(
    ckpt_path: str,
    split: str = "val",          # "val" or "test"
    use_balanced: bool = True,   # True: val_fixed/test_fixed, False: natural ds_val/ds_test
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device)

    # rebuild cfg from checkpoint
    cfg = ExperimentCfg(**ckpt["extra"]["cfg"])

    # manifests
    man_dir = os.path.join(cfg.root4, "manifests_by_group")
    man_train = os.path.join(man_dir, f"manifest_train_W{cfg.W}h_H{cfg.H}h.json")
    man_val   = os.path.join(man_dir, f"manifest_val_W{cfg.W}h_H{cfg.H}h.json")
    man_test  = os.path.join(man_dir, f"manifest_test_W{cfg.W}h_H{cfg.H}h.json")

    # train dataset just to get F and T_fixed
    ds_train = NPZFileDataset(man_train, name2id, max_files=cfg.max_train_files, allow_pickle=True)
    X0, _ = ds_train[0]
    T_fixed = int(X0.shape[1])
    F = int(X0.shape[2])

    # build loader
    if split == "val":
        if use_balanced:
            val_fixed = os.path.join(cfg.out_root, f"W{cfg.W}_H{cfg.H}_two_stage_bal", f"val_fixed_bal_W{cfg.W}_H{cfg.H}.npz")
            dl = make_fixed_loader(val_fixed, batch_size=cfg.fixed_batch_size, num_workers=0)
        else:
            ds_val = NPZFileDataset(man_val, name2id, max_files=cfg.max_val_files, allow_pickle=True)
            dl = DataLoader(
                ds_val,
                batch_size=cfg.batch_files,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                collate_fn=lambda b: collate_take_random_samples_fixedT(
                    b, per_file=cfg.per_file, T_fixed=T_fixed, clamp_abs=cfg.clamp_abs
                ),
            )
    elif split == "test":
        if not os.path.exists(man_test):
            print("No test manifest found.")
            return None
        if use_balanced:
            test_fixed = os.path.join(cfg.out_root, f"W{cfg.W}_H{cfg.H}_two_stage_bal", f"test_fixed_bal_W{cfg.W}_H{cfg.H}.npz")
            dl = make_fixed_loader(test_fixed, batch_size=cfg.fixed_batch_size, num_workers=0)
        else:
            ds_test = NPZFileDataset(man_test, name2id, max_files=cfg.max_test_files, allow_pickle=True)
            dl = DataLoader(
                ds_test,
                batch_size=cfg.batch_files,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                collate_fn=lambda b: collate_take_random_samples_fixedT(
                    b, per_file=cfg.per_file, T_fixed=T_fixed, clamp_abs=cfg.clamp_abs
                ),
            )
    else:
        raise ValueError("split must be 'val' or 'test'")

    # build model
    model_s1 = SimpleTCN(num_features=F, num_classes=2, hidden=cfg.hidden).to(device)
    model_s1.load_state_dict(ckpt["model_state"])
    model_s1.eval()

    # predict
    probs1, y_true = predict_proba(model_s1, dl, device=device)
    p_flare = probs1[:, 1]
    y_bin = (y_true != IDX_Q).astype(np.int64)
    pred_bin = (p_flare >= 0.5).astype(np.int64)

    cm = confusion_matrix(y_bin, pred_bin, labels=[0,1])
    f1b = f1_score(y_bin, pred_bin, average="binary", zero_division=0)
    prec = precision_score(y_bin, pred_bin, zero_division=0)
    rec  = recall_score(y_bin, pred_bin, zero_division=0)

    print(f"\n=== Stage1 {split.upper()} (quiet vs flare) ===")
    print("F1:", f1b, "Precision:", prec, "Recall:", rec)
    print("Confusion matrix [[TN,FP],[FN,TP]]:\n", cm)
    print(classification_report(y_bin, pred_bin, target_names=["quiet","flare"], zero_division=0))

    return {
        "f1": float(f1b),
        "precision": float(prec),
        "recall": float(rec),
        "confusion_matrix": cm.tolist(),
    }

def load_and_eval(cfg: ExperimentCfg):
    # --- SET THIS ---
    OUT_DIR = r"runs_pytorch_stage2_fix\W24_H24_two_stage_bal"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load checkpoints ---
    ckpt1 = torch.load(os.path.join(OUT_DIR, "stage1_last.pt"), map_location=device)
    ckpt2 = torch.load(os.path.join(OUT_DIR, "stage2_last.pt"), map_location=device)

    # Rebuild cfg from checkpoint
    cfg = ExperimentCfg(**ckpt1["extra"]["cfg"])

    # Manifests
    man_dir = os.path.join(cfg.root4, "manifests_by_group")
    man_train = os.path.join(man_dir, f"manifest_train_W{cfg.W}h_H{cfg.H}h.json")
    man_val   = os.path.join(man_dir, f"manifest_val_W{cfg.W}h_H{cfg.H}h.json")
    man_test  = os.path.join(man_dir, f"manifest_test_W{cfg.W}h_H{cfg.H}h.json")

    # Datasets
    ds_train = NPZFileDataset(man_train, name2id, max_files=cfg.max_train_files, allow_pickle=True)
    ds_val   = NPZFileDataset(man_val,   name2id, max_files=cfg.max_val_files, allow_pickle=True)

    X0, _ = ds_train[0]
    T_fixed = int(X0.shape[1])
    F = int(X0.shape[2])

    # fixed balanced sets (use existing files)
    val_fixed = os.path.join(OUT_DIR, f"val_fixed_bal_W{cfg.W}_H{cfg.H}.npz")
    test_fixed = os.path.join(OUT_DIR, f"test_fixed_bal_W{cfg.W}_H{cfg.H}.npz")

    dl_val_fixed = make_fixed_loader(val_fixed, batch_size=cfg.fixed_batch_size, num_workers=0)

    dl_test_fixed = None
    if os.path.exists(man_test) and os.path.exists(test_fixed):
        dl_test_fixed = make_fixed_loader(test_fixed, batch_size=cfg.fixed_batch_size, num_workers=0)

    # --- Rebuild models (match your original sizes) ---
    model_s1 = SimpleTCN(num_features=F, num_classes=2, hidden=cfg.hidden).to(device)
    model_s1.load_state_dict(ckpt1["model_state"])
    model_s1.eval()

    # Stage2 num_classes: set to 4 or 5 based on your run
    # If your Stage2 includes quiet, set to 5
    S2_NUM_CLASSES = 5  # change to 5 if you trained Stage2 with quiet
    model_s2 = SimpleTCN(num_features=F, num_classes=S2_NUM_CLASSES, hidden=cfg.hidden).to(device)
    model_s2.load_state_dict(ckpt2["model_state"])
    model_s2.eval()

    # --- Predict on VAL ---
    probs1_va, y_va = predict_proba(model_s1, dl_val_fixed, device=device)
    p_flare_va = probs1_va[:, 1]
    probs2_va, _ = predict_proba(model_s2, dl_val_fixed, device=device)

    # Choose gate (same as before)
    t_flare, gate_info = tune_t_flare(
        p_flare_va, probs2_va, y_va,
        grid=cfg.t_flare_grid,
        objective=cfg.gate_objective,
        recall_weight=cfg.gate_recall_weight,
        recall_classes=cfg.gate_recall_classes,
    )
    gate_info["method"] = "grid_tuned"

    # Tune thresholds (use your existing function)
    t_sev = None
    sev_info = None
    if cfg.use_sev_thresholds:
        best = tune_sev_thresholds_with_constraints(
            p_flare_va, probs2_va, y_va, t_flare,
            grid=cfg.sev_t_grid,
            min_prec_B=0.05,
            min_rec_M=0.05,
            min_rec_X=0.05
        )
        t_sev = best["t_sev"]
        sev_info = best

    # Fallback if constraints fail
    #if t_sev is None:
    #    print(f"t_sev is none")
    #    t_sev = [0.2, 0.2, 0.2, 0.2]
    #    sev_info = {"fallback": "default_t_sev"}

    pred_va = two_stage_predict_with_quiet(
    p_flare_va, probs2_va, t_flare=t_flare, t_sev=t_sev
    )

    mf1_va = f1_score(y_va, pred_va, average="macro", labels=list(range(K)))
    cm_va = confusion_matrix(y_va, pred_va, labels=list(range(K)))


    val_metrics = {
    "split": "val_fixed_bal",
    "macro_f1_two_stage": float(mf1_va),
    "recall_per_class_two_stage": per_class_recall(cm_va),
    "confusion_matrix_two_stage": cm_va.tolist(),
    "thresholds": {
        "two_stage": {
            "t_flare": float(t_flare),
            "t_sev": [float(x) for x in t_sev] if t_sev is not None else None,
            "sev_info": sev_info,   # ✅ save it here
        }
    },
    "dist": dict(Counter(y_va.tolist())),
    }
    with open(os.path.join(OUT_DIR, "val_fixed_bal_metrics_two_stage_load_and_eval.json"), "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=2)

    print("\n=== VAL_FIXED_BAL (two-stage) ===")
    print(classification_report(y_va, pred_va, labels=list(range(K)), target_names=classes, zero_division=0))

    # --- Predict on TEST ---
    if dl_test_fixed is not None:
        probs1_te, y_te = predict_proba(model_s1, dl_test_fixed, device=device)
        p_flare_te = probs1_te[:, 1]
        probs2_te, _ = predict_proba(model_s2, dl_test_fixed, device=device)

        pred_te = two_stage_predict_with_quiet(
            p_flare_te, probs2_te, t_flare=t_flare, t_sev=t_sev
        )

        mf1_te = f1_score(y_te, pred_te, average="macro", labels=list(range(K)))
        cm_te = confusion_matrix(y_te, pred_te, labels=list(range(K)))

        test_metrics = {
            "split": "test_fixed_bal",
            "macro_f1_two_stage": float(mf1_te),
            "recall_per_class_two_stage": per_class_recall(cm_te),
            "confusion_matrix_two_stage": cm_te.tolist(),
            "thresholds": {
                "two_stage": {
                    "t_flare": float(t_flare),
                    "t_sev": [float(x) for x in t_sev] if t_sev is not None else None,
                    "sev_info": sev_info,   # ✅ save it here too
                }
            },
            "dist": dict(Counter(y_te.tolist())),
        }
        with open(os.path.join(OUT_DIR, "test_fixed_bal_metrics_two_stage_load_and_eval.json"), "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2)

        print("\n=== TEST_FIXED_BAL (two-stage) ===")
        print(classification_report(y_te, pred_te, labels=list(range(K)), target_names=classes, zero_division=0))

def run_one_experiment(cfg: ExperimentCfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    set_seed(cfg.seed)

    man_dir = os.path.join(cfg.root4, "manifests_by_group")
    man_train = os.path.join(man_dir, f"manifest_train_W{cfg.W}h_H{cfg.H}h.json")
    man_val   = os.path.join(man_dir, f"manifest_val_W{cfg.W}h_H{cfg.H}h.json")
    man_test  = os.path.join(man_dir, f"manifest_test_W{cfg.W}h_H{cfg.H}h.json")

    out_dir = os.path.join(cfg.out_root, f"W{cfg.W}_H{cfg.H}_two_stage_bal")
    os.makedirs(out_dir, exist_ok=True)

    # datasets
    ds_train = NPZFileDataset(man_train, name2id, max_files=cfg.max_train_files, allow_pickle=True)
    ds_val   = NPZFileDataset(man_val,   name2id, max_files=cfg.max_val_files, allow_pickle=True)
    print("train files:", len(ds_train), "| val files:", len(ds_val))

    train_counts = summarize_scan(ds_train, "scan-train")
    _ = summarize_scan(ds_val,   "scan-val")

    X0, _ = ds_train[0]
    T_fixed = int(X0.shape[1])
    F = int(X0.shape[2])
    print("T_fixed:", T_fixed, "| F:", F)

    # -------------------------
    # Train loader (Stage1)
    # -------------------------
    sampler_s1 = make_file_weighted_sampler(
        ds_train, wX=cfg.wX, wM=cfg.wM, wC=cfg.wC, wB=cfg.wB, wQ=cfg.wQ
    )
    dl_train_s1 = DataLoader(
        ds_train,
        batch_size=cfg.batch_files,
        sampler=sampler_s1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=lambda b: collate_take_random_samples_fixedT(
            b, per_file=cfg.per_file, T_fixed=T_fixed, clamp_abs=cfg.clamp_abs
        ),
    )

    # -------------------------
    # Stage2 loader (same ds_train, different sampler)
    # -------------------------
    sampler_s2 = make_file_weighted_sampler(
        ds_train, wX=cfg.s2_wX, wM=cfg.s2_wM, wC=cfg.s2_wC, wB=cfg.s2_wB, wQ=cfg.s2_wQ
    )
    dl_train_s2 = DataLoader(
        ds_train,
        batch_size=cfg.batch_files,
        sampler=sampler_s2,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        #collate_fn=lambda b: collate_take_random_samples_fixedT(
        #    b, per_file=cfg.per_file, T_fixed=T_fixed, clamp_abs=cfg.clamp_abs
        collate_fn=lambda b: collate_stage2_quota_fixedT(
        b,
        per_file=cfg.per_file,
        T_fixed=T_fixed,
        clamp_abs=cfg.clamp_abs,
        flare_frac=cfg.s2_flare_frac,
        flare_quota=cfg.s2_flare_quota,
        x_min_per_file=cfg.s2_x_min_per_file,
        x_max_per_file=cfg.s2_x_max_per_file,
        quiet_id=name2id["quiet"],
        B_id=name2id["B"],
        C_id=name2id["C"],
        M_id=name2id["M"],
        X_id=name2id["X"],),
        #flare_frac=0.80,
        #flare_quota={
        #    name2id["B"]: 0.25,
        #    name2id["C"]: 0.45,
        #    name2id["M"]: 0.20,
        #    name2id["X"]: 0.10,
        #},
        #x_max_per_file=4,
        
    )

    # -------- fixed balanced VAL/TEST --------
    if cfg.fixed_min_val is None:
        cfg.fixed_min_val = {"X":200,"M":200,"C":800,"B":200,"quiet":8000}
    if cfg.fixed_min_test is None:
        cfg.fixed_min_test = {"X":200,"M":200,"C":800,"B":200,"quiet":12000}

    val_fixed = os.path.join(out_dir, f"val_fixed_bal_W{cfg.W}_H{cfg.H}.npz")
    if not os.path.exists(val_fixed):
        print("Creating balanced val_fixed:", val_fixed)
        make_fixed_balanced(
            ds_val, val_fixed,
            T_fixed=T_fixed,
            per_class_min=cfg.fixed_min_val,
            seed=cfg.fixed_seed,
            clamp_abs=cfg.clamp_abs
        )
    else:
        print("val_fixed exists:", val_fixed)

    dl_val_fixed = make_fixed_loader(val_fixed, batch_size=cfg.fixed_batch_size, num_workers=0)

    if os.path.exists(man_test):
        ds_test = NPZFileDataset(man_test, name2id, max_files=cfg.max_test_files, allow_pickle=True)
        _ = summarize_scan(ds_test, "scan-test")

        test_fixed = os.path.join(out_dir, f"test_fixed_bal_W{cfg.W}_H{cfg.H}.npz")
        if not os.path.exists(test_fixed):
            print("Creating balanced test_fixed:", test_fixed)
            make_fixed_balanced(
                ds_test, test_fixed,
                T_fixed=T_fixed,
                per_class_min=cfg.fixed_min_test,
                seed=cfg.fixed_seed + 1,
                clamp_abs=cfg.clamp_abs
            )
        else:
            print("test_fixed exists:", test_fixed)

        dl_test_fixed = make_fixed_loader(test_fixed, batch_size=cfg.fixed_batch_size, num_workers=0)
    else:
        test_fixed = None
        dl_test_fixed = None
        print("No test manifest found:", man_test)

    # sanity
    X_tmp, y_tmp = next(iter(dl_train_s1))
    print("sanity batch:", X_tmp.shape, y_tmp.shape)

    # ==========================================================
    # Stage 1: binary flare (quiet vs flare)
    # ==========================================================
    model_s1 = SimpleTCN(num_features=F, num_classes=2, hidden=cfg.hidden).to(device)

    # ✅ Stage1 loss (as requested)
    if cfg.s1_use_focal:
        loss_s1 = FocalLossBinary(gamma=cfg.s1_focal_gamma, pos_weight=cfg.s1_pos_weight)
    else:
        loss_s1 = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, float(cfg.s1_pos_weight)], device=device)
        )

    opt_s1 = torch.optim.Adam(model_s1.parameters(), lr=cfg.lr)

    for ep in range(1, cfg.s1_epochs + 1):
        t0 = time.perf_counter()
        model_s1.train()
        total, n = 0.0, 0

        for X, y in dl_train_s1:
            yb = (y != IDX_Q).long()
            X = X.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt_s1.zero_grad(set_to_none=True)
            logits = model_s1(X)
            loss = loss_s1(logits, yb)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            opt_s1.step()

            total += float(loss.item()) * X.size(0)
            n += X.size(0)

        dt = time.perf_counter() - t0

        probs1_va, y_va = predict_proba(model_s1, dl_val_fixed, device=device)
        p_flare_va = probs1_va[:, 1]
        y_bin = (y_va != IDX_Q).astype(np.int64)
        pred_bin = (p_flare_va >= 0.5).astype(np.int64)
        f1b = f1_score(y_bin, pred_bin, average="binary", zero_division=0)
        print(f"[S1 Ep {ep}] loss={total/max(n,1):.4f} | val_f1_bin@0.5={f1b:.4f} | {dt:.1f}s")

    save_checkpoint(out_dir, "stage1_last", model_s1, opt_s1, extra={"cfg": asdict(cfg)})

    # ==========================================================
    # Stage 2: severity (B/C/M/X) trained on flare samples only
    # ==========================================================
    model_s2 = SimpleTCN(num_features=F, num_classes=4, hidden=cfg.hidden).to(device)

    # ✅ Stage2 loss (as requested)
    if cfg.s2_alpha_mode != "manual":
        alpha_arr = compute_alpha_from_counts(train_counts, SEV_CLASSES, mode=cfg.s2_alpha_mode, power=cfg.s2_alpha_pow)
    else:
        alpha_arr = np.array(list(cfg.s2_alpha), dtype=np.float32)
    alpha = torch.tensor(alpha_arr, dtype=torch.float32, device=device)  # [B,C,M,X]
    if cfg.s2_use_focal:
        loss_s2 = FocalLossMulticlass(gamma=cfg.s2_focal_gamma, alpha=alpha)
    else:
        loss_s2 = nn.CrossEntropyLoss(weight=alpha)

    opt_s2 = torch.optim.Adam(model_s2.parameters(), lr=cfg.lr)

    for ep in range(1, cfg.s2_epochs + 1):
        t0 = time.perf_counter()
        model_s2.train()
        total, n = 0.0, 0

        total, n = 0.0, 0
        steps = 0
        printed = 0   # נדפיס רק 2–3 פעמים בכל epoch
        QUIET_ID = name2id["quiet"]
        # ✅ IMPORTANT: train Stage2 on dl_train_s2 (stage2 sampler)
        for batch_i, (X, y) in enumerate(dl_train_s2):
            
            mask = (y != IDX_Q)
            if mask.sum().item() == 0:
                continue

            X2 = X[mask].to(device, non_blocking=True)
            y2 = (y[mask] - 1).long().to(device, non_blocking=True)  # 0..3
            
            
            opt_s2.zero_grad(set_to_none=True)
            logits = model_s2(X2)

            if not torch.isfinite(logits).all():
                print("⚠️ WARNING: X2 contains NaN/Inf — skipping batch")
                continue

            loss = loss_s2(logits, y2)
            if not torch.isfinite(loss):
                print("⚠️ WARNING: logits contains NaN/Inf — skipping batch")
                continue

            loss.backward()
            if not torch.isfinite(logits).all():
                print("⚠️not torch.isfinite(logits).all() WARNING: logits contains NaN/Inf — skipping batch")
                continue
            torch.nn.utils.clip_grad_norm_(model_s2.parameters(), 1.0)
            opt_s2.step()

            total += float(loss.item()) * X2.size(0)
            n += X2.size(0)
            steps +=1

            

        dt = time.perf_counter() - t0
        print(f"[S2 Ep {ep}] loss={total/max(n,1):.4f} | {dt:.1f}s")
        print(f"[S2 Ep {ep}] steps={steps} loss={total/max(n,1):.10f} | {dt:.1f}s")
        

    save_checkpoint(out_dir, "stage2_last", model_s2, opt_s2, extra={"cfg": asdict(cfg)})

    # ==========================================================
    # Evaluate two-stage on VAL_FIXED_BAL
    # ==========================================================
    probs1_va, y_va = predict_proba(model_s1, dl_val_fixed, device=device)
    p_flare_va = probs1_va[:, 1]
    probs2_va, _ = predict_proba(model_s2, dl_val_fixed, device=device)

    # --------------------------
    # ✅ NEW: Stage1 diagnostics
    # --------------------------
    print("Stage1 p_flare stats on VAL:",
          "min", float(p_flare_va.min()),
          "mean", float(p_flare_va.mean()),
          "max", float(p_flare_va.max()))

    # per-class mean/min/max
    for cls_id, cls_name in [(IDX_Q, "quiet"), (IDX_B, "B"), (IDX_C, "C"), (IDX_M, "M"), (IDX_X, "X")]:
        m = (y_va == cls_id)
        if m.sum() > 0:
            vals = p_flare_va[m]
            print(f"Stage1 {cls_name} p_flare:",
                  "mean", float(vals.mean()),
                  "min", float(vals.min()),
                  "max", float(vals.max()),
                  "n", int(m.sum()))

    # --------------------------
    # --------------------------
    # Gate selection (macro-F1 target)
    # --------------------------
    mask_q = (y_va == IDX_Q)
    q_scores = p_flare_va[mask_q]
    if q_scores.size > 0:
        qs = np.quantile(q_scores, [0.5, 0.9, 0.95, 0.99])
        print("Stage1 quiet p_flare quantiles [50%,90%,95%,99%]:", [float(x) for x in qs])
    else:
        print("Stage1 quiet p_flare quantiles: no quiet samples in VAL")

    gate_info = {}
    if cfg.gate_mode in ("quiet_quantile", "soft_quantile"):
        q = float(cfg.gate_quantile)
        t_flare = float(np.quantile(q_scores, q)) if q_scores.size > 0 else 0.5
        gate_info = {"method": "quiet_quantile", "q": q}
    elif cfg.gate_mode in ("grid_macro_f1", "grid"):
        t_flare, gate_info = tune_t_flare(
            p_flare_va, probs2_va, y_va,
            grid=cfg.t_flare_grid,
            objective=cfg.gate_objective,
            recall_weight=cfg.gate_recall_weight,
            recall_classes=cfg.gate_recall_classes,
        )
        gate_info["method"] = "grid_tuned"
    elif cfg.gate_mode == "no_gate":
        t_flare = 0.0
        gate_info = {"method": "no_gate"}
    else:
        raise ValueError(f"Unknown gate_mode: {cfg.gate_mode}")

    t_sev = None
    sev_info = None
    if cfg.use_sev_thresholds:
        t_sev, sev_info = tune_sev_thresholds(
            p_flare_va, probs2_va, y_va,
            t_flare=t_flare,
            grid=cfg.sev_t_grid,
            iters=cfg.sev_t_iters,
        )
        print("t_sev:", {c: float(v) for c, v in zip(SEV_CLASSES, t_sev)})
        print("sev_info:", sev_info)

    pred_va = two_stage_predict(p_flare_va, probs2_va, t_flare=t_flare, t_sev=t_sev)
    mf1_va = f1_score(y_va, pred_va, average="macro", labels=list(range(K)))

    print("\n=== VAL_FIXED_BAL (two-stage) ===")
    print("gate_mode:", cfg.gate_mode, "| t_flare:", t_flare)
    print("gate_info:", gate_info)
    print(classification_report(y_va, pred_va, labels=list(range(K)), target_names=classes, zero_division=0))

    thresholds = {"two_stage": {"t_flare": float(t_flare), "gate_mode": cfg.gate_mode, "gate_info": gate_info}}
    if t_sev is not None:
        thresholds["two_stage"]["t_sev"] = [float(x) for x in t_sev]
        thresholds["two_stage"]["sev_info"] = sev_info

    # Save VAL artifacts using chosen gate
    np.save(os.path.join(out_dir, "val_p_flare.npy"), p_flare_va.astype(np.float32))
    np.save(os.path.join(out_dir, "val_p_sev.npy"), probs2_va.astype(np.float32))
    np.save(os.path.join(out_dir, "val_pred_two_stage.npy"), pred_va.astype(np.int64))

    cm_va = confusion_matrix(y_va, pred_va, labels=list(range(K)))
    val_metrics = {
        "split": "val_fixed_bal",
        "macro_f1_two_stage": float(mf1_va),
        "recall_per_class_two_stage": per_class_recall(cm_va),
        "confusion_matrix_two_stage": cm_va.tolist(),
        "thresholds": thresholds,
        "dist": dict(Counter(y_va.tolist())),
    }
    with open(os.path.join(out_dir, "val_fixed_bal_metrics_two_stage.json"), "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=2)

    #pred_va = two_stage_predict(p_flare_va, probs2_va, t_flare=t_flare)
    #mf1_va = f1_score(y_va, pred_va, average="macro", labels=list(range(K)))
    #print("\n=== VAL_FIXED_BAL (two-stage) ===")
    #print("t_flare tuned:", t_flare)
    #print("tune_info:", tune_info)
    #print(classification_report(y_va, pred_va, labels=list(range(K)), target_names=classes, zero_division=0))

    #thresholds = {"two_stage": {"t_flare": float(t_flare), **tune_info}}

    #np.save(os.path.join(out_dir, "val_p_flare.npy"), p_flare_va.astype(np.float32))
    #np.save(os.path.join(out_dir, "val_p_sev.npy"), probs2_va.astype(np.float32))
    #np.save(os.path.join(out_dir, "val_pred_two_stage.npy"), pred_va.astype(np.int64))

    #cm_va = confusion_matrix(y_va, pred_va, labels=list(range(K)))
    #val_metrics = {
    #    "split": "val_fixed_bal",
    #    "macro_f1_two_stage": float(mf1_va),
    #    "recall_per_class_two_stage": per_class_recall(cm_va),
    #    "confusion_matrix_two_stage": cm_va.tolist(),
    #    "thresholds": thresholds,
    #    "dist": dict(Counter(y_va.tolist())),
    #}
    #with open(os.path.join(out_dir, "val_fixed_bal_metrics_two_stage.json"), "w", encoding="utf-8") as f:
    #    json.dump(val_metrics, f, indent=2)

    # ==========================================================
    # Evaluate on TEST_FIXED_BAL using same t_flare
    # ==========================================================
    if dl_test_fixed is not None:
        probs1_te, y_te = predict_proba(model_s1, dl_test_fixed, device=device)
        p_flare_te = probs1_te[:, 1]
        probs2_te, _ = predict_proba(model_s2, dl_test_fixed, device=device)

        # debug: M drop
        mask_M = (y_te == IDX_M)
        print("M count:", int(mask_M.sum()))
        if mask_M.sum() > 0:
            print("M mean p_flare:", float(p_flare_te[mask_M].mean()))
            print("M % passed t:", float((p_flare_te[mask_M] >= t_flare).mean()))

            passed_M = mask_M & (p_flare_te >= t_flare)
            if passed_M.sum() > 0:
                sev_pred = predict_severity_with_thresholds(probs2_te[passed_M], t_sev=t_sev)
                print("Stage2 on M (passed) distribution:", dict(Counter(sev_pred.tolist())))

        pred_te = two_stage_predict(p_flare_te, probs2_te, t_flare=float(t_flare), t_sev=t_sev)
        mf1_te = f1_score(y_te, pred_te, average="macro", labels=list(range(K)))

        print("\n=== TEST_FIXED_BAL (two-stage) ===")
        print("t_flare (from VAL):", t_flare)
        print(classification_report(y_te, pred_te, labels=list(range(K)), target_names=classes, zero_division=0))

        np.save(os.path.join(out_dir, "test_p_flare.npy"), p_flare_te.astype(np.float32))
        np.save(os.path.join(out_dir, "test_p_sev.npy"), probs2_te.astype(np.float32))
        np.save(os.path.join(out_dir, "test_pred_two_stage.npy"), pred_te.astype(np.int64))

        cm_te = confusion_matrix(y_te, pred_te, labels=list(range(K)))
        test_metrics = {
            "split": "test_fixed_bal",
            "macro_f1_two_stage": float(mf1_te),
            "recall_per_class_two_stage": per_class_recall(cm_te),
            "confusion_matrix_two_stage": cm_te.tolist(),
            "thresholds": thresholds,
            "dist": dict(Counter(y_te.tolist())),
        }
        with open(os.path.join(out_dir, "test_fixed_bal_metrics_two_stage.json"), "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2)

    return {
        "out_dir": out_dir,
        "val_fixed_bal_path": val_fixed,
        "test_fixed_bal_path": test_fixed,
        "t_flare": float(t_flare),
        "thresholds": thresholds,
    }

def run_one_experiment2(cfg: ExperimentCfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    set_seed(cfg.seed)

    man_dir = os.path.join(cfg.root4, "manifests_by_group")
    man_train = os.path.join(man_dir, f"manifest_train_W{cfg.W}h_H{cfg.H}h.json")
    man_val   = os.path.join(man_dir, f"manifest_val_W{cfg.W}h_H{cfg.H}h.json")
    man_test  = os.path.join(man_dir, f"manifest_test_W{cfg.W}h_H{cfg.H}h.json")

    out_dir = os.path.join(cfg.out_root, f"W{cfg.W}_H{cfg.H}_two_stage_bal")
    os.makedirs(out_dir, exist_ok=True)

    # datasets
    ds_train = NPZFileDataset(man_train, name2id, max_files=cfg.max_train_files, allow_pickle=True)
    ds_val   = NPZFileDataset(man_val,   name2id, max_files=cfg.max_val_files, allow_pickle=True)
    print("train files:", len(ds_train), "| val files:", len(ds_val))

    train_counts = summarize_scan(ds_train, "scan-train")
    _ = summarize_scan(ds_val,   "scan-val")

    X0, _ = ds_train[0]
    T_fixed = int(X0.shape[1])
    F = int(X0.shape[2])
    print("T_fixed:", T_fixed, "| F:", F)

    # -------------------------
    # Train loader (Stage1)
    # -------------------------
    sampler_s1 = make_file_weighted_sampler(
        ds_train, wX=cfg.wX, wM=cfg.wM, wC=cfg.wC, wB=cfg.wB, wQ=cfg.wQ
    )
    dl_train_s1 = DataLoader(
        ds_train,
        batch_size=cfg.batch_files,
        sampler=sampler_s1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=lambda b: collate_take_random_samples_fixedT(
            b, per_file=cfg.per_file, T_fixed=T_fixed, clamp_abs=cfg.clamp_abs
        ),
    )

    # -------------------------
    # Stage2 loader (same ds_train, different sampler)
    # -------------------------
    sampler_s2 = make_file_weighted_sampler(
        ds_train, wX=cfg.s2_wX, wM=cfg.s2_wM, wC=cfg.s2_wC, wB=cfg.s2_wB, wQ=cfg.s2_wQ
    )
    dl_train_s2 = DataLoader(
        ds_train,
        batch_size=cfg.batch_files,
        sampler=sampler_s2,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        #collate_fn=lambda b: collate_take_random_samples_fixedT(
        #    b, per_file=cfg.per_file, T_fixed=T_fixed, clamp_abs=cfg.clamp_abs
        collate_fn=lambda b: collate_stage2_quota_fixedT(
        b,
        per_file=cfg.per_file,
        T_fixed=T_fixed,
        clamp_abs=cfg.clamp_abs,
        flare_frac=cfg.s2_flare_frac,
        flare_quota=cfg.s2_flare_quota,
        x_min_per_file=cfg.s2_x_min_per_file,
        x_max_per_file=cfg.s2_x_max_per_file,
        quiet_id=name2id["quiet"],
        B_id=name2id["B"],
        C_id=name2id["C"],
        M_id=name2id["M"],
        X_id=name2id["X"],),
        #flare_frac=0.80,
        #flare_quota={
        #    name2id["B"]: 0.25,
        #    name2id["C"]: 0.45,
        #    name2id["M"]: 0.20,
        #    name2id["X"]: 0.10,
        #},
        #x_max_per_file=4,
        
    )

    # -------- fixed balanced VAL/TEST --------
    if cfg.fixed_min_val is None:
        cfg.fixed_min_val = {"X":200,"M":200,"C":800,"B":200,"quiet":8000}
    if cfg.fixed_min_test is None:
        cfg.fixed_min_test = {"X":200,"M":200,"C":800,"B":200,"quiet":12000}

    val_fixed = os.path.join(out_dir, f"val_fixed_bal_W{cfg.W}_H{cfg.H}.npz")
    if not os.path.exists(val_fixed):
        print("Creating balanced val_fixed:", val_fixed)
        make_fixed_balanced(
            ds_val, val_fixed,
            T_fixed=T_fixed,
            per_class_min=cfg.fixed_min_val,
            seed=cfg.fixed_seed,
            clamp_abs=cfg.clamp_abs
        )
    else:
        print("val_fixed exists:", val_fixed)

    #dl_val_fixed = make_fixed_loader(val_fixed, batch_size=cfg.fixed_batch_size, num_workers=0)
    dl_val_fixed = DataLoader(
            ds_val,
            batch_size=cfg.batch_files,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=lambda b: collate_take_random_samples_fixedT(
                b, per_file=cfg.per_file, T_fixed=T_fixed, clamp_abs=cfg.clamp_abs
            ),
        )
    if os.path.exists(man_test):
        ds_test = NPZFileDataset(man_test, name2id, max_files=cfg.max_test_files, allow_pickle=True)
        _ = summarize_scan(ds_test, "scan-test")

        test_fixed = os.path.join(out_dir, f"test_fixed_bal_W{cfg.W}_H{cfg.H}.npz")
        if not os.path.exists(test_fixed):
            print("Creating balanced test_fixed:", test_fixed)
            make_fixed_balanced(
                ds_test, test_fixed,
                T_fixed=T_fixed,
                per_class_min=cfg.fixed_min_test,
                seed=cfg.fixed_seed + 1,
                clamp_abs=cfg.clamp_abs
            )
        else:
            print("test_fixed exists:", test_fixed)

        dl_test_fixed = make_fixed_loader(test_fixed, batch_size=cfg.fixed_batch_size, num_workers=0)
    else:
        test_fixed = None
        dl_test_fixed = None
        print("No test manifest found:", man_test)

    # sanity
    X_tmp, y_tmp = next(iter(dl_train_s1))
    print("sanity batch:", X_tmp.shape, y_tmp.shape)

    # ==========================================================
    # Stage 1: binary flare (quiet vs flare)
    # ==========================================================
    model_s1 = SimpleTCN(num_features=F, num_classes=2, hidden=cfg.hidden).to(device)

    # ✅ Stage1 loss (as requested)
    if cfg.s1_use_focal:
        loss_s1 = FocalLossBinary(gamma=cfg.s1_focal_gamma, pos_weight=cfg.s1_pos_weight)
    else:
        loss_s1 = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, float(cfg.s1_pos_weight)], device=device)
        )

    opt_s1 = torch.optim.Adam(model_s1.parameters(), lr=cfg.lr)

    for ep in range(1, cfg.s1_epochs + 1):
        t0 = time.perf_counter()
        model_s1.train()
        total, n = 0.0, 0

        for X, y in dl_train_s1:
            yb = (y != IDX_Q).long()
            X = X.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt_s1.zero_grad(set_to_none=True)
            logits = model_s1(X)
            loss = loss_s1(logits, yb)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            opt_s1.step()

            total += float(loss.item()) * X.size(0)
            n += X.size(0)

        dt = time.perf_counter() - t0

        probs1_va, y_va = predict_proba(model_s1, dl_val_fixed, device=device)
        p_flare_va = probs1_va[:, 1]
        y_bin = (y_va != IDX_Q).astype(np.int64)
        pred_bin = (p_flare_va >= 0.5).astype(np.int64)
        f1b = f1_score(y_bin, pred_bin, average="binary", zero_division=0)
        print(f"[S1 Ep {ep}] loss={total/max(n,1):.4f} | val_f1_bin@0.5={f1b:.4f} | {dt:.1f}s")

    save_checkpoint(out_dir, "stage1_last", model_s1, opt_s1, extra={"cfg": asdict(cfg)})

    # ==========================================================
    # Stage 2: severity (quiet + B/C/M/X) trained on flare samples only
    # ==========================================================
    model_s2 = SimpleTCN(num_features=F, num_classes=5, hidden=cfg.hidden).to(device)
    quiet_w = 0.2

    # ✅ Stage2 loss (as requested)
    if cfg.s2_alpha_mode != "manual":
        alpha_sev  = compute_alpha_from_counts(train_counts, SEV_CLASSES, mode=cfg.s2_alpha_mode, power=cfg.s2_alpha_pow)
    else:
        alpha_sev  = np.array(list(cfg.s2_alpha), dtype=np.float32)

    alpha_arr = np.concatenate(([quiet_w], alpha_sev)).astype(np.float32)
    alpha = torch.tensor(alpha_arr, dtype=torch.float32, device=device)
    # [B,C,M,X]
    
    if cfg.s2_use_focal:
        loss_s2 = FocalLossMulticlass(gamma=cfg.s2_focal_gamma, alpha=alpha)
    else:
        loss_s2 = nn.CrossEntropyLoss(weight=alpha)

    opt_s2 = torch.optim.Adam(model_s2.parameters(), lr=cfg.lr)

    for ep in range(1, cfg.s2_epochs + 1):
        t0 = time.perf_counter()
        model_s2.train()
        total, n = 0.0, 0

        total, n = 0.0, 0
        steps = 0
        printed = 0   # נדפיס רק 2–3 פעמים בכל epoch
        QUIET_ID = name2id["quiet"]
        # ✅ IMPORTANT: train Stage2 on dl_train_s2 (stage2 sampler)
        for batch_i, (X, y) in enumerate(dl_train_s2):
            X2 = X.to(device, non_blocking=True)
            y2 = y.long().to(device, non_blocking=True)  # 0..4 (quiet,B,C,M,X)

            opt_s2.zero_grad(set_to_none=True)
            logits = model_s2(X2)

            if not torch.isfinite(logits).all():
                print("WARNING: logits contains NaN/Inf -- skipping batch")
                continue

            loss = loss_s2(logits, y2)
            if not torch.isfinite(loss):
                print("WARNING: loss contains NaN/Inf -- skipping batch")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_s2.parameters(), 1.0)
            opt_s2.step()

            total += float(loss.item()) * X2.size(0)
            n += X2.size(0)
            steps += 1

            #mask = (y != IDX_Q)
            #if mask.sum().item() == 0:
            #    continue
#
            #X2 = X[mask].to(device, non_blocking=True)
            #y2 = (y[mask] - 1).long().to(device, non_blocking=True)  # 0..3
            #
            #
            #opt_s2.zero_grad(set_to_none=True)
            #logits = model_s2(X2)
#
            #if not torch.isfinite(logits).all():
            #    print("⚠️ WARNING: X2 contains NaN/Inf — skipping batch")
            #    continue
#
            #loss = loss_s2(logits, y2)
            #if not torch.isfinite(loss):
            #    print("⚠️ WARNING: logits contains NaN/Inf — skipping batch")
            #    continue
#
            #loss.backward()
            #if not torch.isfinite(logits).all():
            #    print("⚠️not torch.isfinite(logits).all() WARNING: logits contains NaN/Inf — skipping batch")
            #    continue
            #torch.nn.utils.clip_grad_norm_(model_s2.parameters(), 1.0)
            #opt_s2.step()
#
            #total += float(loss.item()) * X2.size(0)
            #n += X2.size(0)
            #steps +=1

            

        dt = time.perf_counter() - t0
        print(f"[S2 Ep {ep}] loss={total/max(n,1):.4f} | {dt:.1f}s")
        print(f"[S2 Ep {ep}] steps={steps} loss={total/max(n,1):.10f} | {dt:.1f}s")
        

    save_checkpoint(out_dir, "stage2_last", model_s2, opt_s2, extra={"cfg": asdict(cfg)})

    # ==========================================================
    # Evaluate two-stage on VAL_FIXED_BAL
    # ==========================================================
    probs1_va, y_va = predict_proba(model_s1, dl_val_fixed, device=device)
    p_flare_va = probs1_va[:, 1]
    probs2_va, _ = predict_proba(model_s2, dl_val_fixed, device=device)
    
    # --------------------------
    # ✅ NEW: Stage1 diagnostics
    # --------------------------
    print("Stage1 p_flare stats on VAL:",
          "min", float(p_flare_va.min()),
          "mean", float(p_flare_va.mean()),
          "max", float(p_flare_va.max()))

    # per-class mean/min/max
    for cls_id, cls_name in [(IDX_Q, "quiet"), (IDX_B, "B"), (IDX_C, "C"), (IDX_M, "M"), (IDX_X, "X")]:
        m = (y_va == cls_id)
        if m.sum() > 0:
            vals = p_flare_va[m]
            print(f"Stage1 {cls_name} p_flare:",
                  "mean", float(vals.mean()),
                  "min", float(vals.min()),
                  "max", float(vals.max()),
                  "n", int(m.sum()))

    # --------------------------
    # Gate selection (macro-F1 target)
    # --------------------------
    mask_q = (y_va == IDX_Q)
    q_scores = p_flare_va[mask_q]
    if q_scores.size > 0:
        qs = np.quantile(q_scores, [0.5, 0.9, 0.95, 0.99])
        print("Stage1 quiet p_flare quantiles [50%,90%,95%,99%]:", [float(x) for x in qs])
    else:
        print("Stage1 quiet p_flare quantiles: no quiet samples in VAL")

    gate_info = {}
    if cfg.gate_mode in ("quiet_quantile", "soft_quantile"):
        q = float(cfg.gate_quantile)
        t_flare = float(np.quantile(q_scores, q)) if q_scores.size > 0 else 0.5
        gate_info = {"method": "quiet_quantile", "q": q}
    elif cfg.gate_mode in ("grid_macro_f1", "grid"):
        t_flare, gate_info = tune_t_flare_quiet(
            p_flare_va, probs2_va, y_va,
            grid=cfg.t_flare_grid,
            objective=cfg.gate_objective,
            recall_weight=cfg.gate_recall_weight,
            recall_classes=cfg.gate_recall_classes,
        )
        gate_info["method"] = "grid_tuned"
    elif cfg.gate_mode == "no_gate":
        t_flare = 0.0
        gate_info = {"method": "no_gate"}
    else:
        raise ValueError(f"Unknown gate_mode: {cfg.gate_mode}")

    t_sev = None
    sev_info = None
    if cfg.use_sev_thresholds:
        #t_sev, sev_info = tune_sev_thresholds(
        #    p_flare_va, probs2_va, y_va,
        #    t_flare=t_flare,
        #    grid=cfg.sev_t_grid,
        #    iters=cfg.sev_t_iters,
        #    )
        best = tune_sev_thresholds_with_constraints(
            p_flare_va, probs2_va, y_va, t_flare,
            grid=cfg.sev_t_grid,
            #min_prec_B=0.20, min_rec_M=0.30, min_rec_X=0.30
        )
        print(best)
        if best is not None:
            t_sev = best["t_sev"]
            sev_info = best

        print("t_sev:", {c: float(v) for c, v in zip(SEV_CLASSES, t_sev)})
        print("sev_info:", sev_info)

    pred_va = two_stage_predict_with_quiet(p_flare_va, probs2_va, t_flare=t_flare, t_sev=t_sev)
    mf1_va = f1_score(y_va, pred_va, average="macro", labels=list(range(K)))

    print("\n=== VAL_FIXED_BAL (two-stage) ===")
    print("gate_mode:", cfg.gate_mode, "| t_flare:", t_flare)
    print("gate_info:", gate_info)
    print(classification_report(y_va, pred_va, labels=list(range(K)), target_names=classes, zero_division=0))

    # --------------------------
    # GATE CHECKER START (simple ON vs OFF)
    # --------------------------
    t_flare_soft = 0.0  # "no gate" baseline
    if cfg.gate_mode != "no_gate":
        pred_va_soft = two_stage_predict_with_quiet(p_flare_va, probs2_va, t_flare=t_flare_soft, t_sev=t_sev)


        gate_check = gate_checker(
            y_true=y_va,
            y_pred_on=pred_va,
            y_pred_off=pred_va_soft,
            labels=list(range(K)),
            focus=(IDX_B, IDX_C, IDX_M, IDX_X)
        )

        print("\n=== GATE CHECKER (VAL) ===")
        print("macro_f1_on :", gate_check["macro_f1_on"])
        print("macro_f1_off:", gate_check["macro_f1_off"])
        print("decision    :", gate_check["decision"])

        if gate_check["hurts"]:
            print("Gate hurts -> switching to soft/no gate for VAL+TEST")
            t_flare = t_flare_soft
            pred_va = pred_va_soft
            mf1_va = f1_score(y_va, pred_va, average="macro", labels=list(range(K)))

            print("\n=== VAL_FIXED_BAL (two-stage) [SOFT/NO GATE] ===")
            print("t_flare:", t_flare)
            print(classification_report(y_va, pred_va, labels=list(range(K)), target_names=classes, zero_division=0))
    # --------------------------
    # GATE CHECKER END
    # --------------------------

    thresholds = {"two_stage": {"t_flare": float(t_flare), "gate_mode": cfg.gate_mode, "gate_info": gate_info}}
    if t_sev is not None:
        thresholds["two_stage"]["t_sev"] = [float(x) for x in t_sev]
        thresholds["two_stage"]["sev_info"] = sev_info

    # Save VAL artifacts using chosen gate
    np.save(os.path.join(out_dir, "val_p_flare.npy"), p_flare_va.astype(np.float32))
    np.save(os.path.join(out_dir, "val_p_sev.npy"), probs2_va.astype(np.float32))
    np.save(os.path.join(out_dir, "val_pred_two_stage.npy"), pred_va.astype(np.int64))

    cm_va = confusion_matrix(y_va, pred_va, labels=list(range(K)))
    val_metrics = {
        "split": "val_fixed_bal",
        "macro_f1_two_stage": float(mf1_va),
        "recall_per_class_two_stage": per_class_recall(cm_va),
        "confusion_matrix_two_stage": cm_va.tolist(),
        "thresholds": thresholds,
        "dist": dict(Counter(y_va.tolist())),
    }
    with open(os.path.join(out_dir, "val_fixed_bal_metrics_two_stage.json"), "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=2)

    # ==========================================================
    # Evaluate on TEST_FIXED_BAL using same t_flare
    # ==========================================================
    if dl_test_fixed is not None:
        probs1_te, y_te = predict_proba(model_s1, dl_test_fixed, device=device)
        p_flare_te = probs1_te[:, 1]
        probs2_te, _ = predict_proba(model_s2, dl_test_fixed, device=device)

        # debug: M drop
        mask_M = (y_te == IDX_M)
        print("M count:", int(mask_M.sum()))
        if mask_M.sum() > 0:
            print("M mean p_flare:", float(p_flare_te[mask_M].mean()))
            print("M % passed t:", float((p_flare_te[mask_M] >= t_flare).mean()))

            passed_M = mask_M & (p_flare_te >= t_flare)
            if passed_M.sum() > 0:
                sev_pred = predict_severity_with_thresholds(probs2_te[passed_M], t_sev=t_sev)
                print("Stage2 on M (passed) distribution:", dict(Counter(sev_pred.tolist())))

        pred_te = two_stage_predict_with_quiet(p_flare_te, probs2_te, t_flare=float(t_flare), t_sev=t_sev)
        mf1_te = f1_score(y_te, pred_te, average="macro", labels=list(range(K)))

        print("\n=== TEST_FIXED_BAL (two-stage) ===")
        print("t_flare (from VAL):", t_flare)
        print(classification_report(y_te, pred_te, labels=list(range(K)), target_names=classes, zero_division=0))

        np.save(os.path.join(out_dir, "test_p_flare.npy"), p_flare_te.astype(np.float32))
        np.save(os.path.join(out_dir, "test_p_sev.npy"), probs2_te.astype(np.float32))
        np.save(os.path.join(out_dir, "test_pred_two_stage.npy"), pred_te.astype(np.int64))

        cm_te = confusion_matrix(y_te, pred_te, labels=list(range(K)))
        test_metrics = {
            "split": "test_fixed_bal",
            "macro_f1_two_stage": float(mf1_te),
            "recall_per_class_two_stage": per_class_recall(cm_te),
            "confusion_matrix_two_stage": cm_te.tolist(),
            "thresholds": thresholds,
            "dist": dict(Counter(y_te.tolist())),
        }
        with open(os.path.join(out_dir, "test_fixed_bal_metrics_two_stage.json"), "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2)

    return {
        "out_dir": out_dir,
        "val_fixed_bal_path": val_fixed,
        "test_fixed_bal_path": test_fixed,
        "t_flare": float(t_flare),
        "thresholds": thresholds,
    }

def run_one_experiment_skip_stage1(cfg: ExperimentCfgSkipStage1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    set_seed(cfg.seed)

    man_dir = os.path.join(cfg.root4, "manifests_by_group")
    man_train = os.path.join(man_dir, f"manifest_train_W{cfg.W}h_H{cfg.H}h.json")
    man_val   = os.path.join(man_dir, f"manifest_val_W{cfg.W}h_H{cfg.H}h.json")
    man_test  = os.path.join(man_dir, f"manifest_test_W{cfg.W}h_H{cfg.H}h.json")

    out_dir = os.path.join(cfg.out_root, f"W{cfg.W}_H{cfg.H}_two_stage_bal_skip_stage1")
    os.makedirs(out_dir, exist_ok=True)

    # datasets
    ds_train = NPZFileDataset(man_train, name2id, max_files=cfg.max_train_files, allow_pickle=True)
    ds_val   = NPZFileDataset(man_val,   name2id, max_files=cfg.max_val_files, allow_pickle=True)
    print("train files:", len(ds_train), "| val files:", len(ds_val))

    train_counts = summarize_scan(ds_train, "scan-train")
    _ = summarize_scan(ds_val,   "scan-val")

    X0, _ = ds_train[0]
    T_fixed = int(X0.shape[1])
    F = int(X0.shape[2])
    print("T_fixed:", T_fixed, "| F:", F)

    # -------------------------
    # Train loader (Stage1)
    # -------------------------
    sampler_s1 = make_file_weighted_sampler(
        ds_train, wX=cfg.wX, wM=cfg.wM, wC=cfg.wC, wB=cfg.wB, wQ=cfg.wQ
    )
    dl_train_s1 = DataLoader(
        ds_train,
        batch_size=cfg.batch_files,
        sampler=sampler_s1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=lambda b: collate_take_random_samples_fixedT(
            b, per_file=cfg.per_file, T_fixed=T_fixed, clamp_abs=cfg.clamp_abs
        ),
    )

    # -------------------------
    # Stage2 loader (same ds_train, different sampler)
    # -------------------------
    sampler_s2 = make_file_weighted_sampler(
        ds_train, wX=cfg.s2_wX, wM=cfg.s2_wM, wC=cfg.s2_wC, wB=cfg.s2_wB, wQ=cfg.s2_wQ
    )
    dl_train_s2 = DataLoader(
        ds_train,
        batch_size=cfg.batch_files,
        sampler=sampler_s2,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        #collate_fn=lambda b: collate_take_random_samples_fixedT(
        #    b, per_file=cfg.per_file, T_fixed=T_fixed, clamp_abs=cfg.clamp_abs
        collate_fn=lambda b: collate_stage2_quota_fixedT(
        b,
        per_file=cfg.per_file,
        T_fixed=T_fixed,
        clamp_abs=cfg.clamp_abs,
        flare_frac=cfg.s2_flare_frac,
        flare_quota=cfg.s2_flare_quota,
        x_min_per_file=cfg.s2_x_min_per_file,
        x_max_per_file=cfg.s2_x_max_per_file,
        quiet_id=name2id["quiet"],
        B_id=name2id["B"],
        C_id=name2id["C"],
        M_id=name2id["M"],
        X_id=name2id["X"],
        #flare_frac=0.80,
        #flare_quota={
        #    name2id["B"]: 0.25,
        #    name2id["C"]: 0.45,
        #    name2id["M"]: 0.20,
        #    name2id["X"]: 0.10,
        #},
        #x_max_per_file=4,
        ),
    )

    # -------- fixed balanced VAL/TEST --------
    if cfg.fixed_min_val is None:
        cfg.fixed_min_val = {"X":200,"M":200,"C":800,"B":200,"quiet":8000}
    if cfg.fixed_min_test is None:
        cfg.fixed_min_test = {"X":200,"M":200,"C":800,"B":200,"quiet":12000}

    val_fixed = os.path.join(out_dir, f"val_fixed_bal_W{cfg.W}_H{cfg.H}.npz")
    if not os.path.exists(val_fixed):
        print("Creating balanced val_fixed:", val_fixed)
        make_fixed_balanced(
            ds_val, val_fixed,
            T_fixed=T_fixed,
            per_class_min=cfg.fixed_min_val,
            seed=cfg.fixed_seed,
            clamp_abs=cfg.clamp_abs
        )
    else:
        print("val_fixed exists:", val_fixed)

    dl_val_fixed = make_fixed_loader(val_fixed, batch_size=cfg.fixed_batch_size, num_workers=0)

    if os.path.exists(man_test):
        ds_test = NPZFileDataset(man_test, name2id, max_files=cfg.max_test_files, allow_pickle=True)
        _ = summarize_scan(ds_test, "scan-test")

        test_fixed = os.path.join(out_dir, f"test_fixed_bal_W{cfg.W}_H{cfg.H}.npz")
        if not os.path.exists(test_fixed):
            print("Creating balanced test_fixed:", test_fixed)
            make_fixed_balanced(
                ds_test, test_fixed,
                T_fixed=T_fixed,
                per_class_min=cfg.fixed_min_test,
                seed=cfg.fixed_seed + 1,
                clamp_abs=cfg.clamp_abs
            )
        else:
            print("test_fixed exists:", test_fixed)

        dl_test_fixed = make_fixed_loader(test_fixed, batch_size=cfg.fixed_batch_size, num_workers=0)
    else:
        test_fixed = None
        dl_test_fixed = None
        print("No test manifest found:", man_test)

    # sanity
    X_tmp, y_tmp = next(iter(dl_train_s1))
    print("sanity batch:", X_tmp.shape, y_tmp.shape)

    # ==========================================================
    # Stage 1: binary flare (quiet vs flare)
    # ==========================================================
    model_s1 = SimpleTCN(num_features=F, num_classes=2, hidden=cfg.hidden).to(device)

    # ✅ Stage1 loss (as requested)
    if cfg.s1_use_focal:
        loss_s1 = FocalLossBinary(gamma=cfg.s1_focal_gamma, pos_weight=cfg.s1_pos_weight)
    else:
        loss_s1 = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, float(cfg.s1_pos_weight)], device=device)
        )

    opt_s1 = torch.optim.Adam(model_s1.parameters(), lr=cfg.lr)

    if cfg.skip_stage1_train:
        print("Skipping Stage1 training (using loaded weights).")
    else:
        for ep in range(1, cfg.s1_epochs + 1):
            t0 = time.perf_counter()
            model_s1.train()
            total, n = 0.0, 0

            for X, y in dl_train_s1:
                yb = (y != IDX_Q).long()
                X = X.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                opt_s1.zero_grad(set_to_none=True)
                logits = model_s1(X)
                loss = loss_s1(logits, yb)

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                loss.backward()
                opt_s1.step()

                total += float(loss.item()) * X.size(0)
                n += X.size(0)

            dt = time.perf_counter() - t0

            probs1_va, y_va = predict_proba(model_s1, dl_val_fixed, device=device)
            p_flare_va = probs1_va[:, 1]
            y_bin = (y_va != IDX_Q).astype(np.int64)
            pred_bin = (p_flare_va >= 0.5).astype(np.int64)
            f1b = f1_score(y_bin, pred_bin, average="binary", zero_division=0)
            print(f"[S1 Ep {ep}] loss={total/max(n,1):.4f} | val_f1_bin@0.5={f1b:.4f} | {dt:.1f}s")

        save_checkpoint(out_dir, "stage1_last", model_s1, opt_s1, extra={"cfg": asdict(cfg)})

    # ==========================================================
    # Stage 2: severity (B/C/M/X) trained on flare samples only
    # ==========================================================
    model_s2 = SimpleTCN(num_features=F, num_classes=4, hidden=cfg.hidden).to(device)

    # ✅ Stage2 loss (as requested)
    if cfg.s2_alpha_mode != "manual":
        alpha_arr = compute_alpha_from_counts(train_counts, SEV_CLASSES, mode=cfg.s2_alpha_mode, power=cfg.s2_alpha_pow)
    else:
        alpha_arr = np.array(list(cfg.s2_alpha), dtype=np.float32)
    alpha = torch.tensor(alpha_arr, dtype=torch.float32, device=device)  # [B,C,M,X]
    if cfg.s2_use_focal:
        loss_s2 = FocalLossMulticlass(gamma=cfg.s2_focal_gamma, alpha=alpha)
    else:
        loss_s2 = nn.CrossEntropyLoss(weight=alpha)

    opt_s2 = torch.optim.Adam(model_s2.parameters(), lr=cfg.lr)

    for ep in range(1, cfg.s2_epochs + 1):
        t0 = time.perf_counter()
        model_s2.train()
        total, n = 0.0, 0

        total, n = 0.0, 0
        steps = 0
        printed = 0   # נדפיס רק 2–3 פעמים בכל epoch
        
        # ✅ IMPORTANT: train Stage2 on dl_train_s2 (stage2 sampler)
        for batch_i, (X, y) in enumerate(dl_train_s2):
            if batch_i == 0:
                print("S2 first batch arrived. X shape:", tuple(X.shape), "y shape:", tuple(y.shape))

            mask = (y != IDX_Q)
            if mask.sum().item() == 0:
                if printed < 1:
                    print("S2: mask is empty (all quiet) in this batch.")
                    printed += 1
                continue

            X2 = X[mask]
            y2 = (y[mask] - 1).long()  # 1..4 -> 0..3

            X2 = X2.to(device, non_blocking=True)
            y2 = y2.to(device, non_blocking=True)

            opt_s2.zero_grad(set_to_none=True)
            logits = model_s2(X2)
            loss = loss_s2(logits, y2)

            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            
            if printed < 3:
                with torch.no_grad():
                    raw5 = torch.bincount(y, minlength=5).cpu().tolist()
                    y2_cpu = y2.detach().cpu()
                    y2_4 = torch.bincount(y2_cpu, minlength=4).cpu().tolist()
                    pred = logits.argmax(dim=1).detach().cpu()
                    p_4 = torch.bincount(pred, minlength=4).cpu().tolist()

                print("S2 train batch bincount (raw5):", raw5)
                print("S2 train batch bincount (y2_4):", y2_4)
                print("S2 train pred  bincount (p_4): ", p_4)
                print("S2 step loss:", float(loss.item()))
                printed += 1

            loss.backward()
            opt_s2.step()

            total += float(loss.item()) * X2.size(0)
            n += X2.size(0)
            steps +=1

            

        dt = time.perf_counter() - t0
        print(f"[S2 Ep {ep}] loss={total/max(n,1):.4f} | {dt:.1f}s")
        print(f"[S2 Ep {ep}] steps={steps} loss={total/max(n,1):.10f} | {dt:.1f}s")
        

    save_checkpoint(out_dir, "stage2_last", model_s2, opt_s2, extra={"cfg": asdict(cfg)})

    # ==========================================================
    # Evaluate two-stage on VAL_FIXED_BAL
    # ==========================================================
    probs1_va, y_va = predict_proba(model_s1, dl_val_fixed, device=device)
    p_flare_va = probs1_va[:, 1]
    probs2_va, _ = predict_proba(model_s2, dl_val_fixed, device=device)

    gate_info = {}
    if cfg.gate_mode in ("quiet_quantile", "soft_quantile"):
        mask_q = (y_va == IDX_Q)
        q_scores = p_flare_va[mask_q]
        q = float(cfg.gate_quantile)
        t_flare = float(np.quantile(q_scores, q)) if q_scores.size > 0 else 0.5
        gate_info = {"method": "quiet_quantile", "q": q}
    elif cfg.gate_mode in ("grid_macro_f1", "grid"):
        t_flare, gate_info = tune_t_flare(
            p_flare_va, probs2_va, y_va,
            grid=cfg.t_flare_grid,
            objective=cfg.gate_objective,
            recall_weight=cfg.gate_recall_weight,
            recall_classes=cfg.gate_recall_classes,
        )
        gate_info["method"] = "grid_tuned"
    elif cfg.gate_mode == "no_gate":
        t_flare = 0.0
        gate_info = {"method": "no_gate"}
    else:
        raise ValueError(f"Unknown gate_mode: {cfg.gate_mode}")

    t_sev = None
    sev_info = None
    if cfg.use_sev_thresholds:
        t_sev, sev_info = tune_sev_thresholds(
            p_flare_va, probs2_va, y_va,
            t_flare=t_flare,
            grid=cfg.sev_t_grid,
            iters=cfg.sev_t_iters,
        )
        print("t_sev:", {c: float(v) for c, v in zip(SEV_CLASSES, t_sev)})
        print("sev_info:", sev_info)

    pred_va = two_stage_predict(p_flare_va, probs2_va, t_flare=t_flare, t_sev=t_sev)
    mf1_va = f1_score(y_va, pred_va, average="macro", labels=list(range(K)))
    print("\n=== VAL_FIXED_BAL (two-stage) ===")
    print("gate_mode:", cfg.gate_mode, "| t_flare:", t_flare)
    print("gate_info:", gate_info)
    print(classification_report(y_va, pred_va, labels=list(range(K)), target_names=classes, zero_division=0))

    thresholds = {"two_stage": {"t_flare": float(t_flare), "gate_mode": cfg.gate_mode, "gate_info": gate_info}}
    if t_sev is not None:
        thresholds["two_stage"]["t_sev"] = [float(x) for x in t_sev]
        thresholds["two_stage"]["sev_info"] = sev_info
    np.save(os.path.join(out_dir, "val_p_flare.npy"), p_flare_va.astype(np.float32))
    np.save(os.path.join(out_dir, "val_p_sev.npy"), probs2_va.astype(np.float32))
    np.save(os.path.join(out_dir, "val_pred_two_stage.npy"), pred_va.astype(np.int64))

    cm_va = confusion_matrix(y_va, pred_va, labels=list(range(K)))
    val_metrics = {
        "split": "val_fixed_bal",
        "macro_f1_two_stage": float(mf1_va),
        "recall_per_class_two_stage": per_class_recall(cm_va),
        "confusion_matrix_two_stage": cm_va.tolist(),
        "thresholds": thresholds,
        "dist": dict(Counter(y_va.tolist())),
    }
    with open(os.path.join(out_dir, "val_fixed_bal_metrics_two_stage.json"), "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=2)

    # ==========================================================
    # Evaluate on TEST_FIXED_BAL using same t_flare
    # ==========================================================
    if dl_test_fixed is not None:
        probs1_te, y_te = predict_proba(model_s1, dl_test_fixed, device=device)
        p_flare_te = probs1_te[:, 1]
        probs2_te, _ = predict_proba(model_s2, dl_test_fixed, device=device)

        # debug: M drop
        mask_M = (y_te == IDX_M)
        print("M count:", int(mask_M.sum()))
        if mask_M.sum() > 0:
            print("M mean p_flare:", float(p_flare_te[mask_M].mean()))
            print("M % passed t:", float((p_flare_te[mask_M] >= t_flare).mean()))

            passed_M = mask_M & (p_flare_te >= t_flare)
            if passed_M.sum() > 0:
                sev_pred = predict_severity_with_thresholds(probs2_te[passed_M], t_sev=t_sev)
                print("Stage2 on M (passed) distribution:", dict(Counter(sev_pred.tolist())))

        pred_te = two_stage_predict(p_flare_te, probs2_te, t_flare=float(t_flare), t_sev=t_sev)
        mf1_te = f1_score(y_te, pred_te, average="macro", labels=list(range(K)))

        print("\n=== TEST_FIXED_BAL (two-stage) ===")
        print("t_flare (from VAL):", t_flare)
        print(classification_report(y_te, pred_te, labels=list(range(K)), target_names=classes, zero_division=0))

        np.save(os.path.join(out_dir, "test_p_flare.npy"), p_flare_te.astype(np.float32))
        np.save(os.path.join(out_dir, "test_p_sev.npy"), probs2_te.astype(np.float32))
        np.save(os.path.join(out_dir, "test_pred_two_stage.npy"), pred_te.astype(np.int64))

        cm_te = confusion_matrix(y_te, pred_te, labels=list(range(K)))
        test_metrics = {
            "split": "test_fixed_bal",
            "macro_f1_two_stage": float(mf1_te),
            "recall_per_class_two_stage": per_class_recall(cm_te),
            "confusion_matrix_two_stage": cm_te.tolist(),
            "thresholds": thresholds,
            "dist": dict(Counter(y_te.tolist())),
        }
        with open(os.path.join(out_dir, "test_fixed_bal_metrics_two_stage.json"), "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2)

    return {
        "out_dir": out_dir,
        "val_fixed_bal_path": val_fixed,
        "test_fixed_bal_path": test_fixed,
        "t_flare": float(t_flare),
        "thresholds": thresholds,
    }

def run_one_experiment_skip_stage1_all_print(cfg: ExperimentCfgSkipStage1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    set_seed(cfg.seed)

    man_dir = os.path.join(cfg.root4, "manifests_by_group")
    man_train = os.path.join(man_dir, f"manifest_train_W{cfg.W}h_H{cfg.H}h.json")
    man_val   = os.path.join(man_dir, f"manifest_val_W{cfg.W}h_H{cfg.H}h.json")
    man_test  = os.path.join(man_dir, f"manifest_test_W{cfg.W}h_H{cfg.H}h.json")

    out_dir = os.path.join(cfg.out_root, f"W{cfg.W}_H{cfg.H}_two_stage_bal_skip_stage1")
    os.makedirs(out_dir, exist_ok=True)

    # datasets
    ds_train = NPZFileDataset(man_train, name2id, max_files=cfg.max_train_files, allow_pickle=True)
    ds_val   = NPZFileDataset(man_val,   name2id, max_files=cfg.max_val_files, allow_pickle=True)
    print("train files:", len(ds_train), "| val files:", len(ds_val))

    train_counts = summarize_scan(ds_train, "scan-train")
    _ = summarize_scan(ds_val,   "scan-val")

    X0, _ = ds_train[0]
    T_fixed = int(X0.shape[1])
    F = int(X0.shape[2])
    print("T_fixed:", T_fixed, "| F:", F)

    # -------------------------
    # Train loader (Stage1)
    # -------------------------
    sampler_s1 = make_file_weighted_sampler(
        ds_train, wX=cfg.wX, wM=cfg.wM, wC=cfg.wC, wB=cfg.wB, wQ=cfg.wQ
    )
    dl_train_s1 = DataLoader(
        ds_train,
        batch_size=cfg.batch_files,
        sampler=sampler_s1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=lambda b: collate_take_random_samples_fixedT(
            b, per_file=cfg.per_file, T_fixed=T_fixed, clamp_abs=cfg.clamp_abs
        ),
    )

    # -------------------------
    # Stage2 loader (same ds_train, different sampler)
    # -------------------------
    sampler_s2 = make_file_weighted_sampler(
        ds_train, wX=cfg.s2_wX, wM=cfg.s2_wM, wC=cfg.s2_wC, wB=cfg.s2_wB, wQ=cfg.s2_wQ
    )
    dl_train_s2 = DataLoader(
        ds_train,
        batch_size=cfg.batch_files,
        sampler=sampler_s2,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        #collate_fn=lambda b: collate_take_random_samples_fixedT(
        #    b, per_file=cfg.per_file, T_fixed=T_fixed, clamp_abs=cfg.clamp_abs
        collate_fn=lambda b: collate_stage2_quota_fixedT(
        b,
        per_file=cfg.per_file,
        T_fixed=T_fixed,
        clamp_abs=cfg.clamp_abs,
        flare_frac=cfg.s2_flare_frac,
        flare_quota=cfg.s2_flare_quota,
        x_min_per_file=cfg.s2_x_min_per_file,
        x_max_per_file=cfg.s2_x_max_per_file,
        quiet_id=name2id["quiet"],
        B_id=name2id["B"],
        C_id=name2id["C"],
        M_id=name2id["M"],
        X_id=name2id["X"],
        #flare_frac=0.80,
        #flare_quota={
        #    name2id["B"]: 0.25,
        #    name2id["C"]: 0.45,
        #    name2id["M"]: 0.20,
        #    name2id["X"]: 0.10,
        #},
        #x_max_per_file=4,
        ),
    )

    # -------- fixed balanced VAL/TEST --------
    if cfg.fixed_min_val is None:
        cfg.fixed_min_val = {"X":200,"M":200,"C":800,"B":200,"quiet":8000}
    if cfg.fixed_min_test is None:
        cfg.fixed_min_test = {"X":200,"M":200,"C":800,"B":200,"quiet":12000}

    val_fixed = os.path.join(out_dir, f"val_fixed_bal_W{cfg.W}_H{cfg.H}.npz")
    if not os.path.exists(val_fixed):
        print("Creating balanced val_fixed:", val_fixed)
        make_fixed_balanced(
            ds_val, val_fixed,
            T_fixed=T_fixed,
            per_class_min=cfg.fixed_min_val,
            seed=cfg.fixed_seed,
            clamp_abs=cfg.clamp_abs
        )
    else:
        print("val_fixed exists:", val_fixed)

    dl_val_fixed = make_fixed_loader(val_fixed, batch_size=cfg.fixed_batch_size, num_workers=0)

    if os.path.exists(man_test):
        ds_test = NPZFileDataset(man_test, name2id, max_files=cfg.max_test_files, allow_pickle=True)
        _ = summarize_scan(ds_test, "scan-test")

        test_fixed = os.path.join(out_dir, f"test_fixed_bal_W{cfg.W}_H{cfg.H}.npz")
        if not os.path.exists(test_fixed):
            print("Creating balanced test_fixed:", test_fixed)
            make_fixed_balanced(
                ds_test, test_fixed,
                T_fixed=T_fixed,
                per_class_min=cfg.fixed_min_test,
                seed=cfg.fixed_seed + 1,
                clamp_abs=cfg.clamp_abs
            )
        else:
            print("test_fixed exists:", test_fixed)

        dl_test_fixed = make_fixed_loader(test_fixed, batch_size=cfg.fixed_batch_size, num_workers=0)
    else:
        test_fixed = None
        dl_test_fixed = None
        print("No test manifest found:", man_test)

    # sanity
    X_tmp, y_tmp = next(iter(dl_train_s1))
    print("sanity batch:", X_tmp.shape, y_tmp.shape)

    # ==========================================================
    # Stage 1: binary flare (quiet vs flare)
    # ==========================================================
    model_s1 = SimpleTCN(num_features=F, num_classes=2, hidden=cfg.hidden).to(device)

    # ✅ Stage1 loss (as requested)
    if cfg.s1_use_focal:
        loss_s1 = FocalLossBinary(gamma=cfg.s1_focal_gamma, pos_weight=cfg.s1_pos_weight)
    else:
        loss_s1 = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, float(cfg.s1_pos_weight)], device=device)
        )

    opt_s1 = torch.optim.Adam(model_s1.parameters(), lr=cfg.lr)

    if cfg.skip_stage1_train:
        print("Skipping Stage1 training (using loaded weights).")
        print("Stage1 loaded. First layer weight mean:",
        float(next(model_s1.parameters()).detach().abs().mean().cpu()))
        probs1_va, y_va = predict_proba(model_s1, dl_val_fixed, device=device)
        p_flare_va = probs1_va[:, 1]
        print("Stage1 p_flare stats on VAL:",
            "min", float(p_flare_va.min()),
            "mean", float(p_flare_va.mean()),
            "max", float(p_flare_va.max()))
    else:
        for ep in range(1, cfg.s1_epochs + 1):
            t0 = time.perf_counter()
            model_s1.train()
            total, n = 0.0, 0

            for X, y in dl_train_s1:
                yb = (y != IDX_Q).long()
                X = X.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                opt_s1.zero_grad(set_to_none=True)
                logits = model_s1(X)
                loss = loss_s1(logits, yb)

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                loss.backward()
                opt_s1.step()

                total += float(loss.item()) * X.size(0)
                n += X.size(0)

            dt = time.perf_counter() - t0

            probs1_va, y_va = predict_proba(model_s1, dl_val_fixed, device=device)
            p_flare_va = probs1_va[:, 1]
            y_bin = (y_va != IDX_Q).astype(np.int64)
            pred_bin = (p_flare_va >= 0.5).astype(np.int64)
            f1b = f1_score(y_bin, pred_bin, average="binary", zero_division=0)
            print(f"[S1 Ep {ep}] loss={total/max(n,1):.4f} | val_f1_bin@0.5={f1b:.4f} | {dt:.1f}s")

        save_checkpoint(out_dir, "stage1_last", model_s1, opt_s1, extra={"cfg": asdict(cfg)})

    # ==========================================================
    # Stage 2: severity (B/C/M/X) trained on flare samples only
    # ==========================================================
    model_s2 = SimpleTCN(num_features=F, num_classes=4, hidden=cfg.hidden).to(device)

    # ✅ Stage2 loss (as requested)
    if cfg.s2_alpha_mode != "manual":
        alpha_arr = compute_alpha_from_counts(train_counts, SEV_CLASSES, mode=cfg.s2_alpha_mode, power=cfg.s2_alpha_pow)
    else:
        alpha_arr = np.array(list(cfg.s2_alpha), dtype=np.float32)
    alpha = torch.tensor(alpha_arr, dtype=torch.float32, device=device)  # [B,C,M,X]
    if cfg.s2_use_focal:
        loss_s2 = FocalLossMulticlass(gamma=cfg.s2_focal_gamma, alpha=alpha)
    else:
        loss_s2 = nn.CrossEntropyLoss(weight=alpha)

    opt_s2 = torch.optim.Adam(model_s2.parameters(), lr=cfg.lr)

    for ep in range(1, cfg.s2_epochs + 1):
        t0 = time.perf_counter()
        model_s2.train()
        total, n = 0.0, 0

        total, n = 0.0, 0
        steps = 0
        printed = 0   # נדפיס רק 2–3 פעמים בכל epoch
        
        # ✅ IMPORTANT: train Stage2 on dl_train_s2 (stage2 sampler)
        for batch_i, (X, y) in enumerate(dl_train_s2):
            if batch_i == 0:
                print("S2 first batch arrived. X shape:", tuple(X.shape), "y shape:", tuple(y.shape))

            mask = (y != IDX_Q)
            if mask.sum().item() == 0:
                continue

            X2 = X[mask]
            y2 = (y[mask] - 1).long()  # 1..4 -> 0..3
            # ✅ sanity: y2 חייב להיות 0..3
            if (y2.min().item() < 0) or (y2.max().item() > 3):
                print("S2: BAD y2 range:", int(y2.min()), int(y2.max()))
                continue

            X2 = X2.to(device, non_blocking=True)
            y2 = y2.to(device, non_blocking=True)


            X2 = torch.nan_to_num(X2, nan=0.0, posinf=cfg.clamp_abs, neginf=-cfg.clamp_abs) #New
            X2 = torch.clamp(X2, -cfg.clamp_abs, cfg.clamp_abs)                              #New

            opt_s2.zero_grad(set_to_none=True)
            logits = model_s2(X2)
            if not torch.isfinite(logits).all():
                if printed < 3:
                    print("S2: logits has NaN/Inf, skipping batch")
                    printed += 1
                continue

            loss = loss_s2(logits, y2)

            
            
            
            
            if printed < 3:
                with torch.no_grad():
                    raw5 = torch.bincount(y, minlength=5).detach().cpu().tolist()      # [quiet,B,C,M,X]
                    y2_4 = torch.bincount(y2.detach().cpu(), minlength=4).tolist()    # [B,C,M,X]
                    pred = logits.argmax(dim=1).detach().cpu()
                    p_4  = torch.bincount(pred, minlength=4).tolist()

                print("S2 train batch bincount (raw5):", raw5)
                print("S2 train batch bincount (y2_4):", y2_4)
                print("S2 train pred  bincount (p_4): ", p_4)
                print("S2 step loss:", float(loss.detach().cpu()))
                printed += 1

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_s2.parameters(), 1.0)  # New

            opt_s2.step()

            total += float(loss.item()) * X2.size(0)
            n += X2.size(0)
            steps +=1

            

        dt = time.perf_counter() - t0
        print(f"[S2 Ep {ep}] loss={total/max(n,1):.4f} | {dt:.1f}s")
        print(f"[S2 Ep {ep}] steps={steps} loss={total/max(n,1):.10f} | {dt:.1f}s")
        

    save_checkpoint(out_dir, "stage2_last", model_s2, opt_s2, extra={"cfg": asdict(cfg)})

    # ==========================================================
    # Evaluate two-stage on VAL_FIXED_BAL
    # ==========================================================
    probs1_va, y_va = predict_proba(model_s1, dl_val_fixed, device=device)
    p_flare_va = probs1_va[:, 1]
    probs2_va, _ = predict_proba(model_s2, dl_val_fixed, device=device)

    gate_info = {}
    if cfg.gate_mode in ("quiet_quantile", "soft_quantile"):
        mask_q = (y_va == IDX_Q)
        q_scores = p_flare_va[mask_q]
        q = float(cfg.gate_quantile)
        t_flare = float(np.quantile(q_scores, q)) if q_scores.size > 0 else 0.5
        gate_info = {"method": "quiet_quantile", "q": q}
    elif cfg.gate_mode in ("grid_macro_f1", "grid"):
        t_flare, gate_info = tune_t_flare(
            p_flare_va, probs2_va, y_va,
            grid=cfg.t_flare_grid,
            objective=cfg.gate_objective,
            recall_weight=cfg.gate_recall_weight,
            recall_classes=cfg.gate_recall_classes,
        )
        gate_info["method"] = "grid_tuned"
    elif cfg.gate_mode == "no_gate":
        t_flare = 0.0
        gate_info = {"method": "no_gate"}
    else:
        raise ValueError(f"Unknown gate_mode: {cfg.gate_mode}")

    t_sev = None
    sev_info = None
    if cfg.use_sev_thresholds:
        t_sev, sev_info = tune_sev_thresholds(
            p_flare_va, probs2_va, y_va,
            t_flare=t_flare,
            grid=cfg.sev_t_grid,
            iters=cfg.sev_t_iters,
        )
        print("t_sev:", {c: float(v) for c, v in zip(SEV_CLASSES, t_sev)})
        print("sev_info:", sev_info)

    pred_va = two_stage_predict(p_flare_va, probs2_va, t_flare=t_flare, t_sev=t_sev)
    mf1_va = f1_score(y_va, pred_va, average="macro", labels=list(range(K)))
    print("\n=== VAL_FIXED_BAL (two-stage) ===")
    print("gate_mode:", cfg.gate_mode, "| t_flare:", t_flare)
    print("gate_info:", gate_info)
    print(classification_report(y_va, pred_va, labels=list(range(K)), target_names=classes, zero_division=0))

    thresholds = {"two_stage": {"t_flare": float(t_flare), "gate_mode": cfg.gate_mode, "gate_info": gate_info}}
    if t_sev is not None:
        thresholds["two_stage"]["t_sev"] = [float(x) for x in t_sev]
        thresholds["two_stage"]["sev_info"] = sev_info
    np.save(os.path.join(out_dir, "val_p_flare.npy"), p_flare_va.astype(np.float32))
    np.save(os.path.join(out_dir, "val_p_sev.npy"), probs2_va.astype(np.float32))
    np.save(os.path.join(out_dir, "val_pred_two_stage.npy"), pred_va.astype(np.int64))

    cm_va = confusion_matrix(y_va, pred_va, labels=list(range(K)))
    val_metrics = {
        "split": "val_fixed_bal",
        "macro_f1_two_stage": float(mf1_va),
        "recall_per_class_two_stage": per_class_recall(cm_va),
        "confusion_matrix_two_stage": cm_va.tolist(),
        "thresholds": thresholds,
        "dist": dict(Counter(y_va.tolist())),
    }
    with open(os.path.join(out_dir, "val_fixed_bal_metrics_two_stage.json"), "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=2)

    # ==========================================================
    # Evaluate on TEST_FIXED_BAL using same t_flare
    # ==========================================================
    if dl_test_fixed is not None:
        probs1_te, y_te = predict_proba(model_s1, dl_test_fixed, device=device)
        p_flare_te = probs1_te[:, 1]
        probs2_te, _ = predict_proba(model_s2, dl_test_fixed, device=device)

        # debug: M drop
        mask_M = (y_te == IDX_M)
        print("M count:", int(mask_M.sum()))
        if mask_M.sum() > 0:
            print("M mean p_flare:", float(p_flare_te[mask_M].mean()))
            print("M % passed t:", float((p_flare_te[mask_M] >= t_flare).mean()))

            passed_M = mask_M & (p_flare_te >= t_flare)
            if passed_M.sum() > 0:
                sev_pred = predict_severity_with_thresholds(probs2_te[passed_M], t_sev=t_sev)
                print("Stage2 on M (passed) distribution:", dict(Counter(sev_pred.tolist())))

        pred_te = two_stage_predict(p_flare_te, probs2_te, t_flare=float(t_flare), t_sev=t_sev)
        mf1_te = f1_score(y_te, pred_te, average="macro", labels=list(range(K)))

        print("\n=== TEST_FIXED_BAL (two-stage) ===")
        print("t_flare (from VAL):", t_flare)
        print(classification_report(y_te, pred_te, labels=list(range(K)), target_names=classes, zero_division=0))

        np.save(os.path.join(out_dir, "test_p_flare.npy"), p_flare_te.astype(np.float32))
        np.save(os.path.join(out_dir, "test_p_sev.npy"), probs2_te.astype(np.float32))
        np.save(os.path.join(out_dir, "test_pred_two_stage.npy"), pred_te.astype(np.int64))

        cm_te = confusion_matrix(y_te, pred_te, labels=list(range(K)))
        test_metrics = {
            "split": "test_fixed_bal",
            "macro_f1_two_stage": float(mf1_te),
            "recall_per_class_two_stage": per_class_recall(cm_te),
            "confusion_matrix_two_stage": cm_te.tolist(),
            "thresholds": thresholds,
            "dist": dict(Counter(y_te.tolist())),
        }
        with open(os.path.join(out_dir, "test_fixed_bal_metrics_two_stage.json"), "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2)

    return {
        "out_dir": out_dir,
        "val_fixed_bal_path": val_fixed,
        "test_fixed_bal_path": test_fixed,
        "t_flare": float(t_flare),
        "thresholds": thresholds,
    }

def run_stage1_only(cfg: ExperimentCfg,
                    eval_balanced: bool = True,
                    eval_natural: bool = True,
                    eval_test: bool = False,
                    t_eval: float = 0.5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    set_seed(cfg.seed)

    man_dir = os.path.join(cfg.root4, "manifests_by_group")
    man_train = os.path.join(man_dir, f"manifest_train_W{cfg.W}h_H{cfg.H}h.json")
    man_val   = os.path.join(man_dir, f"manifest_val_W{cfg.W}h_H{cfg.H}h.json")
    man_test  = os.path.join(man_dir, f"manifest_test_W{cfg.W}h_H{cfg.H}h.json")

    out_dir = os.path.join(cfg.out_root, f"W{cfg.W}_H{cfg.H}_two_stage_bal")
    os.makedirs(out_dir, exist_ok=True)

    # datasets
    ds_train = NPZFileDataset(man_train, name2id, max_files=cfg.max_train_files, allow_pickle=True)
    ds_val   = NPZFileDataset(man_val,   name2id, max_files=cfg.max_val_files, allow_pickle=True)
    print("train files:", len(ds_train), "| val files:", len(ds_val))

    _ = summarize_scan(ds_train, "scan-train")
    _ = summarize_scan(ds_val,   "scan-val")

    X0, _ = ds_train[0]
    T_fixed = int(X0.shape[1])
    F = int(X0.shape[2])

    # Stage1 train loader
    sampler_s1 = make_file_weighted_sampler(
        ds_train, wX=cfg.wX, wM=cfg.wM, wC=cfg.wC, wB=cfg.wB, wQ=cfg.wQ
    )
    dl_train_s1 = DataLoader(
        ds_train,
        batch_size=cfg.batch_files,
        sampler=sampler_s1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=lambda b: collate_take_random_samples_fixedT(
            b, per_file=cfg.per_file, T_fixed=T_fixed, clamp_abs=cfg.clamp_abs
        ),
    )

    # balanced VAL (val_fixed)
    if cfg.fixed_min_val is None:
        cfg.fixed_min_val = {"X":200,"M":200,"C":800,"B":200,"quiet":8000}
    val_fixed = os.path.join(out_dir, f"val_fixed_bal_W{cfg.W}_H{cfg.H}.npz")
    if not os.path.exists(val_fixed):
        make_fixed_balanced(
            ds_val, val_fixed,
            T_fixed=T_fixed,
            per_class_min=cfg.fixed_min_val,
            seed=cfg.fixed_seed,
            clamp_abs=cfg.clamp_abs
        )
    dl_val_fixed = make_fixed_loader(val_fixed, batch_size=cfg.fixed_batch_size, num_workers=0)

    # natural VAL loader (original distribution)
    dl_val_nat = DataLoader(
        ds_val,
        batch_size=cfg.batch_files,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=lambda b: collate_take_random_samples_fixedT(
            b, per_file=cfg.per_file, T_fixed=T_fixed, clamp_abs=cfg.clamp_abs
        ),
    )

    # Stage1 model
    model_s1 = SimpleTCN(num_features=F, num_classes=2, hidden=cfg.hidden).to(device)

    # loss
    if cfg.s1_use_focal:
        loss_s1 = FocalLossBinary(gamma=cfg.s1_focal_gamma, pos_weight=cfg.s1_pos_weight)
    else:
        loss_s1 = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, float(cfg.s1_pos_weight)], device=device)
        )

    opt_s1 = torch.optim.Adam(model_s1.parameters(), lr=cfg.lr)

    # train
    for ep in range(1, cfg.s1_epochs + 1):
        t0 = time.perf_counter()
        model_s1.train()
        total, n = 0.0, 0

        for X, y in dl_train_s1:
            yb = (y != IDX_Q).long()
            X = X.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt_s1.zero_grad(set_to_none=True)
            logits = model_s1(X)
            loss = loss_s1(logits, yb)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            opt_s1.step()

            total += float(loss.item()) * X.size(0)
            n += X.size(0)

        dt = time.perf_counter() - t0
        print(f"[S1 Ep {ep}] loss={total/max(n,1):.4f} | {dt:.1f}s")

    save_checkpoint(out_dir, "stage1_last", model_s1, opt_s1, extra={"cfg": asdict(cfg)})

    # ---- evaluation helper ----
    def eval_stage1(loader, split_name):
        probs, y_true = predict_proba(model_s1, loader, device=device)
        p_flare = probs[:, 1]
        y_bin = (y_true != IDX_Q).astype(np.int64)
        pred_bin = (p_flare >= t_eval).astype(np.int64)

        cm = confusion_matrix(y_bin, pred_bin, labels=[0,1])
        f1b = f1_score(y_bin, pred_bin, average="binary", zero_division=0)
        prec = precision_score(y_bin, pred_bin, zero_division=0)
        rec  = recall_score(y_bin, pred_bin, zero_division=0)

        print(f"\n=== Stage1 {split_name} ===")
        print("F1:", f1b, "Precision:", prec, "Recall:", rec)
        print("Confusion matrix [[TN,FP],[FN,TP]]:\n", cm)
        print(classification_report(y_bin, pred_bin, target_names=["quiet","flare"], zero_division=0))
        return {"f1": float(f1b), "precision": float(prec), "recall": float(rec), "confusion_matrix": cm.tolist()}

    # run evals
    if eval_balanced:
        eval_stage1(dl_val_fixed, "VAL_FIXED (balanced)")
    if eval_natural:
        eval_stage1(dl_val_nat, "VAL_NATURAL")
        # Sweep thresholds on VAL_NATURAL
        probs_nat, y_nat = predict_proba(model_s1, dl_val_nat, device=device)
        p_flare_nat = probs_nat[:, 1]

        thresholds = [0.05,0.075,0.1,0.125,0.15,0.2,0.25,0.3,0.35,0.4,0.5]
        rows, best = sweep_stage1_thresholds(p_flare_nat, y_nat, thresholds, min_precision=0.10)
        output = {"thresholds": thresholds,
                    "results": [{"t": t, "prec": p, "rec": r, "f1": f1} for t, p, r, f1 in rows],
                    "best": best,
                    "min_precision": 0.10
                }

        print(output)
        # optional save
        out_path = os.path.join(out_dir, "stage1_threshold_sweep_val_natural.json")
        with open(out_path, "w") as f:
            json.dump({
                "thresholds": thresholds,
                "results": [{"t": t, "prec": p, "rec": r, "f1": f1} for t,p,r,f1 in rows],
                "best": best,
                "min_precision": 0.10
            }, f, indent=2)
        print("Saved sweep to:", out_path)

    if eval_test and os.path.exists(man_test):
        ds_test = NPZFileDataset(man_test, name2id, max_files=cfg.max_test_files, allow_pickle=True)
        test_fixed = os.path.join(out_dir, f"test_fixed_bal_W{cfg.W}_H{cfg.H}.npz")
        if os.path.exists(test_fixed):
            dl_test_fixed = make_fixed_loader(test_fixed, batch_size=cfg.fixed_batch_size, num_workers=0)
            eval_stage1(dl_test_fixed, "TEST_FIXED")

def run_one_experiment_skip_stage1_fixed(cfg: ExperimentCfgSkipStage1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    set_seed(cfg.seed)

    man_dir = os.path.join(cfg.root4, "manifests_by_group")
    man_train = os.path.join(man_dir, f"manifest_train_W{cfg.W}h_H{cfg.H}h.json")
    man_val   = os.path.join(man_dir, f"manifest_val_W{cfg.W}h_H{cfg.H}h.json")
    man_test  = os.path.join(man_dir, f"manifest_test_W{cfg.W}h_H{cfg.H}h.json")

    out_dir = os.path.join(cfg.out_root, f"W{cfg.W}_H{cfg.H}_two_stage_bal_skip_stage1")
    os.makedirs(out_dir, exist_ok=True)

    # datasets
    ds_train = NPZFileDataset(man_train, name2id, max_files=cfg.max_train_files, allow_pickle=True)
    ds_val   = NPZFileDataset(man_val,   name2id, max_files=cfg.max_val_files, allow_pickle=True)
    print("train files:", len(ds_train), "| val files:", len(ds_val))

    train_counts = summarize_scan(ds_train, "scan-train")
    _ = summarize_scan(ds_val,   "scan-val")

    X0, _ = ds_train[0]
    T_fixed = int(X0.shape[1])
    F = int(X0.shape[2])
    print("T_fixed:", T_fixed, "| F:", F)

    # -------------------------
    # Stage2 loader
    # -------------------------
    sampler_s2 = make_file_weighted_sampler(
        ds_train, wX=cfg.s2_wX, wM=cfg.s2_wM, wC=cfg.s2_wC, wB=cfg.s2_wB, wQ=cfg.s2_wQ
    )
    #dl_train_s2 = DataLoader(
    #    ds_train,
    #    batch_size=cfg.batch_files,
    #    sampler=sampler_s2,
    #    shuffle=False,
    #    num_workers=0,
    #    pin_memory=True,
    #    collate_fn=lambda b: collate_stage2_quota_fixedT(
    #        b,
    #        per_file=cfg.per_file,
    #        T_fixed=T_fixed,
    #        clamp_abs=cfg.clamp_abs,
    #        flare_frac=cfg.s2_flare_frac,
    #        flare_quota=cfg.s2_flare_quota,
    #        x_min_per_file=cfg.s2_x_min_per_file,
    #        x_max_per_file=cfg.s2_x_max_per_file,
    #        quiet_id=name2id["quiet"],
    #        B_id=name2id["B"],
    #        C_id=name2id["C"],
    #        M_id=name2id["M"],
    #        X_id=name2id["X"],
    #    ),
    #)

    dl_train_s2 = DataLoader(
        ds_train,
        batch_size=cfg.batch_files,
        sampler=sampler_s2,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=lambda b: collate_stage2_quota_fixedT_hardquiet(
            b,
            per_file=cfg.per_file,
            T_fixed=T_fixed,
            clamp_abs=cfg.clamp_abs,
            flare_frac=cfg.s2_flare_frac,
            flare_quota=cfg.s2_flare_quota,
            x_min_per_file=cfg.s2_x_min_per_file,
            x_max_per_file=cfg.s2_x_max_per_file,
            quiet_id=name2id["quiet"],
            B_id=name2id["B"],
            C_id=name2id["C"],
            M_id=name2id["M"],
            X_id=name2id["X"],
            model_s1=model_s1,
            device=device,
            hardquiet_mult=8,
            hardquiet_mode="topk",      # or "threshold"
            t_flare_train=0.26,         # used only when mode="threshold"
            log_every=100,
        ),
    )

    # -------- fixed balanced VAL/TEST --------
    if cfg.fixed_min_val is None:
        cfg.fixed_min_val = {"X":200,"M":200,"C":800,"B":200,"quiet":8000}
    if cfg.fixed_min_test is None:
        cfg.fixed_min_test = {"X":200,"M":200,"C":800,"B":200,"quiet":12000}

    val_fixed = os.path.join(out_dir, f"val_fixed_bal_W{cfg.W}_H{cfg.H}.npz")
    if not os.path.exists(val_fixed):
        print("Creating balanced val_fixed:", val_fixed)
        make_fixed_balanced(
            ds_val, val_fixed,
            T_fixed=T_fixed,
            per_class_min=cfg.fixed_min_val,
            seed=cfg.fixed_seed,
            clamp_abs=cfg.clamp_abs
        )
    else:
        print("val_fixed exists:", val_fixed)

    dl_val_fixed = make_fixed_loader(val_fixed, batch_size=cfg.fixed_batch_size, num_workers=0)

    if os.path.exists(man_test):
        ds_test = NPZFileDataset(man_test, name2id, max_files=cfg.max_test_files, allow_pickle=True)
        _ = summarize_scan(ds_test, "scan-test")

        test_fixed = os.path.join(out_dir, f"test_fixed_bal_W{cfg.W}_H{cfg.H}.npz")
        if not os.path.exists(test_fixed):
            print("Creating balanced test_fixed:", test_fixed)
            make_fixed_balanced(
                ds_test, test_fixed,
                T_fixed=T_fixed,
                per_class_min=cfg.fixed_min_test,
                seed=cfg.fixed_seed + 1,
                clamp_abs=cfg.clamp_abs
            )
        else:
            print("test_fixed exists:", test_fixed)

        dl_test_fixed = make_fixed_loader(test_fixed, batch_size=cfg.fixed_batch_size, num_workers=0)
    else:
        test_fixed = None
        dl_test_fixed = None
        print("No test manifest found:", man_test)

    # ==========================================================
    # Stage 1: load checkpoint (skip training)
    # ==========================================================
    model_s1 = SimpleTCN(num_features=F, num_classes=2, hidden=cfg.hidden).to(device)

    if cfg.skip_stage1_train:
        if not cfg.s1_resume_path:
            raise ValueError("skip_stage1_train=True but s1_resume_path is empty.")
        ckpt = torch.load(cfg.s1_resume_path, map_location=device)
        model_s1.load_state_dict(ckpt["model_state"])
        model_s1.eval()
        print("Loaded Stage1 from:", cfg.s1_resume_path)
    else:
        raise ValueError("This function is for skip_stage1_train=True only.")

    # ==========================================================
    # Stage 2: 5-class (quiet+B+C+M+X)
    # ==========================================================
    model_s2 = SimpleTCN(num_features=F, num_classes=5, hidden=cfg.hidden).to(device)
    #quiet_w = 0.2  # adjust if needed

    if cfg.s2_alpha_mode != "manual":
        alpha_sev = compute_alpha_from_counts(train_counts, SEV_CLASSES, mode=cfg.s2_alpha_mode, power=cfg.s2_alpha_pow)
    else:
        alpha_sev = np.array(list(cfg.s2_alpha), dtype=np.float32)

    alpha_arr = np.concatenate(([cfg.quiet_w], alpha_sev)).astype(np.float32)
    alpha = torch.tensor(alpha_arr, dtype=torch.float32, device=device)

    if cfg.s2_use_focal:
        loss_s2 = FocalLossMulticlass(gamma=cfg.s2_focal_gamma, alpha=alpha)
    else:
        loss_s2 = nn.CrossEntropyLoss(weight=alpha)

    opt_s2 = torch.optim.Adam(model_s2.parameters(), lr=cfg.lr)

    for ep in range(1, cfg.s2_epochs + 1):
        t0 = time.perf_counter()
        model_s2.train()
        total, n = 0.0, 0

        for X, y in dl_train_s2:
            X2 = X.to(device, non_blocking=True)
            y2 = y.long().to(device, non_blocking=True)  # 0..4 (quiet,B,C,M,X)

            opt_s2.zero_grad(set_to_none=True)
            logits = model_s2(X2)

            if not torch.isfinite(logits).all():
                continue

            loss = loss_s2(logits, y2)
            if not torch.isfinite(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_s2.parameters(), 1.0)
            opt_s2.step()

            total += float(loss.item()) * X2.size(0)
            n += X2.size(0)

        dt = time.perf_counter() - t0
        print(f"[S2 Ep {ep}] loss={total/max(n,1):.4f} | {dt:.1f}s")

    if cfg.out_dir_changes == True:
        out_dir = out_dir + str(cfg.t_flare_grid)
    print(f"out_dir: {out_dir}")
    save_checkpoint(out_dir, "stage2_last", model_s2, opt_s2, extra={"cfg": asdict(cfg)})

    # ==========================================================
    # Evaluate on VAL_FIXED_BAL
    # ==========================================================
    probs1_va, y_va = predict_proba(model_s1, dl_val_fixed, device=device)
    p_flare_va = probs1_va[:, 1]
    probs2_va, _ = predict_proba(model_s2, dl_val_fixed, device=device)

    gate_info = {}
    if cfg.gate_mode in ("quiet_quantile", "soft_quantile"):
        mask_q = (y_va == IDX_Q)
        q_scores = p_flare_va[mask_q]
        q = float(cfg.gate_quantile)
        t_flare = float(np.quantile(q_scores, q)) if q_scores.size > 0 else 0.5
        gate_info = {"method": "quiet_quantile", "q": q}
    elif cfg.gate_mode in ("grid_macro_f1", "grid"):
        t_flare, gate_info = tune_t_flare_quiet(
            p_flare_va, probs2_va, y_va,
            grid=cfg.t_flare_grid,
            objective=cfg.gate_objective,
            recall_weight=cfg.gate_recall_weight,
            recall_classes=cfg.gate_recall_classes,
        )
        gate_info["method"] = "grid_tuned"
    elif cfg.gate_mode == "no_gate":
        t_flare = 0.0
        gate_info = {"method": "no_gate"}
    else:
        raise ValueError(f"Unknown gate_mode: {cfg.gate_mode}")

    t_sev = None
    sev_info = None
    if cfg.manual_t_sev == True: 
        t_sev = [0.8, 0.1, 0.2, 0.1]  # [B,C,M,X] fixed baseline
        sev_info = {"method": "manual_fixed", "t_sev": t_sev}
        
        # Optional sweep only for M threshold
        for tM in [0.10, 0.15, 0.20, 0.25, 0.30]:
            t_try = [0.8, 0.1, tM, 0.1]
            pred_try = two_stage_predict_with_quiet(p_flare_va, probs2_va, t_flare=t_flare, t_sev=t_try)
            mf1_try = f1_score(y_va, pred_try, average="macro", labels=list(range(K)))
            rec_try = recall_score(y_va, pred_try, labels=[3], average=None, zero_division=0)[0]  # M recall
            print(f"[VAL sweep] tM={tM:.2f} | macro_f1={mf1_try:.4f} | M_recall={rec_try:.4f}")

    else:
        if cfg.use_sev_thresholds:
            best = tune_sev_thresholds_with_constraints(
                p_flare_va, probs2_va, y_va, t_flare,
                grid=cfg.sev_t_grid,
                min_prec_B=0.05,
                min_rec_M=0.05,
                min_rec_X=0.05
            )
            if best is not None:
                t_sev = best["t_sev"]
                sev_info = best

    pred_va = two_stage_predict_with_quiet(p_flare_va, probs2_va, t_flare=t_flare, t_sev=t_sev)
    mf1_va = f1_score(y_va, pred_va, average="macro", labels=list(range(K)))

    print("\n=== VAL_FIXED_BAL (two-stage) ===")
    print("gate_mode:", cfg.gate_mode, "| t_flare:", t_flare)
    print("gate_info:", gate_info)
    print(classification_report(y_va, pred_va, labels=list(range(K)), target_names=classes, zero_division=0))

    thresholds = {"two_stage": {"t_flare": float(t_flare), "gate_mode": cfg.gate_mode, "gate_info": gate_info}}
    if t_sev is not None:
        thresholds["two_stage"]["t_sev"] = [float(x) for x in t_sev]
        thresholds["two_stage"]["sev_info"] = sev_info

    np.save(os.path.join(out_dir, "val_p_flare.npy"), p_flare_va.astype(np.float32))
    np.save(os.path.join(out_dir, "val_p_sev.npy"), probs2_va.astype(np.float32))
    np.save(os.path.join(out_dir, "val_pred_two_stage.npy"), pred_va.astype(np.int64))

    cm_va = confusion_matrix(y_va, pred_va, labels=list(range(K)))
    val_metrics = {
        "split": "val_fixed_bal",
        "macro_f1_two_stage": float(mf1_va),
        "recall_per_class_two_stage": per_class_recall(cm_va),
        "confusion_matrix_two_stage": cm_va.tolist(),
        "thresholds": thresholds,
        "dist": dict(Counter(y_va.tolist())),
    }
    with open(os.path.join(out_dir, "val_fixed_bal_metrics_two_stage.json"), "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=2)

    # ==========================================================
    # Evaluate on TEST_FIXED_BAL
    # ==========================================================
    if dl_test_fixed is not None:
        probs1_te, y_te = predict_proba(model_s1, dl_test_fixed, device=device)
        p_flare_te = probs1_te[:, 1]
        probs2_te, _ = predict_proba(model_s2, dl_test_fixed, device=device)

        pred_te = two_stage_predict_with_quiet(p_flare_te, probs2_te, t_flare=float(t_flare), t_sev=t_sev)
        mf1_te = f1_score(y_te, pred_te, average="macro", labels=list(range(K)))

        print("\n=== TEST_FIXED_BAL (two-stage) ===")
        print("t_flare (from VAL):", t_flare)
        print(classification_report(y_te, pred_te, labels=list(range(K)), target_names=classes, zero_division=0))

        cm_te = confusion_matrix(y_te, pred_te, labels=list(range(K)))
        test_metrics = {
            "split": "test_fixed_bal",
            "macro_f1_two_stage": float(mf1_te),
            "recall_per_class_two_stage": per_class_recall(cm_te),
            "confusion_matrix_two_stage": cm_te.tolist(),
            "thresholds": thresholds,
            "dist": dict(Counter(y_te.tolist())),
        }
        with open(os.path.join(out_dir, "test_fixed_bal_metrics_two_stage.json"), "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2)

    return {
        "out_dir": out_dir,
        "val_fixed_bal_path": val_fixed,
        "test_fixed_bal_path": test_fixed,
        "t_flare": float(t_flare),
        "thresholds": thresholds,
    }


def stage1_pass_rate_from_ckpt(ckpt_path, split="val", use_balanced=True, t_flare=0.10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ExperimentCfg(**ckpt["extra"]["cfg"])

    # manifests
    man_dir = os.path.join(cfg.root4, "manifests_by_group")
    man_train = os.path.join(man_dir, f"manifest_train_W{cfg.W}h_H{cfg.H}h.json")
    man_val   = os.path.join(man_dir, f"manifest_val_W{cfg.W}h_H{cfg.H}h.json")
    man_test  = os.path.join(man_dir, f"manifest_test_W{cfg.W}h_H{cfg.H}h.json")

    # get T_fixed/F
    ds_train = NPZFileDataset(man_train, name2id, max_files=cfg.max_train_files, allow_pickle=True)
    X0, _ = ds_train[0]
    T_fixed = int(X0.shape[1])
    F = int(X0.shape[2])

    # loader
    if split == "val":
        if use_balanced:
            val_fixed = os.path.join(cfg.out_root, f"W{cfg.W}_H{cfg.H}_two_stage_bal", f"val_fixed_bal_W{cfg.W}_H{cfg.H}.npz")
            dl = make_fixed_loader(val_fixed, batch_size=cfg.fixed_batch_size, num_workers=0)
        else:
            ds_val = NPZFileDataset(man_val, name2id, max_files=cfg.max_val_files, allow_pickle=True)
            dl = DataLoader(
                ds_val,
                batch_size=cfg.batch_files,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                collate_fn=lambda b: collate_take_random_samples_fixedT(
                    b, per_file=cfg.per_file, T_fixed=T_fixed, clamp_abs=cfg.clamp_abs
                ),
            )
    elif split == "test":
        if not os.path.exists(man_test):
            print("No test manifest found.")
            return
        if use_balanced:
            test_fixed = os.path.join(cfg.out_root, f"W{cfg.W}_H{cfg.H}_two_stage_bal", f"test_fixed_bal_W{cfg.W}_H{cfg.H}.npz")
            dl = make_fixed_loader(test_fixed, batch_size=cfg.fixed_batch_size, num_workers=0)
        else:
            ds_test = NPZFileDataset(man_test, name2id, max_files=cfg.max_test_files, allow_pickle=True)
            dl = DataLoader(
                ds_test,
                batch_size=cfg.batch_files,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                collate_fn=lambda b: collate_take_random_samples_fixedT(
                    b, per_file=cfg.per_file, T_fixed=T_fixed, clamp_abs=cfg.clamp_abs
                ),
            )
    else:
        raise ValueError("split must be 'val' or 'test'")

    # model
    model_s1 = SimpleTCN(num_features=F, num_classes=2, hidden=cfg.hidden).to(device)
    model_s1.load_state_dict(ckpt["model_state"])
    model_s1.eval()

    probs, y_true = predict_proba(model_s1, dl, device=device)
    p_flare = probs[:, 1]

    print(f"\nStage1 pass rates (t_flare={t_flare})")
    for cls_id, cls_name in enumerate(classes):
        m = (y_true == cls_id)
        if m.sum() == 0:
            continue
        pass_rate = (p_flare[m] >= t_flare).mean()
        print(f"{cls_name}: pass {pass_rate:.3f} ({(p_flare[m] >= t_flare).sum()}/{m.sum()})")



def choose_t_for_m_goal_strict(
    p_flare, y_true, thresholds,
    m_goal=0.90, x_goal=0.98, quiet_cap=0.30, min_flare_pass=0.0
):
    rows = []
    for t in sorted(set(float(x) for x in thresholds)):
        passed = (p_flare >= t)
        q = float(passed[y_true == IDX_Q].mean()) if np.any(y_true == IDX_Q) else 0.0
        b = float(passed[y_true == IDX_B].mean()) if np.any(y_true == IDX_B) else 0.0
        c = float(passed[y_true == IDX_C].mean()) if np.any(y_true == IDX_C) else 0.0
        m = float(passed[y_true == IDX_M].mean()) if np.any(y_true == IDX_M) else 0.0
        x = float(passed[y_true == IDX_X].mean()) if np.any(y_true == IDX_X) else 0.0
        flare = float(passed[y_true != IDX_Q].mean()) if np.any(y_true != IDX_Q) else 0.0

        feasible = (m >= m_goal) and (x >= x_goal) and (q <= quiet_cap) and (flare >= min_flare_pass)
        rows.append({
            "t": t, "quiet_pass": q, "flare_pass": flare,
            "B_pass": b, "C_pass": c, "M_pass": m, "X_pass": x, "feasible": feasible
        })

    feasible = [r for r in rows if r["feasible"]]
    if feasible:
        best = max(feasible, key=lambda r: (r["M_pass"], r["X_pass"], -r["quiet_pass"], r["flare_pass"]))
        return best, rows, True, ""

    # fallback 1: enforce quiet cap, maximize M/X
    quiet_ok = [r for r in rows if r["quiet_pass"] <= quiet_cap]
    if quiet_ok:
        best = max(quiet_ok, key=lambda r: (r["M_pass"], r["X_pass"], r["flare_pass"]))
        msg = "No fully feasible threshold; fallback to quiet-capped best."
        return best, rows, False, msg

    # fallback 2: unconstrained score (never crash)
    best = max(rows, key=lambda r: (2.0*r["M_pass"] + 0.5*r["X_pass"] - 0.8*r["quiet_pass"]))
    msg = "No feasible or quiet-capped threshold; fallback to best compromise."
    return best, rows, False, msg


def collate_stage1_quota_fixedT(
    batch,
    per_file: int,
    T_fixed: int,
    quiet_frac: float = 0.45,                      # rest is flare
    flare_quota=(0.10, 0.20, 0.50, 0.20),         # B,C,M,X
    m_min_per_file: int = 1,
    x_min_per_file: int = 0,
    x_max_per_file: int = 4,
    quiet_id: int = 0,
    B_id: int = 1,
    C_id: int = 2,
    M_id: int = 3,
    X_id: int = 4,
    clamp_abs: float = None,
    pad_value: float = 0.0,
):
    def _alloc_counts(total, ids, weights):
        w = np.asarray(weights, dtype=np.float64)
        w = w / max(w.sum(), 1e-12)
        raw = total * w
        base = np.floor(raw).astype(int)
        rem = int(total - base.sum())
        if rem > 0:
            frac_idx = np.argsort(-(raw - base))
            for j in frac_idx[:rem]:
                base[j] += 1
        return {cid: int(k) for cid, k in zip(ids, base)}

    X_out, y_out = [], []
    n_quiet = int(round(per_file * quiet_frac))
    n_quiet = max(0, min(n_quiet, per_file))
    n_flare = per_file - n_quiet

    flare_ids = [B_id, C_id, M_id, X_id]
    if len(flare_quota) != 4:
        raise ValueError("flare_quota must be 4 values for B,C,M,X")

    for (X_file, y_file) in batch:
        y_np = y_file if isinstance(y_file, np.ndarray) else y_file.cpu().numpy()
        N = len(y_np)
        if N == 0:
            continue

        idx_by = {cid: np.flatnonzero(y_np == cid) for cid in [quiet_id, B_id, C_id, M_id, X_id]}
        chosen = []

        # flare allocation
        counts = _alloc_counts(n_flare, flare_ids, flare_quota)

        # enforce M/X mins (only if available)
        if len(idx_by[M_id]) > 0:
            counts[M_id] = max(counts[M_id], min(m_min_per_file, len(idx_by[M_id])))
        if len(idx_by[X_id]) > 0:
            counts[X_id] = max(counts[X_id], min(x_min_per_file, len(idx_by[X_id])))

        # cap X
        counts[X_id] = min(counts[X_id], x_max_per_file)

        # fix total back to n_flare
        def total_counts():
            return counts[B_id] + counts[C_id] + counts[M_id] + counts[X_id]

        # reduce extras (prefer reducing B/C first)
        reduce_order = [B_id, C_id, X_id, M_id]
        while total_counts() > n_flare:
            for cid in reduce_order:
                if counts[cid] > 0 and total_counts() > n_flare:
                    counts[cid] -= 1

        # add deficits to available classes (prefer M then C then B then X)
        add_order = sorted(flare_ids, key=lambda cid: len(idx_by[cid]), reverse=True)
        while total_counts() < n_flare:
            added = False
            for cid in [M_id, C_id, B_id, X_id]:
                if cid in add_order:
                    counts[cid] += 1
                    added = True
                    if total_counts() >= n_flare:
                        break
            if not added:
                break

        # sample flare classes
        for cid in flare_ids:
            k = counts[cid]
            pool = idx_by[cid]
            if k <= 0 or len(pool) == 0:
                continue
            take = np.random.choice(pool, size=k, replace=(len(pool) < k))
            chosen.append(take)

        # sample quiet
        if n_quiet > 0:
            q_pool = idx_by[quiet_id]
            if len(q_pool) == 0:
                q_pool = np.arange(N)
            take_q = np.random.choice(q_pool, size=n_quiet, replace=(len(q_pool) < n_quiet))
            chosen.append(take_q)

        if len(chosen):
            chosen = np.concatenate(chosen)
        else:
            chosen = np.random.choice(np.arange(N), size=min(per_file, N), replace=(N < per_file))

        # if still short, fill randomly
        if len(chosen) < per_file:
            extra = np.random.choice(np.arange(N), size=(per_file - len(chosen)), replace=True)
            chosen = np.concatenate([chosen, extra])

        X_sel = X_file[chosen]
        y_sel = y_np[chosen]

        # crop/pad
        T_here = X_sel.shape[1]
        if T_here > T_fixed:
            X_sel = X_sel[:, :T_fixed, :]
        elif T_here < T_fixed:
            pad = np.full((X_sel.shape[0], T_fixed - T_here, X_sel.shape[2]), pad_value, dtype=X_sel.dtype)
            X_sel = np.concatenate([X_sel, pad], axis=1)

        X_sel = np.nan_to_num(X_sel, nan=0.0, posinf=0.0, neginf=0.0)
        if clamp_abs is not None:
            X_sel = np.clip(X_sel, -clamp_abs, clamp_abs)

        X_out.append(torch.as_tensor(X_sel, dtype=torch.float32))
        y_out.append(torch.as_tensor(y_sel, dtype=torch.long))

    if not X_out:
        raise RuntimeError("collate_stage1_quota_fixedT produced empty batch")

    return torch.cat(X_out, dim=0), torch.cat(y_out, dim=0)


def new_run_stage1_only(
    cfg: ExperimentCfg,
    eval_balanced: bool = True,
    eval_natural: bool = True,
    eval_test: bool = False,
    t_eval: float = 0.5,
    m_goal: float = 0.90,
    x_goal: float = 0.98,
    quiet_cap: float = 0.30,
    min_flare_pass: float = 0.0,
    fail_fast_on_m_goal: bool = True,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    set_seed(cfg.seed)

    man_dir = os.path.join(cfg.root4, "manifests_by_group")
    man_train = os.path.join(man_dir, f"manifest_train_W{cfg.W}h_H{cfg.H}h.json")
    man_val   = os.path.join(man_dir, f"manifest_val_W{cfg.W}h_H{cfg.H}h.json")
    man_test  = os.path.join(man_dir, f"manifest_test_W{cfg.W}h_H{cfg.H}h.json")

    out_dir = os.path.join(cfg.out_root, f"W{cfg.W}_H{cfg.H}_two_stage_bal")
    os.makedirs(out_dir, exist_ok=True)

    ds_train = NPZFileDataset(man_train, name2id, max_files=cfg.max_train_files, allow_pickle=True)
    ds_val   = NPZFileDataset(man_val,   name2id, max_files=cfg.max_val_files, allow_pickle=True)
    print("train files:", len(ds_train), "| val files:", len(ds_val))
    _ = summarize_scan(ds_train, "scan-train")
    _ = summarize_scan(ds_val,   "scan-val")

    X0, _ = ds_train[0]
    T_fixed = int(X0.shape[1])
    F = int(X0.shape[2])

    sampler_s1 = make_file_weighted_sampler(
    ds_train, wX=cfg.wX, wM=cfg.wM, wC=cfg.wC, wB=cfg.wB, wQ=cfg.wQ
    )
    dl_train_s1 = DataLoader(
        ds_train,
        batch_size=cfg.batch_files,
        sampler=sampler_s1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=lambda b: collate_stage1_quota_fixedT(
            b,
            per_file=cfg.per_file,
            T_fixed=T_fixed,
            clamp_abs=cfg.clamp_abs,
            quiet_frac=getattr(cfg, "s1_quiet_frac", 0.45),
            flare_quota=getattr(cfg, "s1_flare_quota", (0.10, 0.20, 0.50, 0.20)),
            m_min_per_file=getattr(cfg, "s1_m_min_per_file", 1),
            x_min_per_file=getattr(cfg, "s1_x_min_per_file", 0),
            x_max_per_file=getattr(cfg, "s1_x_max_per_file", 4),
            quiet_id=IDX_Q, B_id=IDX_B, C_id=IDX_C, M_id=IDX_M, X_id=IDX_X,
        ),
    )

    if cfg.fixed_min_val is None:
        cfg.fixed_min_val = {"X": 200, "M": 200, "C": 800, "B": 200, "quiet": 8000}
    val_fixed = os.path.join(out_dir, f"val_fixed_bal_W{cfg.W}_H{cfg.H}.npz")
    if not os.path.exists(val_fixed):
        make_fixed_balanced(ds_val, val_fixed, T_fixed=T_fixed, per_class_min=cfg.fixed_min_val, seed=cfg.fixed_seed, clamp_abs=cfg.clamp_abs)
    dl_val_fixed = make_fixed_loader(val_fixed, batch_size=cfg.fixed_batch_size, num_workers=0)

    dl_val_nat = DataLoader(
        ds_val, batch_size=cfg.batch_files, shuffle=False, num_workers=0, pin_memory=True,
        collate_fn=lambda b: collate_take_random_samples_fixedT(b, per_file=cfg.per_file, T_fixed=T_fixed, clamp_abs=cfg.clamp_abs),
    )

    model_s1 = SimpleTCN(num_features=F, num_classes=2, hidden=cfg.hidden).to(device)
    if cfg.s1_use_focal:
        loss_s1 = FocalLossBinary(gamma=cfg.s1_focal_gamma, pos_weight=cfg.s1_pos_weight)
    else:
        loss_s1 = nn.CrossEntropyLoss(weight=torch.tensor([1.0, float(cfg.s1_pos_weight)], device=device))
    opt_s1 = torch.optim.Adam(model_s1.parameters(), lr=cfg.lr)

    for ep in range(1, cfg.s1_epochs + 1):
        t0 = time.perf_counter()
        model_s1.train()
        total, n = 0.0, 0

        for bi, (X, y) in enumerate(dl_train_s1):
            if bi == 0:
                print(f"[S1 Ep {ep}] batch quiet/B/C/M/X =", torch.bincount(y, minlength=5).tolist())

            yb = (y != IDX_Q).long()
            X = X.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt_s1.zero_grad(set_to_none=True)
            logits = model_s1(X)
            loss = loss_s1(logits, yb)
            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            opt_s1.step()

            total += float(loss.item()) * X.size(0)
            n += X.size(0)

        print(f"[S1 Ep {ep}] loss={total/max(n,1):.4f} | {time.perf_counter()-t0:.1f}s")


    save_checkpoint(out_dir, "stage1_last", model_s1, opt_s1, extra={"cfg": asdict(cfg)})

    def predict_stage1(loader):
        probs, y_true = predict_proba(model_s1, loader, device=device)
        return probs[:, 1], y_true

    def eval_stage1_from_probs(p_flare, y_true, split_name, t_used):
        y_bin = (y_true != IDX_Q).astype(np.int64)
        pred_bin = (p_flare >= t_used).astype(np.int64)

        cm = confusion_matrix(y_bin, pred_bin, labels=[0, 1])
        f1b = f1_score(y_bin, pred_bin, average="binary", zero_division=0)
        prec = precision_score(y_bin, pred_bin, zero_division=0)
        rec = recall_score(y_bin, pred_bin, zero_division=0)

        print(f"\n=== Stage1 {split_name} @ t={t_used:.3f} ===")
        print("F1:", f1b, "Precision:", prec, "Recall:", rec)
        print("Confusion matrix [[TN,FP],[FN,TP]]:\n", cm)
        print(classification_report(y_bin, pred_bin, target_names=["quiet", "flare"], zero_division=0))
        stage1_pass_rate_by_class(p_flare, y_true, t_used)

        return {
            "threshold": float(t_used),
            "f1": float(f1b),
            "precision": float(prec),
            "recall": float(rec),
            "confusion_matrix": cm.tolist(),
        }

    metrics = {}
    selected_t = float(t_eval)

    if eval_balanced:
        p_bal, y_bal = predict_stage1(dl_val_fixed)
        metrics["val_fixed_initial"] = eval_stage1_from_probs(p_bal, y_bal, "VAL_FIXED (balanced)", selected_t)

    if eval_natural:
        p_nat, y_nat = predict_stage1(dl_val_nat)

        thresholds = sorted(set(float(x) for x in (
            cfg.t_flare_grid if getattr(cfg, "t_flare_grid", None)
            else [0.01,0.02,0.03,0.05,0.075,0.1,0.125,0.15,0.2,0.25,0.3,0.35,0.4]
        )))

        best, rows, is_feasible, warn_msg = choose_t_for_m_goal_strict(
            p_nat, y_nat, thresholds=thresholds,
            m_goal=m_goal, x_goal=x_goal, quiet_cap=quiet_cap, min_flare_pass=min_flare_pass
        )
        if not is_feasible:
            print("[WARN]", warn_msg)
        selected_t = float(best["t"])

        print("\nChosen threshold from VAL_NATURAL:")
        print(best)

        metrics["val_natural_tuned"] = eval_stage1_from_probs(p_nat, y_nat, "VAL_NATURAL", selected_t)

        with open(os.path.join(out_dir, "stage1_threshold_sweep_val_natural.json"), "w", encoding="utf-8") as f:
            json.dump({
                "selected_threshold": selected_t,
                "goals": {
                    "m_goal": m_goal, "x_goal": x_goal, "quiet_cap": quiet_cap,
                    "min_flare_pass": min_flare_pass, "fail_fast_on_m_goal": fail_fast_on_m_goal
                },
                "best": best,
                "rows": rows,
            }, f, indent=2)

        if eval_balanced:
            metrics["val_fixed_tuned"] = eval_stage1_from_probs(p_bal, y_bal, "VAL_FIXED (balanced)", selected_t)

    if eval_test and os.path.exists(man_test):
        ds_test = NPZFileDataset(man_test, name2id, max_files=cfg.max_test_files, allow_pickle=True)
        dl_test_nat = DataLoader(
            ds_test, batch_size=cfg.batch_files, shuffle=False, num_workers=0, pin_memory=True,
            collate_fn=lambda b: collate_take_random_samples_fixedT(b, per_file=cfg.per_file, T_fixed=T_fixed, clamp_abs=cfg.clamp_abs),
        )
        p_test_nat, y_test_nat = predict_stage1(dl_test_nat)
        metrics["test_natural"] = eval_stage1_from_probs(p_test_nat, y_test_nat, "TEST_NATURAL", selected_t)

    return {"out_dir": out_dir, "selected_t_flare": selected_t, "metrics": metrics}


def _extract_model_state(ckpt: dict):
    if "model_state" in ckpt:
        return ckpt["model_state"]
    if "model" in ckpt:
        return ckpt["model"]
    if "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    raise KeyError("No model state key found in checkpoint.")

def load_and_eval_skip_stage1(
    ckpt_stage1_path: str,
    ckpt_stage2_path: str,
    use_balanced_fixed: bool = True,
    t_flare_fixed: float = None,           # set e.g. 0.24 to skip gate tuning
    use_sev_thresholds: bool = True,
    baseline_t_sev=(0.8, 0.1, 0.8, 0.1),  # fallback
    enforce_constraint_met: bool = True,
    min_prec_B: float = 0.05,
    min_rec_M: float = 0.05,
    min_rec_X: float = 0.05,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt1 = torch.load(ckpt_stage1_path, map_location=device)
    ckpt2 = torch.load(ckpt_stage2_path, map_location=device)

    cfg_dict = ckpt1.get("extra", {}).get("cfg")
    if cfg_dict is None:
        raise ValueError("Stage1 checkpoint does not contain cfg in ckpt['extra']['cfg']")
    cfg = ExperimentCfg(**cfg_dict)

    out_dir = os.path.dirname(ckpt_stage2_path)

    man_dir = os.path.join(cfg.root4, "manifests_by_group")
    man_train = os.path.join(man_dir, f"manifest_train_W{cfg.W}h_H{cfg.H}h.json")
    man_val   = os.path.join(man_dir, f"manifest_val_W{cfg.W}h_H{cfg.H}h.json")
    man_test  = os.path.join(man_dir, f"manifest_test_W{cfg.W}h_H{cfg.H}h.json")

    ds_train = NPZFileDataset(man_train, name2id, max_files=cfg.max_train_files, allow_pickle=True)
    X0, _ = ds_train[0]
    T_fixed = int(X0.shape[1])
    F = int(X0.shape[2])

    def _build_loader(split: str):
        if use_balanced_fixed:
            fixed_npz = os.path.join(out_dir, f"{split}_fixed_bal_W{cfg.W}_H{cfg.H}.npz")
            if os.path.exists(fixed_npz):
                return make_fixed_loader(fixed_npz, batch_size=cfg.fixed_batch_size, num_workers=0), fixed_npz
        man = man_val if split == "val" else man_test
        if not os.path.exists(man):
            return None, None
        ds = NPZFileDataset(
            man, name2id,
            max_files=cfg.max_val_files if split == "val" else cfg.max_test_files,
            allow_pickle=True
        )
        dl = DataLoader(
            ds,
            batch_size=cfg.batch_files,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=lambda b: collate_take_random_samples_fixedT(
                b, per_file=cfg.per_file, T_fixed=T_fixed, clamp_abs=cfg.clamp_abs
            ),
        )
        return dl, None

    dl_val, val_fixed_path = _build_loader("val")
    dl_test, test_fixed_path = _build_loader("test")

    if dl_val is None:
        raise RuntimeError("Could not build VAL loader.")

    # Stage1 model
    model_s1 = SimpleTCN(num_features=F, num_classes=2, hidden=cfg.hidden).to(device)
    model_s1.load_state_dict(_extract_model_state(ckpt1))
    model_s1.eval()

    # Stage2 model (infer classes from checkpoint fc.weight)
    s2_state = _extract_model_state(ckpt2)
    if "fc.weight" not in s2_state:
        raise KeyError("Stage2 checkpoint missing fc.weight")
    s2_num_classes = int(s2_state["fc.weight"].shape[0])

    model_s2 = SimpleTCN(num_features=F, num_classes=s2_num_classes, hidden=cfg.hidden).to(device)
    model_s2.load_state_dict(s2_state)
    model_s2.eval()

    # VAL predictions
    probs1_va, y_va = predict_proba(model_s1, dl_val, device=device)
    p_flare_va = probs1_va[:, 1]
    probs2_va, _ = predict_proba(model_s2, dl_val, device=device)

    # Gate
    if t_flare_fixed is not None:
        t_flare = float(t_flare_fixed)
        gate_info = {"method": "fixed", "t_flare": t_flare}
    else:
        grid = cfg.t_flare_grid
        if isinstance(grid, (float, int)):
            grid = (float(grid),)  # fixes your previous TypeError
        t_flare, gate_info = tune_t_flare_quiet(
            p_flare_va, probs2_va, y_va,
            grid=grid,
            objective=cfg.gate_objective,
            recall_weight=cfg.gate_recall_weight,
            recall_classes=cfg.gate_recall_classes,
        )
        gate_info["method"] = "grid_tuned"

    # Severity thresholds
    t_sev = list(baseline_t_sev)
    sev_info = {"source": "baseline", "t_sev": list(baseline_t_sev)}

    if use_sev_thresholds:
        best = tune_sev_thresholds_with_constraints(
            p_flare_va, probs2_va, y_va, t_flare=t_flare, grid=cfg.sev_t_grid,
            min_prec_B=min_prec_B, min_rec_M=min_rec_M, min_rec_X=min_rec_X
        )
        if best is not None:
            if (not enforce_constraint_met) or bool(best.get("constraint_met", False)):
                t_sev = list(best["t_sev"])
                sev_info = best
            else:
                sev_info = {
                    "source": "baseline_due_to_constraint_false",
                    "best_candidate": best,
                    "t_sev": list(baseline_t_sev),
                }

    # VAL metrics
    pred_va = two_stage_predict_with_quiet(p_flare_va, probs2_va, t_flare=t_flare, t_sev=t_sev)
    cm_va = confusion_matrix(y_va, pred_va, labels=list(range(K)))
    mf1_va = f1_score(y_va, pred_va, average="macro", labels=list(range(K)))

    print("\n=== VAL (two-stage, eval-only) ===")
    print("t_flare:", t_flare, "| t_sev:", t_sev)
    print(classification_report(y_va, pred_va, labels=list(range(K)), target_names=classes, zero_division=0))

    result = {
        "val": {
            "macro_f1_two_stage": float(mf1_va),
            "recall_per_class_two_stage": per_class_recall(cm_va),
            "confusion_matrix_two_stage": cm_va.tolist(),
            "dist": dict(Counter(y_va.tolist())),
        },
        "thresholds": {
            "two_stage": {
                "t_flare": float(t_flare),
                "t_sev": [float(x) for x in t_sev],
                "gate_info": gate_info,
                "sev_info": sev_info,
            }
        },
        "paths": {
            "stage1_ckpt": ckpt_stage1_path,
            "stage2_ckpt": ckpt_stage2_path,
            "val_fixed_path": val_fixed_path,
            "test_fixed_path": test_fixed_path,
        },
    }

    # TEST metrics
    if dl_test is not None:
        probs1_te, y_te = predict_proba(model_s1, dl_test, device=device)
        p_flare_te = probs1_te[:, 1]
        probs2_te, _ = predict_proba(model_s2, dl_test, device=device)

        pred_te = two_stage_predict_with_quiet(p_flare_te, probs2_te, t_flare=t_flare, t_sev=t_sev)
        cm_te = confusion_matrix(y_te, pred_te, labels=list(range(K)))
        mf1_te = f1_score(y_te, pred_te, average="macro", labels=list(range(K)))

        print("\n=== TEST (two-stage, eval-only) ===")
        print("t_flare:", t_flare, "| t_sev:", t_sev)
        print(classification_report(y_te, pred_te, labels=list(range(K)), target_names=classes, zero_division=0))

        result["test"] = {
            "macro_f1_two_stage": float(mf1_te),
            "recall_per_class_two_stage": per_class_recall(cm_te),
            "confusion_matrix_two_stage": cm_te.tolist(),
            "dist": dict(Counter(y_te.tolist())),
        }
    else:
        result["test"] = None

    out_json = os.path.join(out_dir, "eval_only_skip_stage1.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("\nSaved:", out_json)

    return result

# CELL 1: Helpers (resume-safe A/B framework)
import os, json, time, shutil
import numpy as np
import pandas as pd
import torch
from dataclasses import asdict
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

def clone_cfg(cfg):
    return ExperimentCfg(**asdict(cfg))

def apply_cfg_patch(cfg, patch):
    for k, v in patch.items():
        setattr(cfg, k, v)
    return cfg

def _resolve_entry_path(root_dir, rel_file):
    p = os.path.normpath(rel_file)
    if not os.path.isabs(p):
        p = os.path.normpath(os.path.join(root_dir, p))
    return p

def _save_json(fp, obj):
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def _load_json(fp, default):
    if os.path.exists(fp):
        with open(fp, "r", encoding="utf-8") as f:
            return json.load(f)
    return default

def build_quality_filtered_root(base_root, W=24, H=24, nan_drop=0.35, allnan_drop=40, suffix="_AB_quality_filtered"):
    src_man_dir = os.path.join(base_root, "manifests_by_group")
    dst_root = base_root + suffix
    dst_man_dir = os.path.join(dst_root, "manifests_by_group")
    os.makedirs(dst_man_dir, exist_ok=True)

    src_train = os.path.join(src_man_dir, f"manifest_train_W{W}h_H{H}h.json")
    src_val   = os.path.join(src_man_dir, f"manifest_val_W{W}h_H{H}h.json")
    src_test  = os.path.join(src_man_dir, f"manifest_test_W{W}h_H{H}h.json")

    dst_train = os.path.join(dst_man_dir, f"manifest_train_W{W}h_H{H}h.json")
    dst_val   = os.path.join(dst_man_dir, f"manifest_val_W{W}h_H{H}h.json")
    dst_test  = os.path.join(dst_man_dir, f"manifest_test_W{W}h_H{H}h.json")
    report_fp = os.path.join(dst_man_dir, f"quality_report_train_W{W}h_H{H}h.json")

    if all(os.path.exists(p) for p in [dst_train, dst_val, dst_test, report_fp]):
        with open(report_fp, "r", encoding="utf-8") as f:
            return dst_root, json.load(f)

    with open(src_train, "r", encoding="utf-8") as f:
        m = json.load(f)

    root_dir = m.get("root_dir", "")
    kept, dropped = [], []

    for i, ent in enumerate(m["entries"], start=1):
        fp = _resolve_entry_path(root_dir, ent["file"])
        z = np.load(fp, allow_pickle=True)
        X = z["X"].astype(np.float32, copy=False)

        nan_frac = float(np.isnan(X).mean())
        n_allnan_feat = int(np.all(np.all(np.isnan(X), axis=1), axis=0).sum())
        bad = (nan_frac > nan_drop) or (n_allnan_feat >= allnan_drop)

        if bad:
            dropped.append({
                "file": ent["file"],
                "N": int(X.shape[0]),
                "nan_frac": nan_frac,
                "all_nan_features": n_allnan_feat
            })
        else:
            kept.append(ent)

        if i % 200 == 0:
            print(f"[quality-filter] scanned {i}/{len(m['entries'])}")

    out = dict(m)
    out["entries"] = kept
    out["quality_filter"] = {"nan_drop": nan_drop, "allnan_drop": allnan_drop}

    with open(dst_train, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    shutil.copy2(src_val, dst_val)
    shutil.copy2(src_test, dst_test)

    rep = {
        "base_root": base_root,
        "filtered_root": dst_root,
        "train_total": len(m["entries"]),
        "train_kept": len(kept),
        "train_dropped": len(dropped),
        "nan_drop": nan_drop,
        "allnan_drop": allnan_drop,
        "dropped_examples_top20": sorted(dropped, key=lambda r: (r["nan_frac"], r["all_nan_features"]), reverse=True)[:20],
    }
    _save_json(report_fp, rep)
    return dst_root, rep

def _build_loader_from_cfg(cfg, split="val", use_balanced=False):
    man_dir = os.path.join(cfg.root4, "manifests_by_group")
    man_train = os.path.join(man_dir, f"manifest_train_W{cfg.W}h_H{cfg.H}h.json")
    man_val   = os.path.join(man_dir, f"manifest_val_W{cfg.W}h_H{cfg.H}h.json")
    man_test  = os.path.join(man_dir, f"manifest_test_W{cfg.W}h_H{cfg.H}h.json")

    ds_train = NPZFileDataset(man_train, name2id, max_files=cfg.max_train_files, allow_pickle=True)
    X0, _ = ds_train[0]
    T_fixed = int(X0.shape[1])

    if split == "val":
        if use_balanced:
            val_fixed = os.path.join(cfg.out_root, f"W{cfg.W}_H{cfg.H}_two_stage_bal", f"val_fixed_bal_W{cfg.W}_H{cfg.H}.npz")
            return make_fixed_loader(val_fixed, batch_size=cfg.fixed_batch_size, num_workers=0)
        ds = NPZFileDataset(man_val, name2id, max_files=cfg.max_val_files, allow_pickle=True)
    elif split == "test":
        ds = NPZFileDataset(man_test, name2id, max_files=cfg.max_test_files, allow_pickle=True)
    else:
        raise ValueError("split must be 'val' or 'test'")

    return DataLoader(
        ds,
        batch_size=cfg.batch_files,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=lambda b: collate_take_random_samples_fixedT(
            b, per_file=cfg.per_file, T_fixed=T_fixed, clamp_abs=cfg.clamp_abs
        ),
    )

@torch.no_grad()
def stage1_probs_from_ckpt(ckpt_path, split="val", use_balanced=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ExperimentCfg(**ckpt["extra"]["cfg"])
    set_seed(cfg.seed)

    loader = _build_loader_from_cfg(cfg, split=split, use_balanced=use_balanced)

    man_dir = os.path.join(cfg.root4, "manifests_by_group")
    man_train = os.path.join(man_dir, f"manifest_train_W{cfg.W}h_H{cfg.H}h.json")
    ds_train = NPZFileDataset(man_train, name2id, max_files=cfg.max_train_files, allow_pickle=True)
    X0, _ = ds_train[0]
    F = int(X0.shape[2])

    model = SimpleTCN(num_features=F, num_classes=2, hidden=cfg.hidden).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    probs, y_true = predict_proba(model, loader, device=device)
    return probs[:, 1], y_true, cfg

def metrics_at_t(p_flare, y_true, t):
    y_bin = (y_true != IDX_Q).astype(np.int64)
    pred = (p_flare >= float(t)).astype(np.int64)
    cm = confusion_matrix(y_bin, pred, labels=[0, 1])

    out = {
        "t": float(t),
        "f1_bin": float(f1_score(y_bin, pred, average="binary", zero_division=0)),
        "precision_bin": float(precision_score(y_bin, pred, zero_division=0)),
        "recall_bin": float(recall_score(y_bin, pred, zero_division=0)),
        "cm_bin": cm.tolist(),
        "pass_rate": {},
        "support": {},
    }
    for cid, cname in enumerate(classes):
        m = (y_true == cid)
        out["support"][cname] = int(m.sum())
        out["pass_rate"][cname] = float((p_flare[m] >= float(t)).mean()) if m.sum() else None
    return out

def summarize_results(results):
    rows = []
    for tag, r in results.items():
        if "val_natural" not in r:
            continue
        vn = r["val_natural"]
        rows.append({
            "tag": tag,
            "t_eval": r.get("t_eval"),
            "f1_bin": vn["f1_bin"],
            "prec_bin": vn["precision_bin"],
            "rec_bin": vn["recall_bin"],
            "quiet_pass": vn["pass_rate"]["quiet"],
            "B_pass": vn["pass_rate"]["B"],
            "C_pass": vn["pass_rate"]["C"],
            "M_pass": vn["pass_rate"]["M"],
            "X_pass": vn["pass_rate"]["X"],
            "hours": r.get("hours"),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["M_pass", "quiet_pass"], ascending=[False, True]).reset_index(drop=True)

# ============================================================
# HOW TO RUN
# ============================================================

##Run only check model of stage1
# balanced VAL
#print(load_and_eval_stage1(r"runs_pytorch_stage2_fix\W24_H24_two_stage_bal\stage1_last.pt", split="val", use_balanced=True))
# natural VAL (original distribution)
#print(load_and_eval_stage1(r"runs_pytorch_stage2_fix\W24_H24_two_stage_bal\stage1_last.pt", split="val", use_balanced=False))

#Run only check model of stage2
#res2 = load_and_eval(cfg1)

#Changed loss_s2 = nn.CrossEntropyLoss(weight=stage2_w)


cfg1 = ExperimentCfg(
    root4=r"section3_windows_forecast_24_fixed_section4_fe_ext_compact_TRUE",
    W=24, H=24,
    out_root=r"runs_pytorch_stage2_fix",

    max_train_files=2010,
    max_val_files=121,
    max_test_files=874,

    batch_files=8,
    per_file=256,
    hidden=64,
    lr=1e-3,

    # fixed set
    fixed_seed=123,
    fixed_batch_size=512,

    # Stage1: נשאיר כמו עכשיו (לא מבזבזים ריצה על זה)
    s1_epochs=8,#12
    s1_pos_weight=20.0,
    s1_use_focal=False,

    # Stage2: FIX
    s2_epochs=8,#12
    s2_use_focal=True,
    s2_alpha=(6.0, 10.0, 14.0, 30.0),   #(0.5, 1.0, 10.0, 30.0),
    s2_focal_gamma=2.0,

    # Stage2 sampler אגרסיבי
    #s2_wX=300.0,
    #s2_wM=120.0,
    #s2_wC=80,
    #s2_wB=10,
    #s2_wQ=0.5,
    s2_wB = 20.0,
    s2_wC = 40.0,
    s2_wM = 60.0,
    s2_wX = 300.0,
    s2_wQ = 0.1,

    # t_flare grid לא נמוך מדי עדיין
    t_flare_grid=(0.05,0.075,0.1,0.125,0.15,0.2,0.25,0.3),
    t_flare_lam=3.0,

    seed=123,
)

cfg3 = ExperimentCfg(
    root4=r"section3_windows_forecast_24_fixed_section4_fe_ext_compact_TRUE",
    W=24, H=24,
    out_root=r"runs_pytorch_stage2_fix",

    max_train_files=2010,
    max_val_files=121,
    max_test_files=874,

    batch_files=8,
    per_file=256,
    hidden=64,
    lr=1e-3,

    # fixed set
    fixed_seed=123,
    fixed_batch_size=512,

    # Stage1: נשאיר כמו עכשיו (לא מבזבזים ריצה על זה)
    s1_epochs=10,#12
    s1_pos_weight=3.0,
    s1_use_focal=True,

    # Stage2: FIX
    s2_epochs=15,#12
    s2_use_focal=True,
    s2_alpha=(4.0, 12.0, 20.0, 40.0),   #(0.5, 1.0, 10.0, 30.0),
    s2_focal_gamma=2.0,
    s2_flare_frac = 0.90,
    s2_flare_quota = (0.15, 0.25, 0.35, 0.25),
    s2_alpha_mode = "manual",
    # Stage2 sampler אגרסיבי
    #s2_wX=300.0,
    #s2_wM=120.0,
    #s2_wC=80,
    #s2_wB=10,
    #s2_wQ=0.5,
    s2_wB = 2.0,
    s2_wC = 2.0,
    s2_wM = 3.0,
    s2_wX = 5.0,
    s2_wQ = 10,

    # t_flare grid לא נמוך מדי עדיין
    t_flare_grid = (0.01,0.02,0.03,0.05,0.075,0.1,0.125,0.15,0.2),
    t_flare_lam=3.0,
    
    seed=123,

    gate_mode = "grid_macro_f1",
    gate_objective = "macro_f1_plus_recall",
    gate_recall_weight = 5.0,
    sev_t_grid = (0.01,0.02,0.03,0.05,0.075,0.1,0.125)

)

# Stage-1 focused cfg (more realistic gate behavior)
cfg_stage1_gate = ExperimentCfg(
    root4=r"section3_windows_forecast_24_fixed_section4_fe_ext_compact_TRUE",
    W=24, H=24,
    out_root=r"runs_pytorch_stage2_fix",

    max_train_files=2010,
    max_val_files=121,
    max_test_files=874,

    batch_files=8,
    per_file=256,
    hidden=64,
    lr=1e-3,
    seed=123,

    fixed_seed=123,
    fixed_batch_size=512,

    # Stage1: reduce positive pressure vs your previous setting (20)
    s1_epochs=8,
    s1_use_focal=False,
    s1_pos_weight=6.0,

    # Stage1 file sampler (less extreme than 100/20/5/3)
    wX=20.0,
    wM=10.0,
    wC=4.0,
    wB=2.0,
    wQ=1.0,

    # gate search grid used by run_stage1_only
    t_flare_grid=(
        0.01,0.02,0.03,0.05,0.075,0.1,0.125,0.15,0.2,0.25,0.3,0.35,0.4,0.5,0.6,0.7,0.8,0.9
    ),

)

cfg_stage1_m90 = ExperimentCfg(
    root4=r"section3_windows_forecast_24_fixed_section4_fe_ext_compact_TRUE",
    W=24, H=24,
    out_root=r"runs_pytorch_stage2_fix",

    max_train_files=2010,
    max_val_files=121,
    max_test_files=874,

    batch_files=8,
    per_file=256,
    hidden=64,
    lr=1e-3,
    seed=123,

    fixed_seed=123,
    fixed_batch_size=512,

    s1_epochs=8,
    s1_use_focal=True,
    s1_focal_gamma=2.0,
    s1_pos_weight=12.0,

    wQ=1.0,
    wB=4.0,
    wC=10.0,
    wM=120.0,
    wX=300.0,

    #t_flare_grid=(0.01,0.02,0.03,0.05,0.075,0.1,0.125,0.15,0.2,0.25,0.3,0.35,0.4),
    t_flare_grid=(0.01,0.02,0.03,0.05,0.075,0.1,0.125,0.15,0.2,0.25,0.3,0.35,0.4,0.5,0.6,0.7,0.8,0.9)

)

# extra params used by collate_stage1_quota_fixedT (set after creation)
#cfg_stage1_m90.s1_quiet_frac = 0.40
#cfg_stage1_m90.s1_flare_quota = (0.10, 0.20, 0.55, 0.15)  # B,C,M,X
#cfg_stage1_m90.s1_m_min_per_file = 2
#cfg_stage1_m90.s1_x_min_per_file = 0
#cfg_stage1_m90.s1_x_max_per_file = 3
## run:
#res = new_run_stage1_only(
#    cfg_stage1_m90,
#    eval_balanced=True,
#    eval_natural=True,
#    eval_test=True,
#    t_eval=0.15,
#    m_goal=0.90,
#    x_goal=0.98,
#    quiet_cap=0.30,
#    min_flare_pass=0.0,
#    fail_fast_on_m_goal=True,
#)
#print(res)


#==============================================================================
# Run A/B (one change at a time) + threshold-only sweep
#==============================================================================
'''
AB_ROOT = r"runs_pytorch_stage2_fix_ab"
os.makedirs(AB_ROOT, exist_ok=True)
RESULTS_FP = os.path.join(AB_ROOT, "ab_results.json")
results = _load_json(RESULTS_FP, {})

base_cfg = clone_cfg(cfg3)  # your best baseline config

experiments = [
    {"tag": "A0_baseline",            "t_eval": 0.10, "cfg_patch": {},                                               "quality_filter": False},
    {"tag": "B1_quality_filter_only", "t_eval": 0.10, "cfg_patch": {},                                               "quality_filter": True},
    {"tag": "B3_pos_weight_only",     "t_eval": 0.10, "cfg_patch": {"s1_pos_weight": 2.0},                          "quality_filter": False},
    {"tag": "B4_loss_only_no_focal",  "t_eval": 0.10, "cfg_patch": {"s1_use_focal": False, "s1_pos_weight": 3.0},   "quality_filter": False},
    {"tag": "B5_sampler_only_milder", "t_eval": 0.10, "cfg_patch": {"wX": 60.0, "wM": 12.0, "wC": 4.0, "wB": 2.0, "wQ": 1.0}, "quality_filter": False},
]

for exp in experiments:
    tag = exp["tag"]
    if tag in results:
        print(f"[SKIP] {tag} already done.")
        continue

    cfg_run = clone_cfg(base_cfg)
    apply_cfg_patch(cfg_run, exp["cfg_patch"])
    cfg_run.out_root = os.path.join(AB_ROOT, tag)

    quality_info = None
    if exp["quality_filter"]:
        filtered_root, quality_info = build_quality_filtered_root(
            base_root=base_cfg.root4,
            W=base_cfg.W,
            H=base_cfg.H,
            nan_drop=0.35,
            allnan_drop=40,
            suffix="_AB_quality_filtered",
        )
        cfg_run.root4 = filtered_root

    t_eval = exp["t_eval"]
    t0 = time.time()
    print(f"\n=== RUN {tag} | t_eval={t_eval} ===")

    run_stage1_only(
        cfg_run,
        eval_balanced=True,
        eval_natural=True,
        eval_test=False,
        t_eval=t_eval,
    )

    ckpt_path = os.path.join(cfg_run.out_root, f"W{cfg_run.W}_H{cfg_run.H}_two_stage_bal", "stage1_last.pt")

    p_val_nat, y_val_nat, _ = stage1_probs_from_ckpt(ckpt_path, split="val", use_balanced=False)
    p_val_bal, y_val_bal, _ = stage1_probs_from_ckpt(ckpt_path, split="val", use_balanced=True)

    rec = {
        "tag": tag,
        "t_eval": t_eval,
        "cfg_patch": exp["cfg_patch"],
        "quality_filter": exp["quality_filter"],
        "quality_info": quality_info,
        "ckpt_path": ckpt_path,
        "hours": (time.time() - t0) / 3600.0,
        "val_natural": metrics_at_t(p_val_nat, y_val_nat, t_eval),
        "val_balanced": metrics_at_t(p_val_bal, y_val_bal, t_eval),
    }

    results[tag] = rec
    _save_json(RESULTS_FP, results)

    df = summarize_results(results)
    if not df.empty:
        print(df.to_string(index=False))
        df.to_csv(os.path.join(AB_ROOT, "ab_summary.csv"), index=False)

# B2: threshold-only (no retrain) on baseline checkpoint
if "A0_baseline" in results:
    base_ckpt = results["A0_baseline"]["ckpt_path"]
    p_nat, y_nat, _ = stage1_probs_from_ckpt(base_ckpt, split="val", use_balanced=False)
    p_bal, y_bal, _ = stage1_probs_from_ckpt(base_ckpt, split="val", use_balanced=True)

    thr_grid = [0.07, 0.10, 0.125, 0.15]
    b2_rows = []
    for t in thr_grid:
        rn = metrics_at_t(p_nat, y_nat, t)
        rb = metrics_at_t(p_bal, y_bal, t)
        b2_rows.append({
            "t": t,
            "val_nat_f1": rn["f1_bin"],
            "val_nat_quiet_pass": rn["pass_rate"]["quiet"],
            "val_nat_B_pass": rn["pass_rate"]["B"],
            "val_nat_C_pass": rn["pass_rate"]["C"],
            "val_nat_M_pass": rn["pass_rate"]["M"],
            "val_nat_X_pass": rn["pass_rate"]["X"],
            "val_bal_f1": rb["f1_bin"],
        })

    results["B2_threshold_only"] = {
        "tag": "B2_threshold_only",
        "ckpt_path": base_ckpt,
        "rows": b2_rows
    }
    _save_json(RESULTS_FP, results)

    b2_df = pd.DataFrame(b2_rows).sort_values(["val_nat_M_pass", "val_nat_quiet_pass"], ascending=[False, True])
    print("\n=== B2 threshold-only (baseline ckpt) ===")
    print(b2_df.to_string(index=False))
    b2_df.to_csv(os.path.join(AB_ROOT, "b2_threshold_only.csv"), index=False)

print("\nSaved:", RESULTS_FP)
print("Summary:", os.path.join(AB_ROOT, "ab_summary.csv"))'''
#==============================================================================
#==============================================================================

#print(run_stage1_only(cfg3, eval_balanced=True, eval_natural=True, eval_test=False, t_eval=0.1))

#=========================================================================================================
#Sweep thresholds on the saved checkpoint

# 1) Train ONCE (or skip if already trained and ckpt exists)
#print(run_stage1_only(cfg3, eval_balanced=True, eval_natural=True, eval_test=False, t_eval=0.1))


# 2) Sweep thresholds on the saved checkpoint
#import os
#import numpy as np
#import pandas as pd
'''
ckpt_path = os.path.join(
    cfg3.out_root,
    f"W{cfg3.W}_H{cfg3.H}_two_stage_bal/",
    "stage1_last.pt"
)

p_val_nat, y_val_nat, _ = stage1_probs_from_ckpt(ckpt_path, split="val", use_balanced=False)
p_val_bal, y_val_bal, _ = stage1_probs_from_ckpt(ckpt_path, split="val", use_balanced=True)

thresholds = np.round(np.arange(0.05, 0.4, 0.01), 3)

rows = []
for t in thresholds:
   m_nat = metrics_at_t(p_val_nat, y_val_nat, t)
   m_bal = metrics_at_t(p_val_bal, y_val_bal, t)
   rows.append({
       "t": t,
       "f1_nat": m_nat["f1_bin"],
       "prec_nat": m_nat["precision_bin"],
       "rec_nat": m_nat["recall_bin"],
       "quiet_pass_nat": m_nat["pass_rate"]["quiet"],
       "B_pass_nat": m_nat["pass_rate"]["B"],
       "C_pass_nat": m_nat["pass_rate"]["C"],
       "M_pass_nat": m_nat["pass_rate"]["M"],
       "X_pass_nat": m_nat["pass_rate"]["X"],
       "f1_bal": m_bal["f1_bin"],
   })

df = pd.DataFrame(rows)
print(df.to_string(index=False))
df.to_csv("runs_pytorch_stage2_fix_ab/t_eval_0.2_0.40_sweep.csv", index=False)

p_val_nat, y_val_nat, _ = stage1_probs_from_ckpt(ckpt_path, split="test", use_balanced=False)
p_val_bal, y_val_bal, _ = stage1_probs_from_ckpt(ckpt_path, split="test", use_balanced=True)

thresholds = np.round(np.arange(0.05, 0.401, 0.02), 3)

rows = []
for t in thresholds:
   m_nat = metrics_at_t(p_val_nat, y_val_nat, t)
   m_bal = metrics_at_t(p_val_bal, y_val_bal, t)
   rows.append({
       "t": t,
       "f1_nat": m_nat["f1_bin"],
       "prec_nat": m_nat["precision_bin"],
       "rec_nat": m_nat["recall_bin"],
       "quiet_pass_nat": m_nat["pass_rate"]["quiet"],
       "B_pass_nat": m_nat["pass_rate"]["B"],
       "C_pass_nat": m_nat["pass_rate"]["C"],
       "M_pass_nat": m_nat["pass_rate"]["M"],
       "X_pass_nat": m_nat["pass_rate"]["X"],
       "f1_bal": m_bal["f1_bin"],
   })

df = pd.DataFrame(rows)
print(df.to_string(index=False))
df.to_csv("runs_pytorch_stage2_fix_ab/t_test_0.2_0.40_sweep.csv", index=False)
'''
#=========================================================================================================

 
#stage1_pass_rate_from_ckpt(
#    r"runs_pytorch_stage2_fix\W24_H24_two_stage_bal\stage1_last.pt",
#    split="val",
#    use_balanced=False,
#    t_flare=0.24
#)
#stage1_pass_rate_from_ckpt(
#    r"runs_pytorch_stage2_fix\W24_H24_two_stage_bal\stage1_last.pt",
#    split="test",
#    use_balanced=False,
#    t_flare=0.24
#)

#stage1_pass_rate_from_ckpt(
#    r"runs_pytorch_stage2_fix\W24_H24_two_stage_bal\before_change\stage1_last.pt",
#    split="val",
#    use_balanced=False,
#    t_flare=0.24
#)
#stage1_pass_rate_from_ckpt(
#    r"runs_pytorch_stage2_fix\W24_H24_two_stage_bal\before_change\stage1_last.pt",
#    split="test",
#    use_balanced=False,
#    t_flare=0.24
#)



#res1 = run_one_experiment2(cfg1)
#print("RUN#1 DONE:", res1)

#The first best runner for skip stage1
cfg2 = ExperimentCfgSkipStage1(
    root4=r"section3_windows_forecast_24_fixed_section4_fe_ext_compact_TRUE",
    W=24, H=24,
    out_root=r"runs_pytorch_stage2_fix",
    out_dir_changes = False,
    s1_resume_path=r"runs_pytorch_stage2_fix\W24_H24_two_stage_bal\stage1_last.pt",
    skip_stage1_train=True,
    s1_epochs=0,

    max_train_files=2010,
    max_val_files=121,
    max_test_files=874,

    batch_files=8,
    per_file=256,
    hidden=64,
    lr=1e-3,

    fixed_seed=123,
    fixed_batch_size=512,

    # Stage2 settings (use your tuned ones)
    s2_epochs=8,
    s2_use_focal=True,
    s2_alpha=(4.0,12.0,20.0,40.0),
    s2_alpha_mode="manual",
    s2_focal_gamma=2.0,
    s2_flare_frac=0.90,
    s2_flare_quota=(0.15,0.25,0.35,0.25),

    s2_wB=8.0,
    s2_wC=20.0,
    s2_wM=40.0,
    s2_wX=200.0,
    s2_wQ=0.2,

    # Gate: force t_flare = 0.15
    gate_mode="grid_macro_f1",
    t_flare_grid=(0.25,),
    #sev_t_grid = (0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3),

    seed=123,
    quiet_w = 0.2
)
#res_eval = load_and_eval_skip_stage1(
#        ckpt_stage1_path=r"runs_pytorch_stage2_fix\W24_H24_two_stage_bal\stage1_last.pt",
#        ckpt_stage2_path=r"runs_pytorch_stage2_fix\W24_H24_two_stage_bal_skip_stage1\stage2_last.pt",
#        t_flare_fixed=0.15,   # optional; if None it will tune
#    )

cfg4 = ExperimentCfgSkipStage1(
    root4=r"section3_windows_forecast_24_fixed_section4_fe_ext_compact_TRUE",
    W=24, H=24,
    out_root=r"runs_pytorch_stage2_fix",
    out_dir_changes=False,
    s1_resume_path=r"runs_pytorch_stage2_fix\W24_H24_two_stage_bal\stage1_last.pt",
    skip_stage1_train=True,
    s1_epochs=0,

    max_train_files=2010,
    max_val_files=121,
    max_test_files=874,

    batch_files=8,
    per_file=256,
    hidden=64,
    lr=1e-3,

    fixed_seed=123,
    fixed_batch_size=512,

    # Stage2
    s2_epochs=12,                 # was 8
    s2_use_focal=True,
    s2_alpha=(4.0, 12.0, 20.0, 40.0),
    s2_alpha_mode="manual",
    s2_focal_gamma=2.0,

    s2_flare_frac=0.85,           # was 0.90 -> more quiet/hard-quiet
    s2_flare_quota=(0.12, 0.23, 0.45, 0.20),  # B,C,M,X (more M)

    s2_wB=6.0,
    s2_wC=14.0,
    s2_wM=60.0,
    s2_wX=180.0,
    s2_wQ=1.0,                    # was 0.2

    gate_mode="grid_macro_f1",
    t_flare_grid=(0.36, 0.38, 0.40),  # from your eval sweep
    sev_t_grid=(0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6),

    seed=123,
    quiet_w = 0.6,
    manual_t_sev = True
)

#print(run_one_experiment_skip_stage1_fixed(cfg4))

#cands = []
#for tf in [0.24, 0.26, 0.28, 0.30, 0.32, 0.34]:
#    for tM in [0.10, 0.15, 0.20, 0.25, 0.30]:
#        res = load_and_eval_skip_stage1(
#            ckpt_stage1_path=r"runs_pytorch_stage2_fix\W24_H24_two_stage_bal\stage1_last.pt",
#            ckpt_stage2_path=r"runs_pytorch_stage2_fix\W24_H24_two_stage_bal_skip_stage1\stage2_last.pt",
#            use_balanced_fixed=False,          # natural VAL/TEST
#            t_flare_fixed=tf,
#            use_sev_thresholds=False,
#            baseline_t_sev=(0.8, 0.1, tM, 0.1),
#        )
#        vr = res["val"]["recall_per_class_two_stage"]
#        mf1 = res["val"]["macro_f1_two_stage"]
#
#        ok = (
#            vr["quiet"] >= 0.85 and
#            vr["C"] >= 0.10 and
#            vr["M"] >= 0.10 and
#            vr["X"] >= 0.20
#        )
#        cands.append((ok, mf1, tf, tM, vr))
#
## best feasible first
#cands_sorted = sorted(cands, key=lambda x: (x[0], x[1]), reverse=True)
#print("Top 10:")
#for row in cands_sorted[:10]:
#    print(row)
#best = cands_sorted[0]
#print("BEST:", best)



#res1 = run_one_experiment_skip_stage1_all_print(cfg2)
#print("RUN#1 DONE:", res1)
#res1 = run_one_experiment_skip_stage1_all_print(cfg2) #No Focal for second run
#print("RUN#2 DONE:", res1)

# Compact threshold sweep: natural VAL selection -> natural TEST check -> save CSV

