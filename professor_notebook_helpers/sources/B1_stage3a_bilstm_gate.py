# ==============================================================================
# Generic Stage3A reject-only runner
# Source: Handling Class Imbalance.ipynb cell 17
# ==============================================================================
# ===== Stage3A reject head integrated with Stage3BCMX backbone (train reject only for now) =====
from dataclasses import dataclass, asdict
from typing import Optional, Tuple
import os, json, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Assumes these already exist:
# IDX_Q, IDX_B, IDX_C, IDX_M, IDX_X, name2id
# SimpleTCN, NPZFileDataset, make_fixed_loader, collate_take_random_samples_fixedT
# set_seed, summarize_scan

def _extract_state_any(ckpt):
    # common patterns
    if isinstance(ckpt, dict):
        for k in ("model_state", "model_state_dict", "state_dict", "model"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    # already a state_dict
    return ckpt


class Stage3ABCMXTCN(nn.Module):
    """
    Shared backbone with:
      - head_reject: binary quiet(0)/flare(1) for post-gate cleanup
      - head_bcmx:   4-class BCMX (not trained in this run)
    """
    def __init__(self, num_features: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(num_features, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head_reject = nn.Linear(hidden, 2)  # quiet vs flare
        self.head_bcmx = nn.Linear(hidden, 4)    # B,C,M,X (later)

    def forward(self, x):
        x = x.transpose(1, 2)
        h = self.net(x).squeeze(-1)
        return self.head_reject(h), self.head_bcmx(h)

def y5_to_stage3a_label(y5: torch.Tensor) -> torch.Tensor:
    return (y5 != IDX_Q).long()  # 0 quiet, 1 flare

def build_stage3a_batch_indices(
    y5: torch.Tensor,
    p_flare: torch.Tensor,
    t_flare_train: float,
    k_pos: int = 256,
    neg_ratio: float = 1.0,
    pos_require_s1_pass: bool = False,
):
    device = y5.device
    if pos_require_s1_pass:
        pos_idx = torch.where((y5 != IDX_Q) & (p_flare >= float(t_flare_train)))[0]
    else:
        pos_idx = torch.where(y5 != IDX_Q)[0]

    neg_idx = torch.where((y5 == IDX_Q) & (p_flare >= float(t_flare_train)))[0]

    if pos_idx.numel() == 0 or neg_idx.numel() == 0:
        return None

    k_pos = min(int(k_pos), int(pos_idx.numel()))
    k_neg = min(max(1, int(round(k_pos * float(neg_ratio)))), int(neg_idx.numel()))

    def sample_idx(pool, k):
        if pool.numel() >= k:
            return pool[torch.randperm(pool.numel(), device=device)[:k]]
        ridx = torch.randint(0, pool.numel(), (k,), device=device)
        return pool[ridx]

    pos_take = sample_idx(pos_idx, k_pos)
    neg_take = sample_idx(neg_idx, k_neg)

    idx = torch.cat([pos_take, neg_take], dim=0)
    idx = idx[torch.randperm(idx.numel(), device=device)]
    return idx

@torch.no_grad()
def predict_stage1_stage3a_loader(
    model_s1: nn.Module,
    model_s3: nn.Module,
    loader: DataLoader,
    device: str,
):
    model_s1.eval()
    model_s3.eval()

    p1_all, p2_all, y5_all = [], [], []

    for X, y5 in loader:
        X = X.to(device, non_blocking=True)
        y5_np = y5.numpy().astype(np.int64)

        p1 = torch.softmax(model_s1(X), dim=1)[:, 1].detach().cpu().numpy()  # Stage1 p(flare)
        logits_rej, _ = model_s3(X)
        p2 = torch.softmax(logits_rej, dim=1)[:, 1].detach().cpu().numpy()   # Stage3A p(flare)

        p1_all.append(p1)
        p2_all.append(p2)
        y5_all.append(y5_np)

    return np.concatenate(p1_all), np.concatenate(p2_all), np.concatenate(y5_all)

def eval_stage1_stage3a(y5_true, p1, p2, t_flare, t_reject):
    flare_true = (y5_true != IDX_Q)
    quiet_true = (y5_true == IDX_Q)

    gate = (p1 >= float(t_flare))
    pred_flare = gate & (p2 >= float(t_reject))

    # end-to-end
    flare_recall = float(np.sum(flare_true & pred_flare)) / max(float(np.sum(flare_true)), 1.0)
    quiet_to_flare_fpr = float(np.sum(quiet_true & pred_flare)) / max(float(np.sum(quiet_true)), 1.0)

    # post-gate conditional
    flare_gate = flare_true & gate
    quiet_gate = quiet_true & gate

    flare_recall_given_gate = float(np.sum(flare_gate & (p2 >= float(t_reject)))) / max(float(np.sum(flare_gate)), 1.0)
    quiet_fpr_given_gate = float(np.sum(quiet_gate & (p2 >= float(t_reject)))) / max(float(np.sum(quiet_gate)), 1.0)

    pred_quiet_rate = float(np.mean(~pred_flare))

    return {
        "flare_recall": flare_recall,
        "quiet_to_flare_fpr": quiet_to_flare_fpr,
        "pred_quiet_rate": pred_quiet_rate,
        "flare_recall_given_gate": flare_recall_given_gate,
        "quiet_fpr_given_gate": quiet_fpr_given_gate,
        "gate_rate": float(np.mean(gate)),
    }

def sweep_treject(y5, p1, p2, t_flare, t_reject_grid):
    rows = []
    for tr in t_reject_grid:
        m = eval_stage1_stage3a(y5, p1, p2, t_flare=t_flare, t_reject=float(tr))
        rows.append({"t_flare": float(t_flare), "t_reject": float(tr), **m})
    rows.sort(key=lambda r: (-r["flare_recall"], r["quiet_to_flare_fpr"]))
    return rows

@dataclass
class Stage3ARejectCfg:
    root4: str
    W: int
    H: int
    out_root: str
    s1_ckpt_path: str

    max_train_files: Optional[int] = None
    max_val_files: Optional[int] = None
    max_test_files: Optional[int] = None

    batch_files: int = 8
    per_file: int = 256
    fixed_batch_size: int = 512
    hidden: int = 64
    lr: float = 1e-3
    seed: int = 123
    clamp_abs: float = 1e3

    s3a_epochs: int = 8
    t_flare_train: float = 0.04
    s3a_k_pos: int = 256
    s3a_neg_ratio: float = 1.0
    s3a_pos_require_s1_pass: bool = False

    t_flare_eval: float = 0.04
    t_reject_grid: Tuple[float, ...] = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)

    use_fixed_eval: bool = False
    min_pred_quiet_rate: float = 0.15
    max_quiet_to_flare_fpr: float = 0.30

def run_stage3a_reject_only(cfg: Stage3ARejectCfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(cfg.seed)
    print("device:", device)

    man_dir = os.path.join(cfg.root4, "manifests_by_group")
    man_train = os.path.join(man_dir, f"manifest_train_W{cfg.W}h_H{cfg.H}h.json")
    man_val = os.path.join(man_dir, f"manifest_val_W{cfg.W}h_H{cfg.H}h.json")
    man_test = os.path.join(man_dir, f"manifest_test_W{cfg.W}h_H{cfg.H}h.json")

    if cfg.s3a_pos_require_s1_pass==False:
        out_dir = os.path.join(cfg.out_root, f"W{cfg.W}_H{cfg.H}_stage3a_reject_only")
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = os.path.join(cfg.out_root, f"W{cfg.W}_H{cfg.H}_stage3a_reject_only_s1_pass_True")
        os.makedirs(out_dir, exist_ok=True)

    ds_train = NPZFileDataset(man_train, name2id, max_files=cfg.max_train_files, allow_pickle=True)
    ds_val = NPZFileDataset(man_val, name2id, max_files=cfg.max_val_files, allow_pickle=True)
    print("train files:", len(ds_train), "| val files:", len(ds_val))
    _ = summarize_scan(ds_train, "scan-train")
    _ = summarize_scan(ds_val, "scan-val")

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
        raise RuntimeError("Validation loader is missing.")

    # Stage1 frozen
    ckpt1 = torch.load(cfg.s1_ckpt_path, map_location=device)
    model_s1 = SimpleTCN(num_features=F, num_classes=2, hidden=cfg.hidden).to(device)
    model_s1.load_state_dict(_extract_state_any(ckpt1))
    model_s1.eval()
    print("Loaded Stage1:", cfg.s1_ckpt_path)

    # Stage3A model (reject head trained now; BCMX head frozen)
    model_s3 = Stage3ABCMXTCN(num_features=F, hidden=cfg.hidden).to(device)

    # freeze BCMX head while training reject-only
    for p in model_s3.head_bcmx.parameters():
        p.requires_grad = False

    loss_stage3a = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0], dtype=torch.float32, device=device))
    opt = torch.optim.Adam(
        [p for p in model_s3.parameters() if p.requires_grad],
        lr=cfg.lr
    )

    for ep in range(1, cfg.s3a_epochs + 1):
        model_s3.train()
        model_s1.eval()

        total, n = 0.0, 0
        b_seen = c_seen = m_seen = x_seen = 0
        pos_seen = neg_seen = 0

        # NEW diagnostics
        skipped_no_neg = 0
        skipped_no_pos = 0
        steps = 0
        quiet_total = 0
        quiet_hard_total = 0

        for X, y5 in dl_train:
            X = X.to(device, non_blocking=True)
            y5 = y5.to(device, non_blocking=True)

            with torch.no_grad():
                p_flare = torch.softmax(model_s1(X), dim=1)[:, 1]

            # hard-quiet diagnostics at current t_flare_train
            quiet_mask = (y5 == IDX_Q)
            hard_quiet_mask = quiet_mask & (p_flare >= float(cfg.t_flare_train))
            quiet_total += int(quiet_mask.sum().item())
            quiet_hard_total += int(hard_quiet_mask.sum().item())

            pos_cnt = int((y5 != IDX_Q).sum().item())
            neg_cnt = int(hard_quiet_mask.sum().item())

            idx = build_stage3a_batch_indices(
                y5=y5,
                p_flare=p_flare,
                t_flare_train=cfg.t_flare_train,
                k_pos=cfg.s3a_k_pos,
                neg_ratio=cfg.s3a_neg_ratio,
                pos_require_s1_pass=cfg.s3a_pos_require_s1_pass,
            )
            if idx is None:
                if pos_cnt == 0:
                    skipped_no_pos += 1
                if neg_cnt == 0:
                    skipped_no_neg += 1
                continue

            steps += 1

            Xa = X[idx]
            y5a = y5[idx]
            ya = y5_to_stage3a_label(y5a)

            logits_a, _ = model_s3(Xa)
            loss = loss_stage3a(logits_a, ya)
            if not torch.isfinite(loss):
                continue

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_s3.parameters(), 1.0)
            opt.step()

            total += float(loss.item()) * Xa.size(0)
            n += Xa.size(0)

            pos_mask = (y5a != IDX_Q)
            neg_mask = ~pos_mask
            pos_seen += int(pos_mask.sum().item())
            neg_seen += int(neg_mask.sum().item())

            b_seen += int((y5a == IDX_B).sum().item())
            c_seen += int((y5a == IDX_C).sum().item())
            m_seen += int((y5a == IDX_M).sum().item())
            x_seen += int((y5a == IDX_X).sum().item())

        hard_quiet_frac = quiet_hard_total / max(quiet_total, 1)
        print(
            f"[S3A Ep {ep}] loss={total/max(n,1):.4f} | "
            f"steps={steps} | skip_no_neg={skipped_no_neg} | skip_no_pos={skipped_no_pos} | "
            f"hard_quiet_frac@t={cfg.t_flare_train:.3f}={hard_quiet_frac:.4f} | "
            f"pos/neg={pos_seen}/{neg_seen} | B/C/M/X={b_seen}/{c_seen}/{m_seen}/{x_seen}"
        )

    # save
    torch.save(
        {"model_state": model_s3.state_dict(), "extra": {"cfg": asdict(cfg)}},
        os.path.join(out_dir, "stage3a_reject_last.pt"),
    )

    # VAL eval
    p1_va, p2_va, y5_va = predict_stage1_stage3a_loader(model_s1, model_s3, dl_val, device=device)
    np.savez(os.path.join(out_dir, "val_arrays_stage3a.npz"), p_flare=p1_va, p_reject=p2_va, y5=y5_va)

    rows_val = sweep_treject(y5_va, p1_va, p2_va, t_flare=cfg.t_flare_eval, t_reject_grid=cfg.t_reject_grid)
    df_val = pd.DataFrame(rows_val)
    df_val.to_csv(os.path.join(out_dir, "stage3a_reject_sweep_val.csv"), index=False)

    feasible = df_val[
        (df_val["quiet_to_flare_fpr"] <= cfg.max_quiet_to_flare_fpr) &
        (df_val["pred_quiet_rate"] >= cfg.min_pred_quiet_rate)
    ]
    best_df = feasible if len(feasible) else df_val
    best = best_df.sort_values(["flare_recall", "quiet_to_flare_fpr"], ascending=[False, True]).iloc[0].to_dict()
    print("Best VAL:", best)

    # TEST eval
    test_rows = None
    if dl_test is not None:
        p1_te, p2_te, y5_te = predict_stage1_stage3a_loader(model_s1, model_s3, dl_test, device=device)
        np.savez(os.path.join(out_dir, "test_arrays_stage3a.npz"), p_flare=p1_te, p_reject=p2_te, y5=y5_te)

        rows_te = sweep_treject(y5_te, p1_te, p2_te, t_flare=cfg.t_flare_eval, t_reject_grid=cfg.t_reject_grid)
        test_rows = rows_te
        pd.DataFrame(rows_te).to_csv(os.path.join(out_dir, "stage3a_reject_sweep_test.csv"), index=False)

    out = {
        "out_dir": out_dir,
        "best_val": best,
        "val_sweep_csv": "stage3a_reject_sweep_val.csv",
        "test_sweep_csv": "stage3a_reject_sweep_test.csv" if test_rows is not None else None,
    }
    with open(os.path.join(out_dir, "stage3a_reject_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    return out


# s3a_pos_require_s1_pass=False - W24_H24_stage3a_reject_only
cfg_reject = Stage3ARejectCfg(
    root4=r"section3_windows_forecast_24_fixed_section4_fe_ext_compact_TRUE",
    W=24, H=24,
    out_root=r"runs_pytorch_stage2_fix",
    s1_ckpt_path=r"runs_pytorch_stage2_fix\W24_H24_two_stage_bal_other_stage1\stage1_last.pt",
    s3a_epochs=8,
    t_flare_train=0.04,
    s3a_k_pos=256,
    s3a_neg_ratio=1.0,
    s3a_pos_require_s1_pass=False,
    t_flare_eval=0.04,
    t_reject_grid=(0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50),
    use_fixed_eval=False,
)
#res_reject = run_stage3a_reject_only(cfg_reject)
#print(res_reject)

# s3a_pos_require_s1_pass=True - W24_H24_stage3a_reject_only_s1_pass_True
cfg_reject = Stage3ARejectCfg(
    root4=r"section3_windows_forecast_24_fixed_section4_fe_ext_compact_TRUE",
    W=24, H=24,
    out_root=r"runs_pytorch_stage2_fix",
    s1_ckpt_path=r"runs_pytorch_stage2_fix\W24_H24_two_stage_bal_other_stage1\stage1_last.pt",
    s3a_epochs=8,
    t_flare_train=0.04,
    s3a_k_pos=256,
    s3a_neg_ratio=1.5,
    s3a_pos_require_s1_pass=True,
    t_flare_eval=0.04,
    t_reject_grid=(0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50),
    use_fixed_eval=False,
)
#res_reject = run_stage3a_reject_only(cfg_reject)
#print(res_reject)


# ==============================================================================
# BiLSTM Stage3A override + run config
# Source: Handling Class Imbalance.ipynb cell 19 section
# ==============================================================================
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
