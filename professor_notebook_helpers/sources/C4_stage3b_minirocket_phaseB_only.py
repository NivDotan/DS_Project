# ==============================================================================
# MiniRocket backbone + compatibility hooks
# Source: Handling_Class_Imbalance_MiniRocket_Stage3.ipynb cell 5
# ==============================================================================
import os
import numpy as np
import torch
import torch.nn as nn
from joblib import dump, load
from sktime.transformations.panel.rocket import MiniRocketMultivariate

class _MiniRocketBackbone:
    def __init__(self, num_kernels=2000, random_state=42):
        self.rocket = MiniRocketMultivariate(num_kernels=num_kernels, random_state=random_state)
        self.fitted = False

    @staticmethod
    def _to_panel(x_np):
        # [N,T,F] -> [N,F,T]
        return np.transpose(x_np, (0, 2, 1)).astype(np.float32, copy=False)

    def fit(self, x_np):
        Xp = self._to_panel(x_np)
        _ = self.rocket.fit_transform(Xp)
        self.fitted = True

    def transform(self, x_np):
        # auto-fit fallback for first call in a fresh model instance
        if not self.fitted:
            self.fit(x_np)
        Xp = self._to_panel(x_np)
        Z = self.rocket.transform(Xp)
        return Z.to_numpy(dtype=np.float32) if hasattr(Z, 'to_numpy') else np.asarray(Z, dtype=np.float32)

class Stage3ABCMXTCN(nn.Module):
    """
    MiniRocket + trainable heads.
    Name kept for compatibility with existing code paths.
    """
    def __init__(self, num_features: int, hidden: int = 64, num_kernels: int = 10000, random_state: int = 42):
        super().__init__()
        self.backbone = _MiniRocketBackbone(num_kernels=num_kernels, random_state=random_state)
        # compatibility: existing code touches .net and .net[2]
        self.net = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])
        # Lazy layers infer MiniRocket feature dimension at first forward
        self.head_reject = nn.LazyLinear(2)
        self.head_bcmx = nn.LazyLinear(4)

    def fit_backbone(self, x_np):
        self.backbone.fit(x_np)

    def forward(self, x):
        # x: [N,T,F]
        if isinstance(x, torch.Tensor):
            dev = x.device
            x_np = x.detach().float().cpu().numpy()
        else:
            dev = torch.device('cpu')
            x_np = np.asarray(x, dtype=np.float32)

        z_np = self.backbone.transform(x_np)
        z = torch.from_numpy(z_np).to(dev)
        return self.head_reject(z), self.head_bcmx(z)

def fit_minirocket_from_loader(model, loader, max_batches=20):
    xs = []
    print(f"[MR FIT] collecting up to {max_batches} batches...", flush=True)
    for bi, (X, _) in enumerate(loader):
        xs.append(X.numpy())
        if (bi + 1) % 5 == 0:
            print(f"[MR FIT] collected {bi+1} batches", flush=True)
        if bi + 1 >= max_batches:
            break
    X_fit = np.concatenate(xs, axis=0)
    print(f"[MR FIT] fitting on X shape={X_fit.shape}", flush=True)
    model.fit_backbone(X_fit)
    print("[MR FIT] done", flush=True)

# Patch global class symbol so existing training functions instantiate MiniRocket model
globals()['Stage3ABCMXTCN'] = Stage3ABCMXTCN
print('Patched Stage3ABCMXTCN -> MiniRocket version')




def save_stage3_minirocket(model, pt_path):
    torch.save({"model_state": model.state_dict()}, pt_path)
    rocket_path = pt_path.replace(".pt", ".rocket.joblib")
    dump(model.backbone.rocket, rocket_path)
    print(f"[MR SAVE] {pt_path}", flush=True)
    print(f"[MR SAVE] {rocket_path}", flush=True)


def load_stage3_minirocket(model, pt_path, map_location="cpu"):
    ck = torch.load(pt_path, map_location=map_location)
    model.load_state_dict(_extract_state_any(ck), strict=True)
    rocket_path = pt_path.replace(".pt", ".rocket.joblib")
    if os.path.exists(rocket_path):
        model.backbone.rocket = load(rocket_path)
        model.backbone.fitted = True
        print(f"[MR LOAD] {pt_path}", flush=True)
        print(f"[MR LOAD] {rocket_path}", flush=True)
    else:
        print(f"[MR LOAD] rocket file missing, fallback auto-fit on first forward: {rocket_path}", flush=True)
    return model


# ==============================================================================
# Stage3B MiniRocket runner helper
# Source: Handling_Class_Imbalance_MiniRocket_Stage3.ipynb cell 7
# ==============================================================================
# Run Stage3B in a NEW folder without touching your existing outputs
def run_stage3b_stable_minirocket_newfolder(cfg, subdir='m_fix_minirocket_v1'):
    old = cfg.out_root
    cfg.out_root = os.path.join(old, subdir)
    os.makedirs(cfg.out_root, exist_ok=True)
    try:
        return run_stage3b_stable(cfg)
    finally:
        cfg.out_root = old

print('Ready: call run_stage3b_stable_minirocket_newfolder(cfg_m_fix)')


# ==============================================================================
# Stage3B MiniRocket initial timed config
# Source: Handling_Class_Imbalance_MiniRocket_Stage3.ipynb cell 8
# ==============================================================================
import os, time, json, traceback
import numpy as np
from collections import deque, Counter
import random


# -------- paths --------
s3a_ckpt = r"runs_pytorch_stage2_fix\stage3a_minirocket_v1\W24_H24_stage3a_reject_only\stage3a_reject_last.pt"
s3a_rocket = s3a_ckpt.replace(".pt", ".rocket.joblib")
s3a_val_npz = r"runs_pytorch_stage2_fix\stage3a_minirocket_v1\W24_H24_stage3a_reject_only\val_arrays_stage3a.npz"
s1_ckpt = r"runs_pytorch_stage2_fix\W24_H24_two_stage_bal_other_stage1\stage1_last.pt"

for p in [s1_ckpt, s3a_ckpt, s3a_rocket, s3a_val_npz]:
    print(f"[CHECK] {p} -> {os.path.exists(p)}")
    if not os.path.exists(p):
        raise FileNotFoundError(p)

# -------- config --------
cfg_mr = Stage3BStableCfg(
    root4=r"section3_windows_forecast_24_fixed_section4_fe_ext_compact_TRUE",
    W=24, H=24,
    out_root=r"runs_pytorch_stage2_fix",
    s1_ckpt_path=s1_ckpt,
    s3a_ckpt_path=s3a_ckpt,
)

cfg_mr.s3a_val_npz = s3a_val_npz
cfg_mr.train_bcsev = False
cfg_mr.use_fixed_eval = False

# memory/speed friendly
cfg_mr.batch_files = 1
cfg_mr.per_file = 48
cfg_mr.fixed_batch_size = 128
cfg_mr.phaseB_quota_total = 96


# quick time-check run (increase later for full training)
cfg_mr.phaseA_epochs = 4
cfg_mr.phaseB_epochs = 4
cfg_mr.phaseA_steps_per_epoch = 80
cfg_mr.t_reject_choices = (0.65, 0.70, 0.75)
cfg_mr.t_conf_grid = tuple(np.round(np.arange(0.0, 0.51, 0.05), 2))


cfg_mr.save_phaseA_each_epoch = True
cfg_mr.save_phaseA_every = 1
cfg_mr.phaseB_quota = (0.10, 0.15, 0.45, 0.30)  # more X quota

# -------- runner with timing --------
def run_stage3b_mr_timed(cfg, subdir="stage3b_minirocket_timecheck_v1"):
    old = cfg.out_root
    cfg.out_root = os.path.join(old, subdir)
    os.makedirs(cfg.out_root, exist_ok=True)
    print("[RUN] out_root =", cfg.out_root)

    runner = (
        run_stage3b_stable_minirocket_newfolder
        if "run_stage3b_stable_minirocket_newfolder" in globals()
        else run_stage3b_stable
    )
    print("[RUN] runner =", runner.__name__)

    t0 = time.time()
    try:
        out = runner(cfg)
    except Exception:
        print("[RUN] FAILED after", round(time.time() - t0, 1), "sec")
        traceback.print_exc()
        raise
    finally:
        cfg.out_root = old

    dt = time.time() - t0
    print("[RUN] DONE in", round(dt / 60.0, 2), "minutes")
    print("[RUN] best =", out["best"])
    return out

out_mr = run_stage3b_mr_timed(cfg_mr)
print(out_mr["best"])


# ==============================================================================
# Stage3B MiniRocket PhaseB-only run
# Source: Handling_Class_Imbalance_MiniRocket_Stage3.ipynb cell 9
# ==============================================================================
import numpy as np, os

# 1) Fix NameError in predict_final_5class_with_conf
ID_FROM_BCMX = np.array([IDX_B, IDX_C, IDX_M, IDX_X], dtype=np.int64)

# 2) PhaseB-only run (reuse already-trained PhaseA checkpoint)
phaseA_last_pt = r"runs_pytorch_stage2_fix\stage3b_minirocket_timecheck_v1\W24_H24_stage3b_stable_minirocket_v1\stage3b_phaseA_last.pt"
phaseA_last_rocket = phaseA_last_pt.replace(".pt", ".rocket.joblib")
assert os.path.exists(phaseA_last_pt), phaseA_last_pt
assert os.path.exists(phaseA_last_rocket), phaseA_last_rocket

# keep baseline from Stage3A run
cfg_mr.s3a_val_npz = r"runs_pytorch_stage2_fix\stage3a_minirocket_v1\W24_H24_stage3a_reject_only\val_arrays_stage3a.npz"

# start Stage3B from PhaseA_last
cfg_mr.s3a_ckpt_path = phaseA_last_pt

# skip PhaseA
cfg_mr.phaseA_epochs = 0
cfg_mr.save_phaseA_each_epoch = False
cfg_mr.save_phaseA_last = False

# run
out_mr = run_stage3b_mr_timed(cfg_mr, subdir="stage3b_minirocket_phaseB_only_v1")
print(out_mr["best"])
