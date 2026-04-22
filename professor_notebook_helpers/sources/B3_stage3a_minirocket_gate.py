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
# Stage3A MiniRocket config + run
# Source: Handling_Class_Imbalance_MiniRocket_Stage3.ipynb cell 6
# ==============================================================================
# Optional helper: run Stage3A MiniRocket in a new folder

def run_stage3a_reject_only_minirocket(cfg, subdir='stage3a_minirocket_v1'):
    old = cfg.out_root
    cfg.out_root = os.path.join(old, subdir)
    os.makedirs(cfg.out_root, exist_ok=True)
    try:
        return run_stage3a_reject_only(cfg)
    finally:
        cfg.out_root = old

# Example (uncomment):
cfg_s3a_mr = Stage3ARejectCfg(
    root4=r'section3_windows_forecast_24_fixed_section4_fe_ext_compact_TRUE',
    W=24, H=24,
    out_root=r'runs_pytorch_stage2_fix',
    s1_ckpt_path=r'runs_pytorch_stage2_fix\W24_H24_two_stage_bal_other_stage1\stage1_last.pt',
)
cfg_s3a_mr.max_train_files = 200
cfg_s3a_mr.max_val_files = 80
cfg_s3a_mr.per_file = 96
cfg_s3a_mr.s3a_epochs = 4

cfg_s3a_mr.s3a_k_pos = 384
cfg_s3a_mr.s3a_neg_ratio = 2.0
cfg_s3a_mr.t_flare_train = 0.04

cfg_s3a_mr.t_reject_grid = tuple(np.round(np.arange(0.20, 0.91, 0.05), 2))
cfg_s3a_mr.max_quiet_to_flare_fpr = 0.15
cfg_s3a_mr.min_pred_quiet_rate = 0.25

res_s3a_mr = run_stage3a_reject_only_minirocket(cfg_s3a_mr)
print(res_s3a_mr)
