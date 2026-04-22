# ==============================================================================
# Transformer Stage3 model definition
# Source: Handling_Class_Imbalance_MiniRocket_Stage3.ipynb cell 11
# ==============================================================================
# ===== Transformer Stage3 model (drop-in replacement) =====
import os
import math
import numpy as np
import torch
import torch.nn as nn


class _TransformerBackbone(nn.Module):
    def __init__(self, num_features, d_model=64, nhead=4, num_layers=2, dim_ff=128, dropout=0.1, max_len=256):
        super().__init__()
        self.in_proj = nn.Linear(num_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.max_len = max_len

    def forward(self, x):
        # x: [N,T,F]
        n, t, _ = x.shape
        if t > self.max_len:
            x = x[:, :self.max_len, :]
            t = self.max_len
        h = self.in_proj(x) + self.pos[:, :t, :]
        h = self.encoder(h)
        h = self.norm(h)
        return h.mean(dim=1)  # [N,d_model]


class Stage3ABCMXTCN(nn.Module):
    """
    Transformer version, name kept for compatibility with existing runner code.
    """
    def __init__(self, num_features: int, hidden: int = 64, layers: int = 2, nhead: int = 4, dim_ff: int = 128):
        super().__init__()
        self.net = _TransformerBackbone(
            num_features=num_features,
            d_model=int(hidden),
            nhead=int(nhead),
            num_layers=int(layers),
            dim_ff=int(dim_ff),
            dropout=0.1,
            max_len=256,
        )
        self.head_reject = nn.Linear(int(hidden), 2)
        self.head_bcmx = nn.Linear(int(hidden), 4)

    def forward(self, x):
        h = self.net(x)
        return self.head_reject(h), self.head_bcmx(h)


# Re-attach compatibility alias AFTER latest Stage3ABCMXTCN definition
def _attach_backbone_alias():
    cls = Stage3ABCMXTCN
    def _get_backbone(self):
        return self.net
    def _set_backbone(self, v):
        self.net = v
    cls.backbone = property(_get_backbone, _set_backbone)

_attach_backbone_alias()

# sanity check
m = Stage3ABCMXTCN(num_features=176, hidden=64)
print("has backbone:", hasattr(m, "backbone"))
print("backbone type:", type(m.backbone))




# avoid previous NameError in your Stage3B code path
ID_FROM_BCMX = np.array([IDX_B, IDX_C, IDX_M, IDX_X], dtype=np.int64)


def run_stage3a_reject_only_transformer(cfg, subdir="stage3a_transformer_v1"):
    old = cfg.out_root
    cfg.out_root = os.path.join(old, subdir)
    os.makedirs(cfg.out_root, exist_ok=True)
    try:
        return run_stage3a_reject_only(cfg)
    finally:
        cfg.out_root = old


def run_stage3b_stable_transformer_newfolder(cfg, subdir="stage3b_transformer_v1"):
    old = cfg.out_root
    cfg.out_root = os.path.join(old, subdir)
    os.makedirs(cfg.out_root, exist_ok=True)
    try:
        return run_stage3b_stable(cfg)
    finally:
        cfg.out_root = old


import os
import torch

# Works for both MiniRocket and Transformer models
def fit_minirocket_from_loader(model, loader, max_batches=20):
    if not hasattr(model, "fit_backbone"):
        print("[MR FIT] skipped (model has no fit_backbone).")
        return

    import numpy as np
    xs = []
    print(f"[MR FIT] collecting up to {max_batches} batches...", flush=True)
    for bi, (X, y) in enumerate(loader):
        xs.append(X.numpy())
        if (bi + 1) % 5 == 0:
            print(f"[MR FIT] collected {bi+1} batches", flush=True)
        if bi + 1 >= max_batches:
            break
    X_fit = np.concatenate(xs, axis=0)
    print(f"[MR FIT] fitting on X shape={X_fit.shape}", flush=True)
    model.fit_backbone(X_fit)
    print("[MR FIT] done", flush=True)


import os
import torch
from joblib import dump, load

def save_stage3_minirocket(model, pt_path):
    os.makedirs(os.path.dirname(pt_path), exist_ok=True)
    torch.save({"model_state": model.state_dict()}, pt_path)

    rocket_path = pt_path.replace(".pt", ".rocket.joblib")
    has_mr = hasattr(model, "backbone") and hasattr(model.backbone, "rocket")

    if has_mr:
        dump(model.backbone.rocket, rocket_path)
        print(f"[MR SAVE] {pt_path}")
        print(f"[MR SAVE] {rocket_path}")
    else:
        print(f"[SAVE] {pt_path} (no MiniRocket artifact)")

def load_stage3_minirocket(model, pt_path, map_location="cpu"):
    ck = torch.load(pt_path, map_location=map_location)
    model.load_state_dict(_extract_state_any(ck), strict=True)

    rocket_path = pt_path.replace(".pt", ".rocket.joblib")
    has_mr = hasattr(model, "backbone") and hasattr(model.backbone, "rocket")

    if has_mr:
        if os.path.exists(rocket_path):
            model.backbone.rocket = load(rocket_path)
            model.backbone.fitted = True
            print(f"[MR LOAD] {pt_path}")
            print(f"[MR LOAD] {rocket_path}")
        else:
            raise FileNotFoundError(f"Expected MiniRocket file not found: {rocket_path}")
    else:
        print(f"[LOAD] {pt_path} (no MiniRocket artifact needed)")

    return model


# ==============================================================================
# Stage3A Transformer config + run
# Source: Handling_Class_Imbalance_MiniRocket_Stage3.ipynb cell 12 section
# ==============================================================================
# ===== Stage3A Transformer =====
cfg_s3a_tx = Stage3ARejectCfg(
    root4=r"section3_windows_forecast_24_fixed_section4_fe_ext_compact_TRUE",
    W=24, H=24,
    out_root=r"runs_pytorch_stage2_fix",
    s1_ckpt_path=r"runs_pytorch_stage2_fix\W24_H24_two_stage_bal_other_stage1\stage1_last.pt",
)
cfg_s3a_tx.hidden = 64
cfg_s3a_tx.s3a_epochs = 6
cfg_s3a_tx.max_train_files = 200
cfg_s3a_tx.max_val_files = 80
cfg_s3a_tx.per_file = 64
cfg_s3a_tx.t_flare_train = 0.04
cfg_s3a_tx.s3a_k_pos = 384
cfg_s3a_tx.s3a_neg_ratio = 1.25
cfg_s3a_tx.use_fixed_eval = False

res_s3a_tx = run_stage3a_reject_only_transformer(cfg_s3a_tx, subdir="stage3a_transformer_v2")
print(res_s3a_tx)
