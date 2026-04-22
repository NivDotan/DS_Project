# ==============================================================================
# Base Stage3B Transformer config
# Source: Handling_Class_Imbalance_PhysicsInformed_Stage3.ipynb cell 14
# ==============================================================================
# ===== Stage3B Transformer =====
s3a_ckpt_tx = r"runs_pytorch_stage2_fix\stage3a_transformer_best_029_025\W24_H24_stage3a_reject_only\stage3a_reject_last.pt"
s3a_val_npz_tx = r"runs_pytorch_stage2_fix\stage3a_transformer_best_029_025\W24_H24_stage3a_reject_only\val_arrays_stage3a.npz"

cfg_tx = Stage3BStableCfg(
    root4=r"section3_windows_forecast_24_fixed_section4_fe_ext_compact_TRUE",
    W=24, H=24,
    out_root=r"runs_pytorch_stage2_fix",
    s1_ckpt_path=r"runs_pytorch_stage2_fix\W24_H24_two_stage_bal_other_stage1\stage1_last.pt",
    s3a_ckpt_path=s3a_ckpt_tx,
)

cfg_tx.s3a_ckpt_path = s3a_ckpt_tx
cfg_tx.s3a_val_npz = s3a_val_npz_tx
cfg_tx.train_bcsev = False
cfg_tx.use_fixed_eval = False

cfg_tx.batch_files = 1
cfg_tx.per_file = 32
cfg_tx.fixed_batch_size = 128
cfg_tx.phaseA_epochs = 15
cfg_tx.phaseB_epochs = 8
cfg_tx.phaseA_steps_per_epoch = 80

# כאן אל תרוץ עם reject choices לא קשורים
cfg_tx.t_reject_choices = (0.29,)
cfg_tx.t_conf_grid = tuple(np.round(np.arange(0.0, 0.51, 0.05), 2))
cfg_tx.phaseB_quota_total = 256

out_tx = run_stage3b_stable_transformer_newfolder(
    cfg_tx,
    subdir="stage3b_transformer_best_from_029_025"
)
print(out_tx["best"])


# ==============================================================================
# Physics-informed config helpers
# Source: Handling_Class_Imbalance_PhysicsInformed_Stage3.ipynb cell 25
# ==============================================================================
from dataclasses import dataclass
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

@dataclass
class PhysicsInformedCfg:
    feat_idx: tuple = ()          # optional feature indices used for activity proxy
    lambda_rank: float = 0.08    # ranking penalty weight
    lambda_quiet_low: float = 0.05
    low_quantile: float = 0.2
    margin: float = 0.02
    max_pairs: int = 256
    epochs: int = 3
    lr_backbone: float = 1e-4
    lr_head: float = 5e-4

def _physics_proxy_from_x(X: torch.Tensor, feat_idx: tuple = ()) -> torch.Tensor:
    # X: [N,T,F], proxy reflects magnetic activity level + temporal variation
    if len(feat_idx) > 0:
        V = X[:, :, list(feat_idx)]
    else:
        V = X
    dV = torch.diff(V, dim=1)
    level = V.abs().mean(dim=(1, 2))
    variation = dV.abs().mean(dim=(1, 2))
    return variation + 0.1 * level

def _physics_rank_loss(p_severe: torch.Tensor, proxy: torch.Tensor, margin: float = 0.02, max_pairs: int = 256) -> torch.Tensor:
    n = int(p_severe.numel())
    if n < 4:
        return p_severe.new_tensor(0.0)

    order = torch.argsort(proxy)
    k = min(max_pairs, n // 2)
    if k <= 0:
        return p_severe.new_tensor(0.0)

    lo = order[:k]
    hi = order[-k:]
    diff = p_severe[hi] - p_severe[lo]
    return F.relu(float(margin) - diff).mean()

def _build_stage3_loaders_from_cfg(cfg):
    man_dir = os.path.join(cfg.root4, "manifests_by_group")
    man_train = os.path.join(man_dir, f"manifest_train_W{cfg.W}h_H{cfg.H}h.json")
    man_val = os.path.join(man_dir, f"manifest_val_W{cfg.W}h_H{cfg.H}h.json")
    man_test = os.path.join(man_dir, f"manifest_test_W{cfg.W}h_H{cfg.H}h.json")

    ds_train = NPZFileDataset(man_train, name2id, max_files=cfg.max_train_files, allow_pickle=True)
    ds_val = NPZFileDataset(man_val, name2id, max_files=cfg.max_val_files, allow_pickle=True)

    X0, _ = ds_train[0]
    T_fixed = int(X0.shape[1])
    Fdim = int(X0.shape[2])

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

    return dl_train, dl_val, dl_test, Fdim


# ==============================================================================
# Physics-informed finetune runner
# Source: Handling_Class_Imbalance_PhysicsInformed_Stage3.ipynb cell 26
# ==============================================================================
def run_stage3b_physics_informed_finetune(
    cfg,
    base_stage3b_ckpt_path: str,
    t_reject: float,
    t_conf: float,
    pi_cfg: PhysicsInformedCfg = PhysicsInformedCfg(),
    subdir: str = "stage3_physics_informed_v1",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(cfg.seed)
    print("device:", device)

    out_root_old = cfg.out_root
    cfg.out_root = os.path.join(out_root_old, subdir)
    out_dir = os.path.join(cfg.out_root, f"W{cfg.W}_H{cfg.H}_stage3b_physics_informed")
    os.makedirs(out_dir, exist_ok=True)

    dl_train, dl_val, dl_test, Fdim = _build_stage3_loaders_from_cfg(cfg)
    if dl_val is None:
        raise RuntimeError("Validation loader missing.")

    # frozen Stage1
    ckpt1 = torch.load(cfg.s1_ckpt_path, map_location=device)
    model_s1_frozen = SimpleTCN(num_features=Fdim, num_classes=2, hidden=cfg.hidden).to(device)
    model_s1_frozen.load_state_dict(_extract_state_any(ckpt1))
    model_s1_frozen.eval()
    for p in model_s1_frozen.parameters():
        p.requires_grad = False

    # frozen Stage3A gate
    model_s3a_frozen = Stage3ABCMXTCN(num_features=Fdim, hidden=cfg.hidden).to(device)
    model_s3a_frozen = load_stage3_minirocket(model_s3a_frozen, cfg.s3a_ckpt_path, map_location=device)
    model_s3a_frozen.eval()
    for p in model_s3a_frozen.parameters():
        p.requires_grad = False

    # load base Stage3B model to continue from
    model_pi = Stage3ABCMXTCN(num_features=Fdim, hidden=cfg.hidden).to(device)
    model_pi = load_stage3_minirocket(model_pi, base_stage3b_ckpt_path, map_location=device)

    for p in model_pi.head_reject.parameters():
        p.requires_grad = False
    for p in model_pi.head_bcmx.parameters():
        p.requires_grad = True
    if hasattr(model_pi, "net"):
        for p in model_pi.net.parameters():
            p.requires_grad = True

    backbone_params = list(model_pi.net.parameters()) if hasattr(model_pi, "net") else []
    opt = torch.optim.Adam([
        {"params": backbone_params, "lr": float(pi_cfg.lr_backbone)},
        {"params": model_pi.head_bcmx.parameters(), "lr": float(pi_cfg.lr_head)},
    ])

    w = torch.tensor(cfg.bcmx_ce_weights, dtype=torch.float32, device=device)
    ce4 = nn.CrossEntropyLoss(weight=w)

    for ep in range(1, int(pi_cfg.epochs) + 1):
        t0 = time.perf_counter()
        model_pi.train()
        total, n = 0.0, 0
        seen_b = seen_c = seen_m = seen_x = 0

        for bi, (X, y5) in enumerate(dl_train, start=1):
            X = X.to(device, non_blocking=True)
            y5 = y5.to(device, non_blocking=True)

            with torch.no_grad():
                surv = _frozen_gate_survivor_mask(model_s1_frozen, model_s3a_frozen, X, cfg.t_flare, t_reject)

            if surv.sum().item() == 0:
                continue

            Xs = X[surv]
            ys = y5[surv]
            is_flare = (ys != IDX_Q)

            loss = Xs.new_tensor(0.0)

            # supervised CE on flare survivors
            if is_flare.any():
                Xf = Xs[is_flare]
                yf = ys[is_flare]
                idxq = quota_sample_bcmx_min(
                    ys5=yf,
                    total=cfg.phaseB_quota_total,
                    quota=cfg.phaseB_quota,
                    m_min=cfg.phaseB_m_min,
                    x_min=cfg.phaseB_x_min,
                )
                if idxq is not None:
                    Xq, yq = Xf[idxq], yf[idxq]
                    _, logits4_q = model_pi(Xq)
                    y4 = y5_to_bcmx_index_torch(yq)
                    loss = loss + ce4(logits4_q, y4)

                    seen_b += int((yq == IDX_B).sum().item())
                    seen_c += int((yq == IDX_C).sum().item())
                    seen_m += int((yq == IDX_M).sum().item())
                    seen_x += int((yq == IDX_X).sum().item())

            # physics-informed regularization on survivors
            _, logits4_all = model_pi(Xs)
            p4 = torch.softmax(logits4_all, dim=1)
            p_sev = p4[:, 2] + p4[:, 3]  # M + X

            proxy = _physics_proxy_from_x(Xs, pi_cfg.feat_idx)
            rank_loss = _physics_rank_loss(
                p_sev, proxy, margin=pi_cfg.margin, max_pairs=pi_cfg.max_pairs
            )
            loss = loss + float(pi_cfg.lambda_rank) * rank_loss

            q_low = torch.quantile(proxy.detach(), float(pi_cfg.low_quantile))
            low_quiet = (proxy <= q_low) & (ys == IDX_Q)
            if low_quiet.any():
                quiet_low_pen = p_sev[low_quiet].mean()
                loss = loss + float(pi_cfg.lambda_quiet_low) * quiet_low_pen

            if not torch.isfinite(loss):
                continue

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_pi.parameters(), 1.0)
            opt.step()

            total += float(loss.item()) * Xs.size(0)
            n += Xs.size(0)

            if bi % 20 == 0:
                print(f"[PI Ep {ep}] batch={bi} n={n} B/C/M/X={seen_b}/{seen_c}/{seen_m}/{seen_x}", flush=True)

        ep_sec = time.perf_counter() - t0
        print(
            f"[PI Ep {ep}] loss={total/max(n,1):.4f} | seen B/C/M/X={seen_b}/{seen_c}/{seen_m}/{seen_x} | {ep_sec:.1f}s",
            flush=True,
        )

    ckpt_out = os.path.join(out_dir, f"stage3b_physics_informed_trej_{float(t_reject):.2f}.pt")
    save_stage3_minirocket(model_pi, ckpt_out)

    y_val, pred_val = predict_final_5class_with_conf(
        model_s1_frozen, model_s3a_frozen, model_pi,
        dl_val, device, cfg.t_flare, float(t_reject), float(t_conf)
    )
    val_metrics = eval_5class_summary(y_val, pred_val)

    test_metrics = None
    if dl_test is not None:
        y_test, pred_test = predict_final_5class_with_conf(
            model_s1_frozen, model_s3a_frozen, model_pi,
            dl_test, device, cfg.t_flare, float(t_reject), float(t_conf)
        )
        test_metrics = eval_5class_summary(y_test, pred_test)

    out = {
        "out_dir": out_dir,
        "base_stage3b_ckpt_path": base_stage3b_ckpt_path,
        "t_flare": float(cfg.t_flare),
        "t_reject": float(t_reject),
        "t_conf_bcmx": float(t_conf),
        "ckpt_path": ckpt_out,
        "physics_cfg": pi_cfg.__dict__,
        "val": val_metrics,
        "test": test_metrics,
    }

    with open(os.path.join(out_dir, "stage3b_physics_informed_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Saved:", os.path.join(out_dir, "stage3b_physics_informed_summary.json"))
    print("Best-like output:", {"val": out["val"], "test": out["test"]})

    cfg.out_root = out_root_old
    return out


# ==============================================================================
# Physics-informed sweep incl. pi_severe
# Source: Handling_Class_Imbalance_PhysicsInformed_Stage3.ipynb cell 28
# ==============================================================================
# ===== Physics-informed sweep: 3 configs, separate output folders =====
# Assumes:
# - run_stage3b_physics_informed_finetune(...)
# - PhysicsInformedCfg
# - cfg_tx already defined
# - base stage3b ckpt exists

import os, json, time
import numpy as np
import copy

base_ckpt = r"runs_pytorch_stage2_fix\stage3b_transformer_v3\W24_H24_stage3b_stable_minirocket_v1\stage3b_stable_trej_0.40.pt"
assert os.path.exists(base_ckpt), base_ckpt

# Keep same operating point as your best stage3b_transformer_v3
T_REJECT = 0.40
T_CONF = 0.30

# Use fixed eval for fair comparison across sweep runs (if fixed arrays exist)
cfg_sweep = copy.deepcopy(cfg_tx)
cfg_sweep.use_fixed_eval = True

# Optional: add more C pressure
cfg_sweep.phaseB_quota = (0.10, 0.30, 0.40, 0.20)  # B,C,M,X
cfg_sweep.use_fixed_eval = False

sweep = [
    ("pi_balanced", PhysicsInformedCfg(
        feat_idx=(),
        lambda_rank=0.05,
        lambda_quiet_low=0.08,
        low_quantile=0.2,
        margin=0.02,
        max_pairs=256,
        epochs=3,
        lr_backbone=1e-4,
        lr_head=5e-4,
    )),
    ("pi_severe", PhysicsInformedCfg(
        feat_idx=(),
        lambda_rank=0.10,
        lambda_quiet_low=0.04,
        low_quantile=0.2,
        margin=0.02,
        max_pairs=256,
        epochs=3,
        lr_backbone=1e-4,
        lr_head=5e-4,
    )),
    ("pi_fpr", PhysicsInformedCfg(
        feat_idx=(),
        lambda_rank=0.03,
        lambda_quiet_low=0.10,
        low_quantile=0.2,
        margin=0.02,
        max_pairs=256,
        epochs=3,
        lr_backbone=1e-4,
        lr_head=5e-4,
    )),
]

all_out = []
for name, pi_cfg in sweep:
    print(f"\n===== RUN {name} =====")
    t0 = time.time()

    out = run_stage3b_physics_informed_finetune(
        cfg=cfg_sweep,
        base_stage3b_ckpt_path=base_ckpt,
        t_reject=T_REJECT,
        t_conf=T_CONF,
        pi_cfg=pi_cfg,
        subdir=f"stage3_physics_informed_{name}",
    )

    dt_min = (time.time() - t0) / 60.0
    rec = {
        "name": name,
        "minutes": round(dt_min, 2),
        "ckpt_path": out.get("ckpt_path"),
        "val_severe_recall": out["val"]["severe_recall"],
        "val_quiet_to_flare_fpr": out["val"]["quiet_to_flare_fpr"],
        "val_c_recall": out["val"].get("c_recall", 0.0),
        "test_severe_recall": None if out["test"] is None else out["test"]["severe_recall"],
        "test_quiet_to_flare_fpr": None if out["test"] is None else out["test"]["quiet_to_flare_fpr"],
        "test_c_recall": None if out["test"] is None else out["test"].get("c_recall", 0.0),
        "summary_json": os.path.join(out["out_dir"], "stage3b_physics_informed_summary.json"),
    }
    all_out.append(rec)
    print("[DONE]", rec)

# Save sweep summary
sweep_root = r"runs_pytorch_stage2_fix\stage3_physics_informed_v1"
os.makedirs(sweep_root, exist_ok=True)
summary_path = os.path.join(sweep_root, "physics_informed_sweep_summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(all_out, f, indent=2)

print("\nSaved sweep summary:", summary_path)
print(json.dumps(all_out, indent=2))
