from __future__ import annotations

import copy as copy_module
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

import stage1_w72_notebook_block
import stage2_notebook_results
import stage3_notebook_results
from project_paths import PROJECT_ROOT

from .paths import source_path
from .source_loader import exec_selected_into_namespace, make_module_namespace


CORE_ASSIGN_NAMES = {
    "classes",
    "name2id",
    "id2name",
    "K",
    "IDX_Q",
    "IDX_B",
    "IDX_C",
    "IDX_M",
    "IDX_X",
}

BLOCK_ALIASES = {
    "stage1_core": "stage1_core",
    "stage3a_bilstm": "stage3a_bilstm",
    "stage3a_transformer": "stage3a_transformer",
    "stage3a_minirocket": "stage3a_minirocket",
    "stage3b_tcn": "stage3b_tcn",
    "stage3b_bilstm": "stage3b_tcn",
    "stage3b_transformer": "stage3b_transformer",
    "stage3_transformer": "stage3b_transformer",
    "stage3b_physics_informed": "stage3b_physics_informed",
    "stage3_physics_informed": "stage3b_physics_informed",
    "stage3_minirocket": "stage3_minirocket",
    "stage3b_minirocket": "stage3_minirocket",
}


def _new_namespace(label: str) -> dict[str, Any]:
    return make_module_namespace(f"__prof_bundle_{label}__", source_path("stage1_core_context_cell8.py"))


def _build_stage1_core_ns() -> dict[str, Any]:
    ns = _new_namespace("stage1_core")
    exec_selected_into_namespace(source_path("stage1_core_context_cell8.py"), ns, wanted_assign_names=CORE_ASSIGN_NAMES)
    return ns


def _build_stage3a_bilstm_ns() -> dict[str, Any]:
    ns = _build_stage1_core_ns()
    exec_selected_into_namespace(source_path("B1_stage3a_bilstm_gate.py"), ns, wanted_assign_names={"ID_FROM_BCMX"})
    return ns


def _build_stage3b_tcn_ns() -> dict[str, Any]:
    ns = _build_stage3a_bilstm_ns()
    exec_selected_into_namespace(source_path("C1_stage3b_bilstm_stable.py"), ns, wanted_assign_names={"ID_FROM_BCMX"})
    return ns


def _build_stage3b_transformer_ns() -> dict[str, Any]:
    ns = _build_stage3b_tcn_ns()
    exec_selected_into_namespace(source_path("B2_stage3a_transformer_gate.py"), ns, wanted_assign_names={"ID_FROM_BCMX"})
    return ns


def _build_stage3b_physics_ns() -> dict[str, Any]:
    ns = _build_stage3b_transformer_ns()
    exec_selected_into_namespace(
        source_path("C3_stage3b_physics_informed_pi_severe.py"),
        ns,
        wanted_assign_names={"s3a_ckpt_tx", "s3a_val_npz_tx", "cfg_tx"},
    )
    return ns


def _build_stage3b_minirocket_ns() -> dict[str, Any]:
    ns = _build_stage3b_tcn_ns()
    exec_selected_into_namespace(source_path("B3_stage3a_minirocket_gate.py"), ns)
    exec_selected_into_namespace(source_path("C4_stage3b_minirocket_phaseB_only.py"), ns)
    return ns


@lru_cache(maxsize=None)
def _get_block_namespace(block_name: str) -> dict[str, Any]:
    canonical = BLOCK_ALIASES.get(block_name, block_name)
    builders = {
        "stage1_core": _build_stage1_core_ns,
        "stage3a_bilstm": _build_stage3a_bilstm_ns,
        "stage3a_transformer": _build_stage3b_transformer_ns,
        "stage3a_minirocket": _build_stage3b_minirocket_ns,
        "stage3b_tcn": _build_stage3b_tcn_ns,
        "stage3b_transformer": _build_stage3b_transformer_ns,
        "stage3b_physics_informed": _build_stage3b_physics_ns,
        "stage3_minirocket": _build_stage3b_minirocket_ns,
    }
    if canonical not in builders:
        raise KeyError(f"Unknown helper block: {block_name}")
    return builders[canonical]()


def _exported_symbols_from(ns: dict[str, Any], names: list[str]) -> dict[str, Any]:
    out = {}
    for name in names:
        if name in ns:
            out[name] = ns[name]
    return out


def build_default_cfg_tx():
    physics_ns = _get_block_namespace("stage3b_physics_informed")
    cfg = physics_ns.get("cfg_tx")
    if cfg is None:
        raise KeyError("cfg_tx was not found in the physics-informed helper block.")
    return copy_module.deepcopy(cfg)


def _load_proxy_function(block_name: str, symbol_name: str):
    def _wrapped(*args, **kwargs):
        ns = _get_block_namespace(block_name)
        return ns[symbol_name](*args, **kwargs)

    _wrapped.__name__ = symbol_name
    return _wrapped


DEFAULT_EXPORTED_SYMBOLS = {
    **_exported_symbols_from(
        _get_block_namespace("stage3b_tcn"),
        [
            "ExperimentCfg",
            "ExperimentCfgSkipStage1",
            "run_stage1_only",
            "Stage3ARejectCfg",
            "run_stage3a_reject_only",
            "Stage3BStableCfg",
            "run_stage3b_stable",
            "IDX_Q",
            "IDX_B",
            "IDX_C",
            "IDX_M",
            "IDX_X",
        ],
    ),
    **_exported_symbols_from(
        _get_block_namespace("stage3b_physics_informed"),
        [
            "PhysicsInformedCfg",
            "run_stage3a_reject_only_transformer",
            "run_stage3b_stable_transformer_newfolder",
            "run_stage3b_physics_informed_finetune",
        ],
    ),
    "run_stage3a_reject_only_minirocket": _load_proxy_function("stage3_minirocket", "run_stage3a_reject_only_minirocket"),
    "run_stage3b_stable_minirocket_newfolder": _load_proxy_function("stage3_minirocket", "run_stage3b_stable_minirocket_newfolder"),
    "cfg_tx": build_default_cfg_tx(),
    "copy": copy_module,
}


def inject_notebook_symbols(target: dict[str, Any] | None = None) -> dict[str, Any]:
    symbols = {name: value for name, value in DEFAULT_EXPORTED_SYMBOLS.items()}
    if target is not None:
        target.update(symbols)

    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is not None:
            ip.user_ns.update(symbols)
    except Exception:
        pass

    return symbols


def available_blocks() -> list[str]:
    return sorted(set(BLOCK_ALIASES))


def load_block(block_name: str, target: dict[str, Any] | None = None) -> dict[str, Any]:
    ns = _get_block_namespace(block_name)
    exported = {k: v for k, v in ns.items() if not k.startswith("__")}
    if target is not None:
        target.update(exported)
    return exported


def lightgbm_results() -> dict[str, Path | pd.DataFrame]:
    base = PROJECT_ROOT / "lightgbm_run_inventory_20260412"
    return {
        "base_dir": base,
        "comparison_matrix": pd.read_csv(base / "lightgbm_comparison_matrix.csv"),
        "all_metric_rows": pd.read_csv(base / "lightgbm_all_metric_rows.csv"),
        "precision_recall_compare": pd.read_csv(base / "lightgbm_precision_recall_compare.csv"),
    }


def stage3a_results() -> dict[str, pd.DataFrame]:
    return {
        "results_2424": stage2_notebook_results.get_stage2_results_2424(),
        "results_w72": stage2_notebook_results.get_stage2_results_w72(),
    }


def stage3b_results() -> dict[str, pd.DataFrame]:
    return {
        "results_2424": stage3_notebook_results.get_stage3_results_2424(),
        "results_w72": stage3_notebook_results.get_stage3_results_w72(),
    }


def temporal_results() -> dict[str, Path]:
    base = PROJECT_ROOT / "temporal_agg_cache"
    return {
        "base_dir": base,
        "w72_h72": base / "w72_h72_bilstm_trej_015_safe",
        "w72_h48": base / "w72_h48_bilstm_trej_020_safe",
        "w72_h24": base / "w72_h24_bilstm_trej_020_safe",
        "best_2424_temporal": base / "temporal_merge_rerun_bilstm_x_hybrid020_v1",
    }
