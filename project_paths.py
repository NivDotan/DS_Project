from __future__ import annotations

import os
from pathlib import Path


def get_project_root() -> Path:
    env_root = os.getenv("SOLAR_FLARE_PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    here = Path(__file__).resolve().parent
    anchors = {
        "Multi-Class Solar Flare Prediction Notebook.ipynb",
        "section6_runtime_bundle_minimal.py",
        "stage1_w72_notebook_block.py",
    }

    for candidate in [here, *here.parents]:
        if any((candidate / anchor).exists() for anchor in anchors):
            return candidate
    return here


PROJECT_ROOT = get_project_root()
