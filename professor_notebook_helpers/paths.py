from __future__ import annotations

from pathlib import Path

from project_paths import PROJECT_ROOT


HELPER_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = HELPER_ROOT / "sources"


def source_path(name: str) -> Path:
    path = SOURCE_ROOT / name
    if not path.exists():
        raise FileNotFoundError(f"Bundled source file not found: {path}")
    return path
