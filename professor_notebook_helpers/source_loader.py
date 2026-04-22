from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterable


def exec_selected_into_namespace(
    py_path: str | Path,
    ns: dict,
    wanted_assign_names: Iterable[str] | None = None,
) -> dict:
    py_path = Path(py_path)
    source = py_path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(source, filename=str(py_path))
    wanted = set(wanted_assign_names or [])
    selected_nodes = []

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
            selected_nodes.append(node)
        elif isinstance(node, ast.Assign) and wanted:
            names = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if names & wanted:
                selected_nodes.append(node)

    module = ast.Module(body=selected_nodes, type_ignores=[])
    ns.setdefault("__file__", str(py_path))
    ns.setdefault("__name__", f"__bundle_exec_{py_path.stem}__")
    exec(compile(module, str(py_path), "exec"), ns)
    return ns


def make_module_namespace(module_name: str, file_hint: str | Path) -> dict:
    file_hint = Path(file_hint)
    module_obj = ModuleType(module_name)
    module_obj.__file__ = str(file_hint)
    module_obj.__name__ = module_name
    sys.modules[module_name] = module_obj
    ns = module_obj.__dict__
    ns["__file__"] = str(file_hint)
    ns["__name__"] = module_name
    return ns
