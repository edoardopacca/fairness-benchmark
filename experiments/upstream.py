"""Load only the pinned upstream files shipped in third_party/; no network or Git needed."""
import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "third_party"
sys.dont_write_bytecode = True


def provenance():
    manifest = json.loads((VENDOR / "manifest.json").read_text())
    records = {}
    for name, record in manifest.items():
        for relative, item in record["files"].items():
            path = ROOT / relative
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                raise RuntimeError(f"Pinned upstream file changed or missing: {relative}")
        records[name] = {"commit": record["commit"], "origin": record["origin"], "status": "",
                         "files": {p: entry["sha256"] for p, entry in record["files"].items()}}
    return records


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


provenance()  # Verify original source bytes before executing any vendored code.
mary = load_file("_mary_hgr", VENDOR / "mary/hgr.py")
kde = load_file("_mary_kde", VENDOR / "mary/pytorch_kde.py").kde


def upstream_static(class_name, method_name, namespace):
    """Execute the original function body, without unrelated Moving Targets imports."""
    path = VENDOR / "gedi/metrics.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    fn.decorator_list = []
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[method_name]


_ns = {"np": np, "torch": torch, "Any": object}
weights = upstream_static("RegressionWeights", "get_weights", _ns)
_ns["RegressionWeights"] = SimpleNamespace(get_weights=weights)
gedi = upstream_static("GeneralizedDIDI", "generalized_didi", _ns)


def fairgroups_class():
    """Keep upstream algorithms unchanged; bypass its unused visualization imports."""
    package_name = "_benchmark_fairgroups"
    module_name = package_name + ".partition_estimation"
    if module_name not in sys.modules:
        package = ModuleType(package_name)
        package.__path__ = [str(VENDOR / "fairgroups")]
        package.__package__ = package_name
        sys.modules[package_name] = package
        load_file(module_name, VENDOR / "fairgroups/partition_estimation.py")
    return sys.modules[module_name].FairGroups
