"""Read-only inventory. Does not import torch/numpy or manage packages."""
import ast
from importlib import metadata, util
import json
from pathlib import Path
import platform
import sys


def collect():
    root = Path(__file__).resolve().parents[1]
    packages = {}
    for name in ("numpy", "torch", "comfyui-frontend-package"):
        try:
            version = metadata.version(name)
        except metadata.PackageNotFoundError:
            version = None
        packages[name] = {"version": version}
        if name in ("numpy", "torch"):
            packages[name]["module_found"] = util.find_spec(name) is not None
    syntax_errors = []
    files = list(root.glob("*.py")) + list((root/"tools").glob("*.py"))
    for path in files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeError, OSError) as exc:
            syntax_errors.append({"file": str(path.relative_to(root)), "error": str(exc)})
    return {
        "python_executable": sys.executable, "python_version": platform.python_version(),
        "platform": platform.platform(), "packages": packages,
        "syntax_errors": syntax_errors,
        "comfyui_core_version": "not detected; record from actual host",
        "gpu_cuda_runtime": "not probed; torch is not imported",
        "compatibility_status": "not_verified",
        "note": "只读清点，不修改环境；包版本存在不代表二进制兼容或工作流可运行。",
    }


if __name__ == "__main__":
    report = collect()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    missing = any(not report["packages"][name]["module_found"]
                  or report["packages"][name]["version"] is None for name in ("numpy", "torch"))
    sys.exit(1 if report["syntax_errors"] or missing else 0)
