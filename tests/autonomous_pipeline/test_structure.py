from __future__ import annotations

import ast
import inspect
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "autonomous_pipeline"
STEPS = [f"step{i}" for i in range(1, 7)]
FORBIDDEN_IMPORT_PREFIXES = ("agent_", "main_orchestrator")
FORBIDDEN_IMPORT_NAMES = {"step1_agent", "step23_agent", "step4_agent", "step5_agent", "step6_agent", "step7_agent"}


def _python_files():
    return [p for p in PKG.rglob("*.py") if "__pycache__" not in p.parts]


def test_step_directories_have_required_entry_files():
    for step in STEPS:
        step_dir = PKG / "steps" / step
        assert step_dir.is_dir(), step
        for filename in ("agent.py", "tools.py", "prompt.py", "logic.py"):
            assert (step_dir / filename).is_file(), f"{step}/{filename} missing"


def test_no_required_tool_sequence_in_autonomous_package():
    offenders = []
    for path in _python_files():
        if "required_tool_sequence" in path.read_text(encoding="utf-8"):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_no_legacy_module_imports():
    offenders = []
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                first = name.split(".", 1)[0]
                if first.startswith(FORBIDDEN_IMPORT_PREFIXES) or first in FORBIDDEN_IMPORT_NAMES:
                    offenders.append(f"{path.relative_to(ROOT)} imports {name}")
    assert offenders == []


def test_tool_functions_have_args_docstrings():
    for step in STEPS:
        module = __import__(f"autonomous_pipeline.steps.{step}.tools", fromlist=["TOOL_FUNCTIONS"])
        assert module.TOOL_FUNCTIONS, step
        for func in module.TOOL_FUNCTIONS:
            doc = inspect.getdoc(func) or ""
            assert "Args:" in doc, f"{step}.{func.__name__} missing Args docstring"
            assert len(inspect.signature(func).parameters) >= 1, f"{step}.{func.__name__} should expose arguments"
