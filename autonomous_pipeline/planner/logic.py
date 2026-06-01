from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from autonomous_pipeline.table_io import ensure_dir, read_json, write_json

STEPS = ("step1", "step2", "step3", "step4", "step5", "step6")
EXPECTED_OUTPUTS = {
    "step1": "step1_results",
    "step2": "step2_results/next_input/input.csv",
    "step3": "step3_results/next_input/filtered.csv",
    "step4": "step4_results/next_input/filtered.csv",
    "step5": "step5_results/next_input",
    "step6": "step6_results/next_input/latest_dataset.json",
}


def append_tool_trace(output_root: str | Path, tool: str, arguments: dict[str, Any], payload: dict[str, Any], started_at: float) -> str:
    root = ensure_dir(Path(output_root) / "_meta")
    path = root / "tool_trace.json"
    try:
        trace = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    except Exception:
        trace = []
    trace.append(
        {
            "tool": tool,
            "arguments": {k: _compact(v, 240) for k, v in arguments.items() if k != "reason"},
            "status": payload.get("status"),
            "summary": payload.get("summary"),
            "artifacts": payload.get("artifacts", {}),
            "issues": payload.get("issues", []),
            "duration_ms": int((time.time() - started_at) * 1000),
            "finished_at": datetime.now().isoformat(),
        }
    )
    write_json(path, trace)
    return str(path)


def _compact(value: Any, limit: int = 500) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str) if isinstance(value, (dict, list)) else str(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def resolve_step_input(round_root: str | Path, step: str, original_input: str | Path) -> dict[str, Any]:
    root = Path(round_root).expanduser().resolve()
    if step == "step1":
        input_path = Path(original_input).expanduser().resolve()
    elif step == "step2":
        input_path = root / "step1_results"
    elif step == "step3":
        input_path = root / "step2_results" / "next_input" / "input.csv"
    elif step == "step4":
        input_path = root / "step3_results" / "next_input" / "filtered.csv"
    elif step == "step5":
        input_path = root / "step4_results" / "next_input" / "filtered.csv"
    elif step == "step6":
        input_path = root / "step5_results" / "next_input"
    else:
        raise ValueError(f"Unsupported step: {step}")
    return {"step": step, "input_path": str(input_path), "exists": input_path.exists()}


def run_child_step(step: str, input_path: str, output_root: str, task_text: str = "", task_spec_path: str = "", retries: int = 2, timeout_seconds: int | None = None, env_overrides: dict[str, str] | None = None) -> dict[str, Any]:
    if step not in STEPS:
        raise ValueError(f"Unsupported step: {step}")
    resolved_input = Path(input_path).expanduser()
    if not resolved_input.exists():
        raise FileNotFoundError(f"{step} 输入不存在: {resolved_input}")
    task_payload = task_text or ""
    if task_spec_path:
        task_payload = f"{task_payload}\nTaskSpec 文件：{task_spec_path}".strip()
    cmd = [
        sys.executable,
        "-m",
        "autonomous_pipeline.cli",
        "run-step",
        "--step",
        step,
        "--input",
        str(resolved_input),
        "--task",
        task_payload,
        "--output-root",
        str(Path(output_root).expanduser()),
        "--retries",
        str(int(retries)),
    ]
    env = os.environ.copy()
    if env_overrides:
        env.update({str(k): str(v) for k, v in env_overrides.items()})
    started = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=str(Path(__file__).resolve().parents[2]),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    output_lines: list[str] = []
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", file=sys.stderr, flush=True)
            output_lines.append(line.rstrip("\n"))
        returncode = proc.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        proc.kill()
        returncode = proc.wait()
        output_lines.append(f"[Planner] {step} timeout after {timeout_seconds}s")
    tail = output_lines[-80:]
    expected = Path(output_root).expanduser() / EXPECTED_OUTPUTS[step]
    return {
        "step": step,
        "input_path": str(resolved_input),
        "output_root": str(Path(output_root).expanduser()),
        "returncode": int(returncode),
        "duration_ms": int((time.time() - started) * 1000),
        "expected_output": str(expected),
        "expected_output_exists": expected.exists(),
        "stdout_tail": tail,
    }


def inspect_round_artifacts(round_root: str | Path) -> dict[str, Any]:
    root = Path(round_root).expanduser().resolve()
    outputs: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for step, rel in EXPECTED_OUTPUTS.items():
        path = root / rel
        outputs[step] = {"path": str(path), "exists": path.exists()}
        if not path.exists():
            missing.append(step)
    patient_cases = root / "step4_results" / "next_input" / "patient_cases.jsonl"
    outputs["patient_cases"] = {"path": str(patient_cases), "exists": patient_cases.exists()}
    if not patient_cases.exists():
        missing.append("patient_cases")
    return {"round_root": str(root), "outputs": outputs, "missing": missing, "complete": not missing}


def write_round_summary(round_root: str | Path, planner_response: str = "", feedback_path: str = "") -> dict[str, Any]:
    root = ensure_dir(round_root)
    artifacts = inspect_round_artifacts(root)
    summary = {
        "round_root": str(Path(root).resolve()),
        "planner_response": planner_response,
        "feedback_path": feedback_path,
        "artifacts": artifacts,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    summary_path = write_json(Path(root) / "round_summary.json", summary)
    return {"round_summary_path": summary_path, **artifacts}


def create_run_root(output_root: str | Path, run_id: str | None = None) -> Path:
    stamp = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    return ensure_dir(Path(output_root).expanduser().resolve() / "planner_runs" / stamp)


def load_public_feedback(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path).expanduser()
    if not p.is_file():
        return {}
    data = read_json(p)
    return data if isinstance(data, dict) else {}
