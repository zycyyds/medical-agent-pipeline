from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from agentscope.message import Msg

from autonomous_pipeline.evaluator import agent as evaluator_agent
from autonomous_pipeline.evaluator import logic as evaluator_logic
from autonomous_pipeline.table_io import ensure_dir, read_json, write_json
from autonomous_pipeline.task_analysis import agent as task_analysis_agent
from autonomous_pipeline.task_analysis import logic as task_analysis_logic

from . import agent as planner_agent
from . import logic as planner_logic


def _message_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "\n".join(str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in content)
    return str(content)


def _clean_model_text(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    return match.group(1).strip() if match else text


async def _call_agent(create_agent, prompt: str) -> str:
    agent = create_agent()
    response = await agent(Msg(name="user", role="user", content=prompt))
    return _clean_model_text(_message_text(response))


async def run_planner_loop(
    input_path: str,
    task_text: str,
    gold_dir: str,
    output_root: str,
    max_rounds: int = 3,
    run_id: str | None = None,
) -> dict[str, Any]:
    run_root = planner_logic.create_run_root(output_root, run_id=run_id)
    run_state: dict[str, Any] = {
        "run_id": run_root.name,
        "input_path": str(Path(input_path).expanduser().resolve()),
        "task_text": task_text,
        "gold_dir": str(Path(gold_dir).expanduser().resolve()),
        "run_root": str(run_root),
        "max_rounds": int(max_rounds),
        "rounds": [],
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(run_root / "run_state.json", run_state)

    task_prompt = task_analysis_agent.build_task_prompt(input_path=input_path, output_root=str(run_root), task_text=task_text, gold_dir=gold_dir)
    task_response = await _call_agent(task_analysis_agent.create_agent, task_prompt)
    task_spec_path = run_root / "task_spec.json"
    if not task_spec_path.is_file():
        # Fallback keeps the loop recoverable if the model failed to call the final tool.
        task_analysis_logic.build_task_spec(input_path, gold_dir, run_root, task_text=task_text)
    run_state["task_analysis_response"] = task_response
    run_state["task_spec_path"] = str(task_spec_path)
    write_json(run_root / "run_state.json", run_state)

    previous_score = 0.0
    no_improvement = 0
    retained: dict[str, Any] = {}
    feedback_path = ""
    for round_index in range(1, int(max_rounds) + 1):
        round_root = ensure_dir(run_root / "rounds" / f"round_{round_index:02d}")
        prompt = planner_agent.build_task_prompt(
            original_input=input_path,
            round_root=str(round_root),
            task_text=task_text,
            task_spec_path=str(task_spec_path),
            public_feedback_path=feedback_path,
            previous_score=previous_score,
        )
        planner_response = await _call_agent(planner_agent.create_agent, prompt)
        planner_logic.write_round_summary(round_root, planner_response=planner_response, feedback_path=feedback_path)

        evaluation_root = ensure_dir(run_root / "evaluations" / f"round_{round_index:02d}")
        eval_prompt = evaluator_agent.build_task_prompt(str(round_root), gold_dir, str(evaluation_root), previous_score=previous_score)
        eval_response = await _call_agent(evaluator_agent.create_agent, eval_prompt)
        public_feedback = evaluation_root / "public_feedback.json"
        if not public_feedback.is_file():
            evaluator_logic.evaluate_round_outputs(round_root, gold_dir, evaluation_root, previous_score=previous_score)
        feedback = read_json(public_feedback) if public_feedback.is_file() else {}
        score = float(feedback.get("score", 0.0) or 0.0)
        improved = bool(feedback.get("improved", score > previous_score + 0.01))
        if improved:
            retained = {"round": round_index, "round_root": str(round_root), "score": score, "public_feedback_path": str(public_feedback)}
            previous_score = score
            no_improvement = 0
        else:
            no_improvement += 1

        round_record = {
            "round": round_index,
            "round_root": str(round_root),
            "planner_response": planner_response,
            "evaluator_response": eval_response,
            "public_feedback_path": str(public_feedback),
            "score": score,
            "improved": improved,
        }
        run_state["rounds"].append(round_record)
        write_json(run_root / "run_state.json", run_state)

        if not feedback.get("continue_recommended", False) or no_improvement >= 2:
            break

    final_result = {
        "status": "SUCCESS" if retained else "NEEDS_REPAIR",
        "summary": "Planner loop completed.",
        "run_root": str(run_root),
        "task_spec_path": str(task_spec_path),
        "retained_result": retained,
        "round_count": len(run_state["rounds"]),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(run_root / "retained_result.json", retained)
    final_path = write_json(run_root / "final_result.json", final_result)
    run_state["final_result_path"] = final_path
    write_json(run_root / "run_state.json", run_state)
    return final_result


def run_planner_loop_sync(input_path: str, task_text: str, gold_dir: str, output_root: str, max_rounds: int = 3, run_id: str | None = None) -> dict[str, Any]:
    return asyncio.run(run_planner_loop(input_path=input_path, task_text=task_text, gold_dir=gold_dir, output_root=output_root, max_rounds=max_rounds, run_id=run_id))
