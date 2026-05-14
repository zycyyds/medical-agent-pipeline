import sys
from pathlib import Path


ORCHESTRATOR_DIR = Path(__file__).resolve().parents[1]
if str(ORCHESTRATOR_DIR) not in sys.path:
    sys.path.insert(0, str(ORCHESTRATOR_DIR))

import step_wrappers


def test_step6_and_step7_results_dirs_are_split(monkeypatch, tmp_path):
    output_root = tmp_path / "program" / "output"
    monkeypatch.setattr(step_wrappers, "_get_program_output_root", lambda: str(output_root))

    assert step_wrappers._get_step67_step6_output_dir() == str(output_root / "step6_results")
    assert step_wrappers._get_step67_step7_output_dir() == str(output_root / "step7_results")
