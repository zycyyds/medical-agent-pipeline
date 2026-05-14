from pathlib import Path

import agent_5.main as agent5_main


def test_agent5_standalone_defaults_use_cleaning_paths():
    project_root = Path(agent5_main.get_project_root())

    assert agent5_main.get_default_input_dir() == str(project_root / "agent_5" / "data_input")
    assert agent5_main.get_default_workspace_dir() == str(project_root / "program" / "output" / "step5_results")
