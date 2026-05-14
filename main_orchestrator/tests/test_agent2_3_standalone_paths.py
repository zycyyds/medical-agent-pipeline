import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[2] / "agent_2-3" / "main_v2.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("agent2_3_main_v2_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_agent2_3_standalone_defaults_use_data_input_and_program_output():
    module = _load_module()
    project_root = MODULE_PATH.parents[1]

    assert module.get_default_input_path() == str(project_root / "agent_2-3" / "data_input")
    assert module.get_default_output_dir() == str(project_root / "program" / "output" / "step2_3_results")
