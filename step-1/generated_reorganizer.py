from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(r"/Users/mkbk/PycharmProjects/new")
STEP1_DIR = PROJECT_ROOT / "step-1"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(STEP1_DIR) not in sys.path:
    sys.path.insert(0, str(STEP1_DIR))

from step1_runtime import run_reorganize_from_records


if __name__ == "__main__":
    result = run_reorganize_from_records(
        records_path=Path(r"/Users/mkbk/PycharmProjects/new/reorganized_output/_meta/records.json"),
        output_root=Path(r"/Users/mkbk/PycharmProjects/new/program/output/step1_results"),
        input_root=Path(r"/Users/mkbk/PycharmProjects/new/mimic-10") if r"/Users/mkbk/PycharmProjects/new/mimic-10" else None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result.get("status") == "SUCCESS" else 1)
