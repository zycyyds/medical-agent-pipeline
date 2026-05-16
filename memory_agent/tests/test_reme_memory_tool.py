import json

import pytest

from memory_agent import reme_memory_tool as tool


def _payload(response):
    return json.loads(response.content)


@pytest.fixture()
def isolated_case_store(tmp_path, monkeypatch):
    cases_path = tmp_path / "reme_memory" / "cases" / "step1_script_cases.jsonl"
    monkeypatch.setattr(tool, "STEP1_SCRIPT_CASES_PATH", cases_path)
    monkeypatch.setattr(tool, "REME_MEMORY_ROOT", cases_path.parents[1])
    return cases_path


def test_build_step1_input_profile_for_directory_and_single_file(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "notes.csv").write_text("subject_id,value\n1,2\n", encoding="utf-8")
    image_dir = input_dir / "images"
    image_dir.mkdir()
    (image_dir / "scan.png").write_bytes(b"png")

    directory_profile = _payload(tool.build_step1_input_profile(str(input_dir)))
    assert directory_profile["exists"] is True
    assert directory_profile["path_type"] == "directory"
    assert directory_profile["file_count"] == 2
    assert directory_profile["suffix_counts"] == {".csv": 1, ".png": 1}
    assert "notes.csv" in directory_profile["sample_files"]

    single_profile = _payload(tool.build_step1_input_profile(str(input_dir / "notes.csv")))
    assert single_profile["path_type"] == "file"
    assert single_profile["file_count"] == 1
    assert single_profile["suffix_counts"] == {".csv": 1}


def test_initialize_reme_memory_store_creates_jsonl_store(isolated_case_store):
    response = _payload(tool.initialize_reme_memory_store())

    assert response["status"] == "OK"
    assert response["backend"] == "jsonl_fallback"
    assert isolated_case_store.is_file()
    assert response["storage_exists"] is True


def test_build_step1_input_profile_for_missing_path(tmp_path):
    profile = _payload(tool.build_step1_input_profile(str(tmp_path / "missing")))

    assert profile["exists"] is False
    assert profile["path_type"] == "missing"
    assert profile["file_count"] == 0
    assert profile["suffix_counts"] == {}


def test_record_step1_script_case_writes_metadata_without_script_body(tmp_path, isolated_case_store):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "table.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    script_path = tmp_path / "generated_reorganizer.py"
    script_path.write_text("print('do not store full body')\n", encoding="utf-8")
    profile = _payload(tool.build_step1_input_profile(str(input_dir)))

    response = _payload(
        tool.record_step1_script_case(
            {
                "input_profile": profile,
                "script_path": str(script_path),
                "execution": {"status": "SUCCESS", "output_root": "program/output/step1_results"},
                "summary": "csv-only step1 case",
            }
        )
    )

    assert response["status"] == "RECORDED"
    assert response["case"]["memory_type"] == "step1_script_case"
    assert response["case"]["schema_version"] == tool.REME_CASE_SCHEMA_VERSION
    assert response["case"]["script_hash"]
    assert response["case"]["script_size"] == script_path.stat().st_size
    assert "do not store full body" not in json.dumps(response, ensure_ascii=False)

    stored = [json.loads(line) for line in isolated_case_store.read_text(encoding="utf-8").splitlines()]
    assert len(stored) == 1
    assert stored[0]["case_id"] == response["case"]["case_id"]
    assert "script_content" not in stored[0]


def test_record_step1_script_case_is_idempotent_for_same_case(tmp_path, isolated_case_store):
    script_path = tmp_path / "script.py"
    script_path.write_text("print('ok')\n", encoding="utf-8")
    payload = {
        "input_profile": {"profile_hash": "same", "path_type": "file", "file_count": 1, "suffix_counts": {".csv": 1}},
        "script_path": str(script_path),
        "summary": "same case",
    }

    first = _payload(tool.record_step1_script_case(payload))
    second = _payload(tool.record_step1_script_case(payload))

    assert first["status"] == "RECORDED"
    assert second["status"] == "EXISTS"
    assert first["case"]["case_id"] == second["case"]["case_id"]
    assert len(isolated_case_store.read_text(encoding="utf-8").splitlines()) == 1


def test_retrieve_step1_script_case_ranks_by_suffix_and_file_count(tmp_path, isolated_case_store):
    csv_profile = {"exists": True, "path_type": "directory", "file_count": 2, "suffix_counts": {".csv": 2}}
    image_profile = {"exists": True, "path_type": "directory", "file_count": 2, "suffix_counts": {".png": 2}}
    script_path = tmp_path / "script.py"
    script_path.write_text("print('ok')\n", encoding="utf-8")

    tool.record_step1_script_case({"input_profile": image_profile, "script_path": str(script_path), "summary": "image"})
    tool.record_step1_script_case({"input_profile": csv_profile, "script_path": str(script_path), "summary": "csv"})

    result = _payload(tool.retrieve_step1_script_case(csv_profile, limit=2))

    assert result["status"] == "OK"
    assert [case["summary"] for case in result["candidates"]] == ["csv", "image"]
    assert result["candidates"][0]["similarity_score"] > result["candidates"][1]["similarity_score"]


def test_list_and_status_work_for_empty_and_existing_store(tmp_path, isolated_case_store):
    empty_status = _payload(tool.get_reme_memory_status())
    assert empty_status["backend"] == "jsonl_fallback"
    assert empty_status["case_count"] == 0
    assert empty_status["real_reme_available"] is False

    empty_list = _payload(tool.list_step1_script_cases())
    assert empty_list["cases"] == []

    script_path = tmp_path / "script.py"
    script_path.write_text("print('ok')\n", encoding="utf-8")
    tool.record_step1_script_case(
        {
            "input_profile": {"exists": True, "path_type": "file", "file_count": 1, "suffix_counts": {".csv": 1}},
            "script_path": str(script_path),
            "summary": "one case",
        }
    )

    status = _payload(tool.get_reme_memory_status())
    listed = _payload(tool.list_step1_script_cases(limit=1))

    assert status["case_count"] == 1
    assert listed["cases"][0]["summary"] == "one case"
