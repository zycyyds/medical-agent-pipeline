import sys
import types
from pathlib import Path


def _install_layout_stubs():
    if "doclayout_yolo" not in sys.modules:
        doclayout_stub = types.ModuleType("doclayout_yolo")

        class DummyYOLOv10:
            def __init__(self, *args, **kwargs):
                pass

        doclayout_stub.YOLOv10 = DummyYOLOv10
        sys.modules["doclayout_yolo"] = doclayout_stub

    if "PyQt5" not in sys.modules:
        pyqt5_stub = types.ModuleType("PyQt5")
        qtwidgets_stub = types.ModuleType("PyQt5.QtWidgets")
        qtgui_stub = types.ModuleType("PyQt5.QtGui")
        qtcore_stub = types.ModuleType("PyQt5.QtCore")

        for name in [
            "QApplication", "QMainWindow", "QLabel", "QPushButton", "QVBoxLayout",
            "QHBoxLayout", "QWidget", "QMessageBox", "QSizePolicy",
        ]:
            setattr(qtwidgets_stub, name, type(name, (), {}))

        qtgui_stub.QPixmap = type("QPixmap", (), {})
        qtcore_stub.Qt = type("Qt", (), {})

        sys.modules["PyQt5"] = pyqt5_stub
        sys.modules["PyQt5.QtWidgets"] = qtwidgets_stub
        sys.modules["PyQt5.QtGui"] = qtgui_stub
        sys.modules["PyQt5.QtCore"] = qtcore_stub


def test_process_table_file_drops_blank_columns(tmp_path):
    _install_layout_stubs()
    import pandas as pd
    from agent_1.layout_analysis_tool import _process_table_file

    src = tmp_path / "input.csv"
    dst = tmp_path / "output.csv"
    pd.DataFrame(
        {
            "id": ["1", "2"],
            "empty_all_blank": ["", "   "],
            "empty_all_null": [None, None],
            "value": ["a", "b"],
        }
    ).to_csv(src, index=False)

    ok = _process_table_file(str(src), str(dst))

    assert ok is True
    cleaned = pd.read_csv(dst)
    assert list(cleaned.columns) == ["id", "value"]


def test_layout_inference_uses_batch_predict_when_outputs_disabled(tmp_path, monkeypatch):
    _install_layout_stubs()
    from agent_1 import layout_analysis_tool as tool

    image_a = tmp_path / "a.jpg"
    image_b = tmp_path / "b.jpg"
    calls = []

    class FakeResult:
        def tojson(self):
            return "[]"

    class FakeModel:
        def predict(self, image_paths, **kwargs):
            calls.append(image_paths)
            if not isinstance(image_paths, list):
                raise AssertionError("expected batch image path list")
            return [FakeResult() for _ in image_paths]

    monkeypatch.setattr(tool, "_get_model", lambda model_path=None: FakeModel())

    classification, processed = tool._run_inference_and_save(
        file_entries=[(str(image_a), "a.jpg"), (str(image_b), "b.jpg")],
        out_img_dir=str(tmp_path / "out_img"),
        out_json_dir=str(tmp_path / "out_json"),
        conf=0.2,
        imgsz=960,
        device="cpu",
        model_path=None,
        figure_class_id=3,
        figure_ratio_threshold=0.5,
        save_outputs=False,
    )

    assert calls == [[str(image_a), str(image_b)]]
    assert processed == 2
    assert [item["relative_path"] for item in classification] == ["a.jpg", "b.jpg"]
