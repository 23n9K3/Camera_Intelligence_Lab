import json
from pathlib import Path
from shutil import copytree

import numpy as np
from streamlit.testing.v1 import AppTest

from src.camera.models import RawSpec
from src.camera.raw_reader import load_capture
from src.camera.synthetic_data import generate
from src.orchestration.graph import Run
from src.orchestration.state import Task
from src.storage.store import Store
from src.utils import config
from src.visualization.workbench import PAGES, save_dataset


def test_dataset_manager_shows_raw_pixel_histogram(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    copytree(root / "configs", tmp_path / "configs")
    monkeypatch.setattr(config, "ROOT", tmp_path)
    store = Store(tmp_path / "artifacts")

    app = AppTest.from_file(str(root / "app.py"), default_timeout=45).run()
    app.sidebar.radio[0].set_value("Dataset Manager").run()
    next(widget for widget in app.selectbox if widget.label == "Bayer Pattern").set_value("GBRG")
    next(button for button in app.button if button.label == "生成数据集").click().run()

    assert not app.exception, [error.message for error in app.exception]
    datasets = [row for row in store.rows("artifacts") if row["payload"]["kind"] == "dataset"]
    raw_id = datasets[0]["payload"]["metadata"]["inputs"]["raw"]
    raw, spec = load_capture(store, raw_id)
    assert spec.bayer_pattern == "GBRG"

    histogram = json.loads(app.get("plotly_chart")[0].proto.spec)
    assert histogram["layout"]["title"]["text"] == "RAW 直方图"
    trace = histogram["data"][0]
    assert sum(trace["y"]) == raw.size
    bin_width = (1 << spec.bit_depth) // 256
    np.testing.assert_array_equal(trace["y"], np.bincount(raw.ravel() // bin_width, minlength=256))


def test_load_all_pages_and_run_workspace(tmp_path, monkeypatch):
    # Leave the developer's persistent SQLite database untouched.
    root = Path(__file__).resolve().parents[2]
    sensor = config.configuration("sensor")
    copytree(root / "configs", tmp_path / "configs")
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setenv("CAMERA_LLM_MODE", "mock")
    store = Store(tmp_path / "artifacts")
    inputs = generate(store, RawSpec(**sensor))
    save_dataset(store, "QA dataset", "synthetic", inputs)
    for kind in ("sensor", "comparison", "optimization"):
        state = Run(store, Task(user_request=kind, task_type=kind, input_artifacts=inputs,
                                trials=2), mock=True).execute()
        assert state["status"] == "succeeded", state["errors"]
    app = AppTest.from_file(str(root / "app.py"), default_timeout=45).run()
    for page in PAGES:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, (page, [error.message for error in app.exception])
    app.sidebar.radio[0].set_value("Agent Workspace").run()
    app.text_area[0].set_value("分析暗场与平场")
    next(button for button in app.button if button.label == "运行任务").click().run()
    assert not app.exception
    assert store.rows("experiments")[-1]["payload"]["status"] == "succeeded"

