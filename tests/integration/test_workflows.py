import csv
import json

import pytest

from src.camera.models import Inputs, RawSpec
from src.camera.synthetic_data import generate
from src.orchestration.graph import Run
from src.orchestration.state import Task
from src.storage.store import Store
from src.utils.config import configuration


@pytest.fixture
def fixture(tmp_path):
    store = Store(tmp_path)
    return store, generate(store, RawSpec(**configuration("sensor")), seed=123)


@pytest.mark.parametrize("kind", ["sensor", "comparison", "optimization", "hdr", "burst"])
def test_workflow_reports_and_grounding(fixture, kind):
    store, inputs = fixture
    state = Run(store, Task(user_request=kind, task_type=kind, input_artifacts=inputs,
                            trials=3), mock=True).execute()
    assert state["status"] == "succeeded", state["errors"]
    assert state["agent_trace"][0]["agent"] == "supervisor"
    assert state["agent_trace"][-1]["agent"] == "report"
    exp = state["experiment_id"]
    with store.path(state["reports"]["csv"]).open(encoding="utf-8-sig") as stream:
        csv_rows = list(csv.DictReader(stream))
    db_ids = {row["id"] for row in store.rows("iq_metrics", exp)}
    assert db_ids and db_ids == {row["record_id"] for row in csv_rows}
    exported = json.loads(store.path(state["reports"]["json"]).read_text(encoding="utf-8"))
    assert db_ids == {row["id"] for row in exported["iq_metrics"]}
    assert store.path(state["report_artifact_id"]).is_file()
    if kind == "sensor":
        sensor = state["sensor_results"]
        assert sensor["hot_pixels_xy"] == [[42, 32]]
        assert sensor["dead_pixels_xy"] == [[85, 64]]
        assert sensor["ptc"]["gain_dn_per_e"] == pytest.approx(0.65, abs=0.08)
    if kind == "optimization":
        search = state["optimization_results"]
        assert len(search["trials"]) == 3
        assert search["recommended"]["number"] in search["pareto"]


def test_isp_stage_can_chain_prior_artifacts(fixture):
    store, inputs = fixture
    for name, rgb in [("black_level_correction", False), ("demosaic_bilinear", True),
                      ("gamma_encode", True)]:
        state = Run(store, Task(user_request=name, input_artifacts=inputs, isp_stage=name,
                                stage_input=key if name != "black_level_correction" else None),
                    mock=True).execute()
        assert state["status"] == "succeeded", state["errors"]
        output = state["isp_results"][name]
        key = output["artifact_id"]
        assert output["rgb"] == rgb
        assert len(output["stages"]) == 1
        assert store.array(key).ndim == (3 if rgb else 2)


def test_invalid_artifact_is_logged(tmp_path):
    store = Store(tmp_path)
    task = Task(user_request="comparison", input_artifacts=Inputs(raw="../../invalid"))
    state = Run(store, task, mock=True).execute()
    calls = store.rows("tool_calls", state["experiment_id"])
    assert state["status"] == "failed"
    assert calls[0]["payload"]["status"] == "failed"
    assert "Unknown artifacts ID" in calls[0]["payload"]["error"]

