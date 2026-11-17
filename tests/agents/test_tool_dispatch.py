from functools import wraps

import pytest

from src.camera.models import Inputs, RawSpec
from src.camera.synthetic_data import generate
from src.orchestration.graph import Run
from src.orchestration.state import Task
from src.storage.store import Store
from src.utils.config import configuration


@pytest.fixture
def captures(tmp_path):
    store = Store(tmp_path)
    inputs = generate(store, RawSpec(**configuration("sensor")), seed=123)
    return store, inputs


@pytest.mark.parametrize("request_text,route,tool", [
    ("分析暗场与平场的 DSNU、PRNU", "sensor", "analyze_flat_field"),
    ("比较去马赛克输出的 PSNR 和 SSIM", "comparison", "run_isp_pipeline"),
    ("优化低照度 ISP 的降噪参数", "optimization", "finish_trial"),
    ("合并曝光序列生成 HDR 图像", "hdr", "hdr_fusion"),
    ("用 Burst 做多帧降噪", "burst", "burst_average"),
])
def test_natural_language_dispatch_returns_grounded_results(captures, request_text, route, tool):
    store, inputs = captures
    task = Task(user_request=request_text, input_artifacts=inputs, trials=1)
    state = Run(store, task, mock=True).execute()
    assert task.task_type is None
    assert state["status"] == "succeeded", state["errors"]
    assert state["task_type"] == route

    experiment_id = state["experiment_id"]
    calls = [row["payload"] for row in store.rows("tool_calls", experiment_id)]
    assert all(call["status"] == "succeeded" for call in calls)
    selected = next(call for call in calls if call["tool"] == tool)
    output = selected["output"]
    if route == "sensor":
        assert output["prnu_percent"] == state["sensor_results"]["prnu_percent"] > 0
        assert store.array(output["flat_mean_artifact"]).shape == (96, 128)
    else:
        assert store.array(output["artifact_id"]).size > 0
        if route in {"hdr", "burst"}:
            assert state["isp_results"]["baseline"]["input_artifact_id"] == output["artifact_id"]
        elif route == "comparison":
            assert state["isp_results"]["bilinear"]["artifact_id"] == output["artifact_id"]
        else:
            assert output == state["optimization_results"]["trials"][0]

    recorded_metrics = {row["id"]: row["payload"] for row in store.rows("iq_metrics", experiment_id)}
    metric_name = "snr" if route == "sensor" else "psnr"
    metric_group = next(iter(state["iq_results"].values()))
    metric = metric_group[metric_name]
    assert metric["valid"]
    recorded = recorded_metrics[metric["record_id"]]
    assert recorded["name"] == metric_name
    assert recorded["value"] == metric["value"]
    assert recorded["input_artifact_ids"] == metric["input_artifact_ids"]
    assert all(store.path(key).is_file() for key in recorded["input_artifact_ids"])


def test_permanent_decode_failure_is_logged_and_stops_workflow(tmp_path):
    store = Store(tmp_path)
    raw = store.put(b"short", "raw", {"spec": configuration("sensor")}, suffix=".bin")
    task = Task(user_request="比较去马赛克输出", input_artifacts=Inputs(raw=raw))
    state = Run(store, task, mock=True).execute()
    calls = [row["payload"] for row in store.rows("tool_calls", state["experiment_id"])]

    assert state["status"] == "failed"
    assert [(call["tool"], call["status"]) for call in calls] == [("load_raw", "failed")]
    assert "RAW requires" in calls[0]["error"]
    assert state["errors"][-1]["error"] == calls[0]["error"]
    assert "report_artifact_id" not in state


def test_transient_read_error_is_logged_and_retried(captures):
    store, inputs = captures
    task = Task(user_request="比较去马赛克输出", input_artifacts=Inputs(raw=inputs.raw),
                algorithms=["bilinear"])
    run = Run(store, task, mock=True)
    group, schema, load_raw = run.registry.tools["load_raw"]
    attempts = 0

    @wraps(load_raw)
    def fail_once(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("temporary read failure")
        return load_raw(*args, **kwargs)

    run.registry.tools["load_raw"] = group, schema, fail_once
    state = run.execute()
    assert state["status"] == "succeeded", state["errors"]
    traces = [trace for trace in state["agent_trace"] if trace["agent"] == "data"]
    assert [(trace["attempt"], trace["status"]) for trace in traces] == [
        (1, "failed"), (2, "succeeded")]
    calls = [row["payload"] for row in store.rows("tool_calls", state["experiment_id"])
             if row["payload"]["tool"] == "load_raw"]
    assert [call["status"] for call in calls] == ["failed", "succeeded"]
    assert calls[0]["error"] == "OSError: temporary read failure"
    assert calls[1]["output"]["artifact_id"] == inputs.raw
