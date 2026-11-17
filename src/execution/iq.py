from src.orchestration.state import IQOutput, State, Task, ToolCaller, VerificationOutput


def measure(state, task, call, output, record_label):
    return call("evaluate_image_quality", inputs=task.input_artifacts.model_dump(mode="json"),
                image_id=output.get("artifact_id") if output.get("rgb", True) else None,
                gamma=output.get("gamma", 2.2), sensor_results=state.get("sensor_results"),
                label=record_label)["metrics"]


def run(state: State, task: Task, call: ToolCaller) -> IQOutput:
    results = dict(state.get("iq_results", {}))
    if state["task_type"] == "optimization" and not state.get("isp_results"):
        if not task.input_artifacts.reference or not task.input_artifacts.uniform_roi:
            raise ValueError("Multi-objective optimization requires an aligned reference and uniform ROI")
        return {"iq_results": results}
    sensor_label = "sensor_followup" if state.get("review", {}).get("action") == "expand" else "sensor"
    outputs = state.get("isp_results", {}) or {sensor_label: {}}
    for label, output in outputs.items():
        if label not in results:
            results[label] = measure(state, task, call, output, label)
    return {"iq_results": results}


def verify(state: State, task: Task, call: ToolCaller) -> VerificationOutput:
    label = state["review"]["target"]
    original = state["iq_results"][label]
    output = state.get("isp_results", {}).get(label, {})
    measured = measure(state, task, call, output, f"verification:{label}")
    matches = all((reading["value"], reading["valid"]) ==
                  (original[name]["value"], original[name]["valid"])
                  for name, reading in measured.items())
    return {"verification": {"target": label, "matches": matches, "metrics": measured}}

