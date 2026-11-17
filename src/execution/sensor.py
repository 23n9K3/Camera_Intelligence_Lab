from src.orchestration.state import State, Task, SensorOutput, ToolCaller


def run(state: State, task: Task, call: ToolCaller) -> SensorOutput:
    inputs = task.input_artifacts
    checks = state["experiment_plan"]["sensor_checks"]
    result = dict(state.get("sensor_results", {}))
    if not result:
        result.update(call("analyze_dark_frame", artifact_ids=inputs.dark))
    if "flat" in checks:
        result.update(call("analyze_flat_field", artifact_ids=inputs.flat, dark_ids=inputs.dark))
    if "ptc" in checks:
        result.update(call("analyze_ptc", groups=inputs.ptc, dark_ids=inputs.dark))
    return {"sensor_results": result}

