from src.orchestration.state import State, Task, DataOutput, ToolCaller


def run(state: State, task: Task, call: ToolCaller) -> DataOutput:
    inputs = task.input_artifacts
    kind = state["task_type"]
    if kind == "sensor" and not inputs.dark:
        raise ValueError("Sensor analysis requires dark frames")
    if kind in {"comparison", "optimization"} and not inputs.raw and not (
            task.isp_stage and task.stage_input):
        raise ValueError("This workflow requires a RAW image")
    if kind == "hdr" and len(inputs.exposures) < 2:
        raise ValueError("HDR requires an exposure sequence")
    if kind == "burst" and len(inputs.burst) < 2:
        raise ValueError("Burst denoising requires a burst sequence")
    ids = [key for key in (inputs.raw, *inputs.dark, *inputs.flat,
                           *(key for group in inputs.ptc for key in group),
                           *inputs.exposures, *inputs.burst) if key]
    captures = {key: call("load_raw", artifact_id=key) for key in dict.fromkeys(ids)}
    preview = call("visualize_bayer", artifact_id=inputs.raw) if inputs.raw else None
    extras = {key: call("parse_metadata", artifact_id=key)
              for key in (inputs.reference, inputs.edge, task.stage_input) if key}
    return {"metadata": {"captures": captures, "preview": preview, "extra": extras}}

