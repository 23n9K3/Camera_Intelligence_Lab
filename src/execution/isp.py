from src.isp.pipeline import ISPConfig


from src.orchestration.state import State, Task, ISPOutput, ToolCaller


def run(state: State, task: Task, call: ToolCaller) -> ISPOutput:
    if task.isp_stage:
        result = call("run_isp_stage", stage_name=task.isp_stage,
                      artifact_id=task.stage_input or task.input_artifacts.raw,
                      config=task.isp_config.model_dump(mode="json"))
        return {"isp_results": {task.isp_stage: result}}
    inputs = task.input_artifacts
    kind = state["task_type"]
    raw_id = inputs.raw
    normalized = kind in {"hdr", "burst"}
    if kind == "hdr":
        raw_id = call("hdr_fusion", artifact_ids=inputs.exposures,
                      exposure_times=inputs.exposure_times)["artifact_id"]
    elif kind == "burst":
        raw_id = call("burst_average", artifact_ids=inputs.burst)["artifact_id"]

    algorithms = state["experiment_plan"]["algorithms"]
    if kind == "comparison":
        variants = {algorithm: task.isp_config.model_copy(update={"demosaic": algorithm})
                    for algorithm in algorithms}
    elif state.get("trial"):
        variants = {state["label"]: ISPConfig(**state["isp_config"])}
    else:
        variants = {"baseline": task.isp_config.model_copy(
            update={"demosaic": algorithms[0]} if algorithms else {})}
    outputs = {label: call("run_isp_pipeline", artifact_id=raw_id,
                           config=config.model_dump(mode="json"), normalized=normalized)
               for label, config in variants.items()}
    return {"isp_results": {**state.get("isp_results", {}), **outputs}}

