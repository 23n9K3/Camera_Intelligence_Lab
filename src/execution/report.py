from src.orchestration.state import State, Task, ReportOutput, ToolCaller


def run(state: State, task: Task, call: ToolCaller) -> ReportOutput:
    summary = {key: state[key] for key in ("task_type", "sensor_results", "isp_results",
                                          "iq_results", "optimization_results", "model_decisions",
                                          "review", "verification") if key in state}
    reports = {
        "html": call("generate_experiment_report", summary=summary)["artifact_id"],
        "csv": call("export_experiment_csv")["artifact_id"],
        "json": call("export_experiment_json", summary=summary)["artifact_id"],
    }
    return {"report_artifact_id": reports["html"], "reports": reports}

