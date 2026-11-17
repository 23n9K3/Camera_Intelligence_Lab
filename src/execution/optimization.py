from src.optimization.search import objective_values


from src.orchestration.state import State, Task, OptimizationOutput, ToolCaller


def run(state: State, task: Task, call: ToolCaller) -> OptimizationOutput:
    baseline = state["iq_results"]["baseline"]
    objective_values(baseline)
    if state.get("trial"):
        label = state["label"]
        call("finish_trial", **state["trial"], metrics=state["iq_results"][label],
             artifact_id=state["isp_results"][label]["artifact_id"])
    results = state.get("optimization_results") or call("define_search_space")
    trial = call("run_optimization")["trial"]
    if trial is not None:
        config = {**state["isp_results"]["baseline"]["config"], **trial["params"]}
        return {"trial": trial, "label": f"trial_{trial['number']}", "isp_config": config,
                "optimization_results": results}
    results = {**results, **call("get_pareto_trials"), **call("compare_trials"),
               **call("get_best_trial", focus=state["focus"])}
    recommended = results["recommended"]
    if recommended:
        results["recommended_config"] = state["isp_results"][f"trial_{recommended['number']}"]["config"]
        results["improvement"] = {
            name: (baseline[name]["value"] - value) * (-1 if name == "ssim" else 1)
            for name, value in recommended["objectives"].items()}
    return {"trial": None, "optimization_results": results}

