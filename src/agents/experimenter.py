from src.agents.models import record_decision
from src.orchestration.state import ExperimentPlan, ExperimentPlanOutput, State, Task, ToolCaller


def available_experiments(state, task):
    kind = state["task_type"]
    inputs = task.input_artifacts
    if kind == "sensor":
        results = state.get("sensor_results", {})
        checks = [check for check, present in (("flat", inputs.flat), ("ptc", inputs.ptc))
                  if present and not results.get(f"{check}_inputs")]
        return [], checks
    if task.isp_stage:
        return [], []
    if kind == "comparison":
        return [name for name in dict.fromkeys(task.algorithms)
                if name not in state.get("isp_results", {})], []
    return list(dict.fromkeys([task.isp_config.demosaic, *task.algorithms])), []


def run(state: State, task: Task, call: ToolCaller, gateway) -> ExperimentPlanOutput:
    algorithms, checks = available_experiments(state, task)
    kind = state["task_type"]
    maximum = len(algorithms) if kind == "comparison" else min(1, len(algorithms))
    if gateway.mode == "openai":
        plan, tokens = gateway.decide(
            "experimenter", "select_experiments", ExperimentPlan,
            "Choose experiments within the allowed lists and the requested workflow. "
            "For an ISP comparison select at least one candidate. Honor explicit requests for "
            "all variants or specific measurements. A non-comparison ISP selection defines the "
            "algorithm for the whole run. For Sensor runs the dark analysis is automatic. "
            "If the reviewer requests expansion, select at least one unmeasured option. "
            "Use a short rationale in the user's language. Never invent measurements.",
            {"request": task.user_request, "workflow": kind, "focus": state["focus"],
             "allowed_algorithms": algorithms, "maximum_algorithms": maximum,
             "default_algorithm": task.isp_config.demosaic, "available_sensor_checks": checks,
             "dark_frame_count": len(task.input_artifacts.dark),
             "ptc_level_count": len(task.input_artifacts.ptc),
             "review_feedback": state.get("review")})
    else:
        selected = algorithms if kind == "comparison" else algorithms[:maximum]
        plan = ExperimentPlan(algorithms=selected, sensor_checks=checks,
                              rationale="执行已配置且可用的实验。")
        tokens = None
    if (len(plan.algorithms) > maximum or not set(plan.algorithms) <= set(algorithms)
            or not set(plan.sensor_checks) <= set(checks)):
        raise ValueError("Experiment selection exceeds available inputs or configured algorithms")
    if kind == "comparison" and algorithms and not plan.algorithms:
        raise ValueError("An ISP comparison requires an allowed algorithm")
    if state.get("review", {}).get("action") == "expand" and not (plan.algorithms or plan.sensor_checks):
        raise ValueError("Reviewer requested an additional experiment")
    return {"experiment_plan": plan.model_dump(mode="json"),
            **record_decision(state, gateway, "experimenter", plan, tokens)}
