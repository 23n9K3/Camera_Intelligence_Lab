from src.agents.experimenter import available_experiments
from src.agents.models import record_decision
from src.orchestration.state import ReviewDecision, ReviewOutput, State, Task, ToolCaller


def run(state: State, task: Task, call: ToolCaller, gateway) -> ReviewOutput:
    followup_available = state.get("review_round", 0) == 0
    kind = state["task_type"]
    algorithms, checks = available_experiments(state, task)
    expandable = kind in {"comparison", "sensor"} and bool(algorithms or checks)
    iq = state["iq_results"]
    recommendation = state.get("optimization_results", {}).get("recommended")
    targets = (["baseline", f"trial_{recommendation['number']}"] if recommendation
               else list(iq))
    actions = ["report"]
    if followup_available:
        if expandable:
            actions.append("expand")
        if targets:
            actions.append("verify")
    if gateway.mode == "openai":
        decision, tokens = gateway.decide(
            "reviewer", "review_experiment", ReviewDecision,
            "Review the measured evidence against the user request and the experimental plan. "
            "Choose report when evidence suffices, expand to ask the experimenter to add untested "
            "options, or verify to repeat IQ on one listed target. Use only allowed actions. "
            "For verify set target to its label, otherwise omit target. Never invent a reading "
            "or a cause. Explain the decision briefly in the user's language.",
            {"request": task.user_request, "workflow": kind, "focus": state["focus"],
             "experiment_plan": state["experiment_plan"],
             "available_actions": actions, "verification_targets": targets,
             "remaining_algorithms": algorithms if expandable else [],
             "remaining_sensor_checks": checks if expandable else [],
             "measured_iq": {label: {name: {key: metric[key] for key in ("value", "valid", "unit")}
                                     for name, metric in iq[label].items()} for label in targets},
             "sensor_summary": {name: state.get("sensor_results", {}).get(name)
                                for name in ("dsnu_dn", "prnu_percent", "dark_noise_dn")},
             "previous_review": state.get("review"), "verification": state.get("verification")})
    else:
        decision = ReviewDecision(action="report", rationale="已完成离线验收所需测量。")
        tokens = None
    if decision.action not in actions or (decision.target not in targets if decision.action == "verify"
                                          else decision.target is not None):
        raise ValueError("Reviewer requested an unavailable action or target")
    return {"review": decision.model_dump(mode="json"),
            "review_round": state.get("review_round", 0) + 1,
            **record_decision(state, gateway, "reviewer", decision, tokens)}
