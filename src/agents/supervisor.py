import re

from src.agents.models import ModelGateway, decision_trace
from src.orchestration.state import Plan, State, Task, SupervisorOutput, ToolCaller

PLANS = {
    "sensor": ["data", "experimenter", "sensor", "iq", "reviewer", "report"],
    "comparison": ["data", "experimenter", "isp", "iq", "reviewer", "report"],
    "optimization": ["data", "experimenter", "iq", "isp", "iq", "optimization",
                     "(isp → iq → optimization) × budget", "reviewer", "report"],
    "hdr": ["data", "experimenter", "isp", "iq", "reviewer", "report"],
    "burst": ["data", "experimenter", "isp", "iq", "reviewer", "report"],
}


class Planner:
    def __init__(self, mock=False, gateway=None):
        self.gateway = gateway or ModelGateway(mock)
        self.mode = self.gateway.mode

    def plan(self, request, task_type, *, use_tools=True):
        if self.mode == "mock":
            routes = [
                ("optimization", r"优化|optimi[sz]|pareto|调参"),
                ("hdr", r"hdr|高动态融合"),
                ("burst", r"burst|多帧降噪"),
                ("sensor", r"sensor|暗场|平场|坏点|dsnu|prnu|ptc|hot pixel|dead pixel"),
            ]
            route = next((kind for kind, regex in routes if re.search(regex, request, re.I)),
                         "comparison")
            focus = "noise" if re.search(r"low.light|低照", request, re.I) else "balanced"
            return Plan(task_type=task_type or route, focus=focus), None
        plan, tokens = self.gateway.decide(
            "supervisor", "select_workflow", Plan,
            "Classify a camera lab request. Routes: sensor, comparison, optimization, hdr, burst. "
            "Focus: balanced, noise, detail, color. Honor any specified route. "
            "Return task_type and focus. Do not estimate measurements or execute code.",
            {"route": task_type, "request": request}, use_tools=use_tools)
        if task_type:
            plan.task_type = task_type
        return plan, tokens


def run(state: State, task: Task, call: ToolCaller, planner: Planner) -> SupervisorOutput:
    plan, tokens = planner.plan(task.user_request, "comparison" if task.isp_stage else task.task_type)
    return {"task_type": plan.task_type, "focus": plan.focus,
            "execution_plan": ["supervisor", *PLANS[plan.task_type]],
            "token_usage": tokens, "llm_mode": planner.mode,
            "model_decisions": [decision_trace(planner.gateway, "supervisor", plan, tokens)]}

