import logging
import time
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from pydantic import TypeAdapter

from src.agents import experimenter, reviewer, supervisor
from src.agents.models import ModelGateway, decision_trace
from src.execution import data, iq, isp, optimization, report as export, sensor
from src.optimization.search import Search
from src.orchestration.state import State, Task
from src.storage import report
from src.tools import camera_tools, stage_tools
from src.tools.registry import Registry
from src.utils.config import configuration

logger = logging.getLogger(__name__)

AGENTS = {
    "supervisor": supervisor.run, "experimenter": experimenter.run, "reviewer": reviewer.run,
}
STEPS = {"data": data.run, "sensor": sensor.run, "isp": isp.run, "iq": iq.run,
         "optimization": optimization.run, "verification": iq.verify, "report": export.run}
NODES = {**AGENTS, **STEPS}
OUTPUT_SCHEMA = {name: TypeAdapter(run.__annotations__["return"]) for name, run in NODES.items()}


def next_stage(name, state):
    if state["status"] == "failed" or name == "report":
        return END
    if name == "supervisor":
        return "data"
    if name == "data":
        return "experimenter"
    if name == "experimenter":
        return {"sensor": "sensor", "optimization": "iq"}.get(state["task_type"], "isp")
    if name in {"sensor", "isp"}:
        return "iq"
    if name == "iq":
        if state["task_type"] != "optimization":
            return "reviewer"
        return "optimization" if state.get("isp_results") else "isp"
    if name == "optimization":
        return "isp" if state.get("trial") else "reviewer"
    if name == "verification":
        return "reviewer"
    return {"expand": "experimenter", "verify": "verification", "report": "report"}[
        state["review"]["action"]]


class Run:
    def __init__(self, store, task: Task, mock=False, on_update=None, architecture="multi_agent"):
        self.store, self.task, self.on_update = store, task, on_update
        self.architecture = architecture
        self.planner = supervisor.Planner(mock)
        self.role_gateway = self.planner.gateway if architecture == "multi_agent" else ModelGateway(mock=True)
        self.registry = Registry(store)
        camera_tools.register(self.registry)
        stage_tools.register(self.registry)
        report.register(self.registry)
        self.search = Search(store, task)
        self.search.register(self.registry)
        self.retries = configuration("agents")["retries"]

    def node(self, name):
        def execute(state):
            exp = state["experiment_id"]
            if self.on_update:
                self.on_update({**state, "current_stage": name})
            elapsed_errors = list(state["errors"])
            traces = list(state["agent_trace"])
            actor = name if self.architecture == "multi_agent" else "single"
            def call(tool, **params):
                return self.registry.call(exp, actor, tool, **params)
            for attempt in range(self.retries+1):
                start = time.perf_counter()
                event = {"agent": actor, "stage": name, "attempt": attempt+1, "status": "succeeded"}
                gateway = self.planner.gateway if name == "supervisor" else self.role_gateway
                event["kind"] = ("llm_agent" if gateway.mode == "openai" else "mock_policy") \
                    if name in AGENTS else "execution"
                if event["kind"] == "llm_agent":
                    event["model"] = gateway.models[name]
                try:
                    args = (state, self.task, call)
                    if name == "supervisor":
                        args += (self.planner,)
                    elif name in AGENTS:
                        args += (gateway,)
                    output = OUTPUT_SCHEMA[name].validate_python(NODES[name](*args))
                except Exception as exc:
                    validation_failure = isinstance(exc, ValueError)
                    event.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                    elapsed_errors.append({"agent": name, "error": event["error"]})
                    logger.info("Agent %s failed in %s: %s", name, exp, exc)
                event["seconds"] = time.perf_counter()-start
                traces.append(event)
                self.store.record("agent_traces", event, exp)
                if event["status"] == "succeeded":
                    return {**output, "current_stage": name, "errors": elapsed_errors,
                            "agent_trace": traces, "status": "succeeded" if name == "report" else "running"}
                # Validation failures and stateful trial/report stages are deterministic or side-effectful.
                if name not in {*AGENTS, "data"} or name == "data" and validation_failure:
                    break
            if self.search.active:
                call("stop_optimization")
            return {"status": "failed", "current_stage": name,
                    "errors": elapsed_errors, "agent_trace": traces}
        return execute

    def graph(self):
        graph = StateGraph(State)
        for name in NODES:
            graph.add_node(name, self.node(name))
            graph.add_conditional_edges(name, lambda state, node=name: next_stage(node, state),
                                        [*NODES, END])
        graph.add_edge(START, "supervisor")
        return graph.compile()

    def execute(self):
        task_id, exp = uuid4().hex, uuid4().hex
        task_data = self.task.model_dump(mode="json")
        state = {"task_id": task_id, "experiment_id": exp, "user_request": self.task.user_request,
                 "input_artifacts": task_data["input_artifacts"], "isp_config": task_data["isp_config"],
                 "errors": [], "agent_trace": [], "status": "running"}
        metadata = {"task_id": task_id, "configuration": task_data,
                    "architecture": self.architecture, "status": "running"}
        self.store.record("tasks", task_data, exp, task_id)
        self.store.record("experiments", metadata, exp, exp)
        started = time.perf_counter()
        try:
            if self.architecture == "multi_agent":
                state = self.graph().invoke(state, {"recursion_limit": 30+3*self.task.trials})
            elif self.architecture == "single_tools":
                stage = "supervisor"
                while stage != END:
                    state.update(self.node(stage)(state))
                    stage = next_stage(stage, state)
            elif self.architecture == "single":
                plan, tokens = self.planner.plan(self.task.user_request, self.task.task_type, use_tools=False)
                state.update(task_type=plan.task_type, execution_plan=supervisor.PLANS[plan.task_type],
                             token_usage=tokens, llm_mode=self.planner.mode, status="planned_only",
                             model_decisions=[decision_trace(self.planner.gateway, "supervisor", plan, tokens)])
                event = {"agent": "single", "stage": "planning", "status": "succeeded",
                         "seconds": time.perf_counter()-started, "attempt": 1}
                state["agent_trace"] = [event]
                self.store.record("agent_traces", event, exp)
            else:
                raise ValueError("Unknown architecture")
        except Exception as exc:
            logger.exception("Experiment %s failed", exp)
            if self.search.active:
                self.registry.call(exp, "optimization", "stop_optimization")
            state.update(status="failed", errors=[*state["errors"], {"error": str(exc)}])
        metadata.update(status=state["status"], state=state, seconds=time.perf_counter()-started)
        self.store.record("experiments", metadata, exp, exp)
        if state.get("reports"):
            self.store.record("reports", state["reports"], exp)
        if self.on_update:
            self.on_update(state)
        return state

