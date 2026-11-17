import json

import httpx
import openai
import pytest

from src.camera.models import RawSpec
from src.camera.synthetic_data import generate
from src.orchestration.graph import AGENTS, Run
from src.orchestration.state import Task
from src.storage.store import Store
from src.utils.config import configuration


@pytest.mark.parametrize("action", ["expand", "verify"])
@pytest.mark.parametrize("provider", ["openai", "deepseek"])
def test_model_decisions_change_experiment_via_real_sdk(tmp_path, monkeypatch, action, provider):
    store = Store(tmp_path)
    inputs = generate(store, RawSpec(**configuration("sensor")), seed=123)
    models = {"supervisor": "test-plan", "experimenter": "gpt-6-luna", "reviewer": "test-review"}
    is_deepseek = provider == "deepseek"
    if is_deepseek:
        models = {"supervisor": "deepseek-flash", "experimenter": "deepseek-flash",
                  "reviewer": "deepseek-v4-pro"}
    for role, model in models.items():
        monkeypatch.setenv(f"CAMERA_{role.upper()}_MODEL", model)
    monkeypatch.setenv("CAMERA_LLM_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepseek.com" if is_deepseek else "")
    calls = []

    def respond(request):
        body = json.loads(request.content)
        assert body.get("reasoning_effort") == (
            "none" if is_deepseek or body["model"] == "gpt-6-luna" else None)
        if is_deepseek:
            assert "parallel_tool_calls" not in body
        context = json.loads(body["messages"][1]["content"])
        operation = body["tool_choice"]["function"]["name"]
        calls.append((body["model"], operation, context))
        if operation == "select_workflow":
            decision = {"task_type": "comparison", "focus": "detail"}
        elif operation == "select_experiments":
            decision = {"algorithms": [context["allowed_algorithms"][0]], "rationale": "Measure a candidate"}
        elif not context["previous_review"]:
            assert set(context["measured_iq"]) == {"bilinear"}
            assert context["measured_iq"]["bilinear"]["psnr"]["valid"]
            assert action in context["available_actions"]
            decision = {"action": action, "rationale": "Check the current evidence"}
            if action == "verify":
                decision["target"] = "bilinear"
        else:
            assert context["available_actions"] == ["report"]
            if action == "expand":
                assert set(context["measured_iq"]) == {"bilinear", "edge_adaptive"}
            else:
                assert context["verification"]["matches"]
            decision = {"action": "report", "rationale": "Evidence is ready"}
        return httpx.Response(200, json={
            "id": "chatcmpl-test", "object": "chat.completion", "created": 1, "model": body["model"],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "tool_calls": [{"id": "call-test", "type": "function",
                    "function": {"name": operation, "arguments": json.dumps(decision)}}],
            }}],
        })

    client_type = openai.OpenAI

    def local_client(**kwargs):
        assert kwargs["base_url"] == (
            "https://api.deepseek.com" if is_deepseek else "https://api.openai.com/v1")
        kwargs.update(base_url="https://example.invalid/v1",
                      http_client=httpx.Client(transport=httpx.MockTransport(respond)))
        return client_type(**kwargs)

    monkeypatch.setattr(openai, "OpenAI", local_client)
    task = Task(user_request="比较去马赛克算法", input_artifacts=inputs,
                algorithms=["bilinear", "edge_adaptive"])
    result = Run(store, task).execute()
    assert result["status"] == "succeeded", result["errors"]
    assert set(AGENTS) == {"supervisor", "experimenter", "reviewer"}
    roles = ["supervisor", "experimenter", "reviewer"]
    roles += ["experimenter", "reviewer"] if action == "expand" else ["reviewer"]
    assert [model for model, _, _ in calls] == [models[role] for role in roles]
    assert [item["role"] for item in result["model_decisions"]] == roles
    assert result["token_usage"] == 10 * len(roles)
    assert [item["kind"] for item in result["agent_trace"]].count("llm_agent") == len(roles)
    if action == "expand":
        assert calls[3][2]["review_feedback"]["action"] == "expand"
        assert set(result["isp_results"]) == {"bilinear", "edge_adaptive"}
    else:
        assert set(result["isp_results"]) == {"bilinear"}
        assert result["verification"]["matches"]
        metric = result["verification"]["metrics"]["psnr"]
        assert store.get("iq_metrics", metric["record_id"])["payload"]["label"] == "verification:bilinear"
    report = json.loads(store.path(result["reports"]["json"]).read_text(encoding="utf-8"))
    assert report["summary"]["model_decisions"] == result["model_decisions"]
