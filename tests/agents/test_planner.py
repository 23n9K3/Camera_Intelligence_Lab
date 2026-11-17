import pytest

from src.agents.supervisor import Planner
from src.orchestration.state import Task
from src.tools.registry import Registry


def test_mock_keeps_explicit_route(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("CAMERA_LLM_MODE", "auto")
    plan, tokens = Planner().plan("分析暗场", "comparison")
    assert plan.task_type == "comparison"
    assert tokens is None


def test_tool_argument_validation_is_recorded(tmp_path):
    from src.storage.store import Store
    store = Store(tmp_path)
    registry = Registry(store)

    def test_tool(store, experiment, value: int):
        return {"value": value}

    registry.register("test", test_tool)
    with pytest.raises(ValueError):
        registry.call("exp", "tester", "test_tool", value="not an integer")
    assert store.rows("tool_calls", "exp")[0]["payload"]["status"] == "failed"
    assert registry.call("exp", "tester", "test_tool", value=5) == {"value": 5}


def test_task_budget_is_validated():
    with pytest.raises(ValueError):
        Task(user_request="优化", input_artifacts={}, trials=101)

