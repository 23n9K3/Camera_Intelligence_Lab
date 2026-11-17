import inspect
import logging
import time
from collections.abc import Callable

from pydantic import JsonValue, create_model

from src.camera.models import Model

logger = logging.getLogger(__name__)


class ToolOutput(Model):
    data: dict[str, JsonValue]


class Registry:
    def __init__(self, store):
        self.store = store
        self.tools = {}

    def register(self, group: str, handler: Callable, name=None):
        name = name or handler.__name__
        fields = {}
        for parameter in list(inspect.signature(handler).parameters.values())[2:]:
            default = ... if parameter.default is inspect.Parameter.empty else parameter.default
            if parameter.annotation is inspect.Parameter.empty:
                raise TypeError(f"{name}.{parameter.name} needs an annotation")
            fields[parameter.name] = (parameter.annotation, default)
        schema = create_model(f"{name}_input", __base__=Model, **fields)
        if name in self.tools:
            raise ValueError(f"Duplicate tool {name}")
        self.tools[name] = group, schema, handler

    def call(self, experiment, agent, name, **params):
        start = time.perf_counter()
        record = {"tool": name, "agent": agent, "params": params, "status": "failed"}
        try:
            _, schema, handler = self.tools[name]
            parsed = schema.model_validate(params)
            record["params"] = parsed.model_dump(mode="json")
            args = {field: getattr(parsed, field) for field in type(parsed).model_fields}
            data = ToolOutput(data=handler(self.store, experiment, **args)).data
            record.update(status="succeeded", output=data)
            return data
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            logger.info("Tool %s failed in %s: %s", name, experiment, exc)
            raise
        finally:
            record["seconds"] = time.perf_counter()-start
            self.store.record("tool_calls", record, experiment)

