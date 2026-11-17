import json
import os

from dotenv import load_dotenv

from src.utils.config import ROOT, configuration


ROLE_MODELS = {"supervisor": "gpt-6.1-terra", "experimenter": "gpt-6-luna",
               "reviewer": "gpt-6.1-sol"}


class ModelGateway:
    def __init__(self, mock=False):
        load_dotenv(ROOT / ".env")
        mode = "mock" if mock else os.getenv("CAMERA_LLM_MODE", "auto").lower()
        key = os.getenv("OPENAI_API_KEY")
        if mode not in {"mock", "auto", "openai"} or mode == "openai" and not key:
            raise ValueError("Configure CAMERA_LLM_MODE and OPENAI_API_KEY")
        self.mode = "openai" if key and mode != "mock" else "mock"
        shared = os.getenv("CAMERA_LLM_MODEL")
        self.models = {role: os.getenv(f"CAMERA_{role.upper()}_MODEL") or shared or default
                       for role, default in ROLE_MODELS.items()}

    def decide(self, role, name, schema, instructions, context, *, use_tools=True):
        from openai import OpenAI

        messages = [{"role": "system", "content": instructions},
                    {"role": "user", "content": json.dumps(context, ensure_ascii=False)}]
        model = self.models[role]
        deepseek = model.startswith("deepseek-")
        # GPT-6 and DeepSeek require thinking disabled for named Chat Completions tool calls.
        options = {"reasoning_effort": "none"} if deepseek or model.startswith(("gpt-6-", "gpt-6.")) else {}
        base_url = os.getenv("OPENAI_BASE_URL", "").strip() or "https://api.openai.com/v1"
        with OpenAI(base_url=base_url,
                    timeout=configuration("agents")["llm_timeout_seconds"], max_retries=0) as client:
            if use_tools:
                if not deepseek:
                    options["parallel_tool_calls"] = False
                reply = client.chat.completions.create(
                    model=model, messages=messages, **options,
                    tools=[{"type": "function", "function": {
                        "name": name, "description": schema.__doc__ or name,
                        "parameters": schema.model_json_schema()}}],
                    tool_choice={"type": "function", "function": {"name": name}})
                calls = reply.choices[0].message.tool_calls
                if not calls or len(calls) != 1 or calls[0].function.name != name:
                    raise ValueError(f"Model did not return {name}")
                decision = schema.model_validate_json(calls[0].function.arguments)
            else:
                messages.append({"role": "system", "content": "Return a JSON object."})
                reply = client.chat.completions.create(
                    model=model, messages=messages, response_format={"type": "json_object"}, **options)
                decision = schema.model_validate_json(reply.choices[0].message.content or "")
        return decision, reply.usage.total_tokens if reply.usage else None


def decision_trace(gateway, role, decision, tokens):
    return {"role": role, "mode": gateway.mode,
            "model": gateway.models[role] if gateway.mode == "openai" else None,
            "decision": decision.model_dump(mode="json"), "tokens": tokens}


def record_decision(state, gateway, role, decision, tokens):
    total = state.get("token_usage")
    return {
        "model_decisions": [*state.get("model_decisions", []),
                            decision_trace(gateway, role, decision, tokens)],
        "token_usage": total if tokens is None else (total or 0) + tokens,
    }
