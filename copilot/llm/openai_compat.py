"""OpenAI-compatible chat completions: Ollama locally, Azure OpenAI in azure mode (via cloudkit)."""

from __future__ import annotations

import json
import os

from .base import ProviderError, ToolCall, Turn


class OpenAICompatProvider:
    name = "openai_compat"

    def __init__(
        self,
        model: str | None = None,
        client=None,
        temperature: float = 0.2,
        timeout_s: float | None = None,
    ):
        self.model = model or os.environ.get("LOCAL_CHAT_MODEL", "qwen3:4b-instruct")
        self.temperature = temperature
        # The SDK default is 10 min x 3 attempts. A cheap-first router needs to fall through
        # fast instead: one stuck local generation once cost a 30-minute turn.
        self.timeout_s = timeout_s or float(os.environ.get("LOCAL_TIMEOUT_S", "45"))
        self._client = client

    @property
    def client(self):
        if self._client is None:
            from cloudkit import llm_client

            self._client = llm_client().with_options(timeout=self.timeout_s, max_retries=0)
        return self._client

    @staticmethod
    def _messages(system: str, history: list[dict]) -> list[dict]:
        msgs: list[dict] = [{"role": "system", "content": system}]
        for m in history:
            if m["role"] == "assistant":
                entry: dict = {"role": "assistant", "content": m.get("content") or ""}
                if m.get("tool_calls"):
                    entry["tool_calls"] = [
                        {
                            "id": c["id"],
                            "type": "function",
                            "function": {
                                "name": c["name"],
                                "arguments": json.dumps(c["arguments"], ensure_ascii=False),
                            },
                        }
                        for c in m["tool_calls"]
                    ]
                msgs.append(entry)
            elif m["role"] == "tool":
                msgs.append(
                    {"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]}
                )
            else:
                msgs.append({"role": "user", "content": m["content"]})
        return msgs

    @staticmethod
    def _tools(tools: list[dict]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in tools
        ]

    def complete(self, system: str, history: list[dict], tools: list[dict]) -> Turn:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=self._messages(system, history),
                tools=self._tools(tools),
                temperature=self.temperature,
            )
        except Exception as exc:  # connection refused, 5xx, model missing...
            raise ProviderError(f"{self.name}: {exc}") from exc
        choice = resp.choices[0]
        calls = []
        for i, c in enumerate(choice.message.tool_calls or []):
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_unparseable": c.function.arguments}
            calls.append(ToolCall(c.id or f"call_{i}", c.function.name, args))
        usage = resp.usage
        return Turn(
            text=choice.message.content or "",
            tool_calls=calls,
            stop="tool_use"
            if calls
            else ("max_tokens" if choice.finish_reason == "length" else "end"),
            provider=self.name,
            model=self.model,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )
