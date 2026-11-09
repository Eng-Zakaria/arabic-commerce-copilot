"""Claude via the official Anthropic SDK.

- Model: claude-opus-5-5 by default (override with CLAUDE_MODEL).
- Thinking: adaptive by default on this model (not configured; it cannot be disabled).
  Effort is set explicitly (CLAUDE_EFFORT, default "medium").
- Refusals: server-side fallbacks ("default" mode, beta server-side-fallback-2026-07-01),
  so a declined request is retried on Anthropic's recommended fallback model in the same call.
- Tool choice stays "auto" (forced tool use is rejected on this model); tools are strict.
- The assistant's full content (thinking + tool_use blocks) is stored verbatim in history["raw"]
  and replayed unchanged, as the API requires during a tool loop.
"""

from __future__ import annotations

import os

from .base import ProviderError, ToolCall, Turn

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class ClaudeProvider:
    name = "claude"

    def __init__(self, model: str | None = None, effort: str | None = None, client=None):
        self.model = model or os.environ.get("CLAUDE_MODEL", "claude-opus-5-5")
        self.effort = effort or os.environ.get("CLAUDE_EFFORT", "medium")
        self._client = client

    @property
    def client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    @staticmethod
    def _tools(tools: list[dict]) -> list[dict]:
        return [{**t, "strict": True} for t in tools]

    def _messages(self, history: list[dict]) -> list[dict]:
        msgs: list[dict] = []
        pending_results: list[dict] = []

        def flush():
            if pending_results:
                msgs.append({"role": "user", "content": list(pending_results)})
                pending_results.clear()

        for m in history:
            if m["role"] == "tool":
                pending_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": m["tool_call_id"],
                        "content": m["content"],
                    }
                )
                continue
            flush()  # all results of one assistant turn go back in a single user message
            if m["role"] == "assistant":
                raw = (m.get("raw") or {}).get(self.name)
                if raw is not None:
                    msgs.append({"role": "assistant", "content": raw})
                else:  # turn produced by another provider: rebuild plain blocks
                    blocks: list[dict] = []
                    if m.get("content"):
                        blocks.append({"type": "text", "text": m["content"]})
                    blocks += [
                        {
                            "type": "tool_use",
                            "id": c["id"],
                            "name": c["name"],
                            "input": c["arguments"],
                        }
                        for c in m.get("tool_calls") or []
                    ]
                    msgs.append(
                        {
                            "role": "assistant",
                            "content": blocks or [{"type": "text", "text": "..."}],
                        }
                    )
            else:
                msgs.append({"role": "user", "content": m["content"]})
        flush()
        return msgs

    def complete(self, system: str, history: list[dict], tools: list[dict]) -> Turn:
        import anthropic

        try:
            resp = self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=self._messages(history),
                tools=self._tools(tools),
                output_config={"effort": self.effort},
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
        except (
            anthropic.APIConnectionError,
            anthropic.RateLimitError,
            anthropic.InternalServerError,
        ) as exc:
            raise ProviderError(f"{self.name}: {exc}") from exc
        except anthropic.AuthenticationError as exc:
            raise ProviderError(f"{self.name}: not authenticated ({exc})") from exc

        if resp.stop_reason == "refusal":  # the whole fallback chain declined
            return Turn(text="", stop="refusal", provider=self.name, model=resp.model)

        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [
            ToolCall(b.id, b.name, dict(b.input)) for b in resp.content if b.type == "tool_use"
        ]
        return Turn(
            text=text,
            tool_calls=calls,
            stop={"tool_use": "tool_use", "max_tokens": "max_tokens"}.get(resp.stop_reason, "end"),
            provider=self.name,
            model=resp.model,  # reports the fallback model if one served the turn
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            raw=[b.model_dump(exclude_none=True) for b in resp.content],
        )
