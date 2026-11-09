"""Provider-neutral conversation format.

History entries (plain dicts, JSON-serialisable):
  {"role": "user", "content": str}
  {"role": "assistant", "content": str, "tool_calls": [ToolCall...], "raw": {provider: payload}}
  {"role": "tool", "tool_call_id": str, "name": str, "content": str}

"raw" keeps the provider's own assistant payload so a provider can replay its turns exactly
(Claude requires its thinking blocks to be passed back unchanged during a tool loop).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict

    def as_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


@dataclass
class Turn:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop: str = "end"  # end | tool_use | refusal | max_tokens | error
    provider: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    raw: object = None  # provider payload to store in history["raw"][provider]


class Provider(Protocol):
    name: str
    model: str

    def complete(self, system: str, history: list[dict], tools: list[dict]) -> Turn: ...


class ProviderError(RuntimeError):
    """Raised for transport/availability failures so the router can fall through."""
