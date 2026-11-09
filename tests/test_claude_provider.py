"""ClaudeProvider request shape, against a fake client (no network, no credentials)."""

from types import SimpleNamespace

from copilot.agent.tools import TOOL_SPECS
from copilot.llm.anthropic_provider import FALLBACK_BETA, ClaudeProvider


class Block(SimpleNamespace):
    def model_dump(self, exclude_none=True):
        return {k: v for k, v in vars(self).items() if v is not None}


class FakeMessages:
    def __init__(self, response):
        self.response, self.calls = response, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def fake_client(content, stop="tool_use"):
    msgs = FakeMessages(
        SimpleNamespace(
            content=content,
            stop_reason=stop,
            model="claude-opus-5-5",
            usage=SimpleNamespace(input_tokens=120, output_tokens=40),
        )
    )
    return SimpleNamespace(beta=SimpleNamespace(messages=msgs)), msgs


def test_request_uses_fallbacks_effort_strict_tools_and_auto_tool_choice():
    client, msgs = fake_client([Block(type="text", text="hi")], stop="end_turn")
    ClaudeProvider(client=client).complete(
        "sys", [{"role": "user", "content": "مرحبا"}], TOOL_SPECS
    )
    kw = msgs.calls[0]
    assert kw["model"] == "claude-opus-5-5"
    assert kw["fallbacks"] == "default" and FALLBACK_BETA in kw["betas"]
    assert kw["output_config"] == {"effort": "medium"}
    assert "thinking" not in kw  # adaptive by default; disabling is rejected on this model
    assert "tool_choice" not in kw  # forced tool use is rejected; auto is the default
    assert all(t["strict"] for t in kw["tools"])


def test_tool_use_turn_is_parsed_and_raw_blocks_kept_for_replay():
    content = [
        Block(type="thinking", thinking="", signature="sig-abc"),
        Block(type="tool_use", id="tu_1", name="search_offers", input={"query": "جيم"}),
    ]
    client, _ = fake_client(content)
    turn = ClaudeProvider(client=client).complete(
        "sys", [{"role": "user", "content": "جيم"}], TOOL_SPECS
    )
    assert turn.stop == "tool_use" and turn.tool_calls[0].name == "search_offers"
    assert turn.raw[0] == {"type": "thinking", "thinking": "", "signature": "sig-abc"}


def test_history_replays_raw_blocks_and_groups_tool_results():
    raw = [
        {"type": "thinking", "thinking": "", "signature": "s"},
        {"type": "tool_use", "id": "a", "name": "search_offers", "input": {}},
        {"type": "tool_use", "id": "b", "name": "search_help", "input": {}},
    ]
    history = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [], "raw": {"claude": raw}},
        {"role": "tool", "tool_call_id": "a", "name": "search_offers", "content": "{}"},
        {"role": "tool", "tool_call_id": "b", "name": "search_help", "content": "{}"},
    ]
    msgs = ClaudeProvider(client=object())._messages(history)
    assert msgs[1] == {"role": "assistant", "content": raw}  # replayed unchanged
    assert msgs[2]["role"] == "user" and len(msgs[2]["content"]) == 2  # one message, both results
    assert {c["tool_use_id"] for c in msgs[2]["content"]} == {"a", "b"}


def test_turns_from_other_providers_are_rebuilt_as_plain_blocks():
    history = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "x", "name": "get_offer", "arguments": {"offer_id": "KZ-0001"}}],
        },
        {"role": "tool", "tool_call_id": "x", "name": "get_offer", "content": "{}"},
    ]
    msgs = ClaudeProvider(client=object())._messages(history)
    assert msgs[1]["content"] == [
        {"type": "tool_use", "id": "x", "name": "get_offer", "input": {"offer_id": "KZ-0001"}}
    ]


def test_refusal_is_reported_not_read_as_content():
    client, _ = fake_client([], stop="refusal")
    turn = ClaudeProvider(client=client).complete(
        "sys", [{"role": "user", "content": "x"}], TOOL_SPECS
    )
    assert turn.stop == "refusal" and turn.text == ""
