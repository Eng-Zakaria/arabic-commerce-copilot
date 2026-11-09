"""HTTP API + a minimal RTL chat page.

uv run uvicorn copilot.api:app --port 8080
open http://localhost:8080
"""

from __future__ import annotations

import uuid
from pathlib import Path

from cloudkit import telemetry
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .agent.loop import Agent
from .agent.tools import Toolbox

telemetry.configure("kanz-copilot")
app = FastAPI(title="Kanz Copilot", version="0.1.0")
_agent: Agent | None = None
_sessions: dict[str, list[dict]] = {}  # in-memory; a real deployment keeps this in Redis/Postgres
MAX_SESSIONS = 1000


def agent() -> Agent:
    global _agent
    if _agent is None:
        _agent = Agent(Toolbox())
    return _agent


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None


class ChatOut(BaseModel):
    session_id: str
    reply: str
    provider: str
    model: str
    tool_calls: list[dict]
    latency_ms: int
    guard: dict


@app.post("/chat", response_model=ChatOut)
def chat(body: ChatIn) -> ChatOut:
    sid = body.session_id or uuid.uuid4().hex
    if sid not in _sessions and len(_sessions) >= MAX_SESSIONS:
        _sessions.pop(next(iter(_sessions)))
    history = _sessions.setdefault(sid, [])
    r = agent().respond(history, body.message)
    return ChatOut(
        session_id=sid,
        reply=r.text,
        provider=r.provider,
        model=r.model,
        tool_calls=r.tool_calls,
        latency_ms=r.latency_ms,
        guard={
            "clean": r.guard.clean,
            "injection_in_input": r.guard.injection_in_input,
            "ungrounded_prices": r.guard.ungrounded,
        },
    )


@app.get("/search")
def search(q: str, limit: int = 5) -> dict:
    if not q.strip():
        raise HTTPException(422, "empty query")
    return agent().toolbox.search_offers(q, limit=limit)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "offers": len(agent().toolbox.offers)}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(Path(__file__).parent / "web" / "index.html")
