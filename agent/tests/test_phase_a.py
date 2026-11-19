"""Phase A tests: answer-aware data_through, async guard path, REPL errors."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import memory as mem
from app_agent import Agent, make_client  # noqa: E402  (make_client for A1)
from fake_model import AsyncOnlyFakeChatModel, FakeChatModel, tool_call  # noqa: E402
from langchain.agents import create_agent  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

# Two fake "server" payloads a multi-tool turn would produce.
OHLCV_PAYLOAD = {"current_price": 1570.4, "data_through": "2025-12-31"}
RETURNS_PAYLOAD = {"return_20d": 2.2483, "data_through": "2025-12-19"}


@tool
def get_ohlcv(symbol: str, as_of: str | None = None) -> str:
    """Fake server OHLCV tool."""
    return json.dumps(OHLCV_PAYLOAD)


@tool
def get_returns(symbol: str, as_of: str | None = None) -> str:
    """Fake server returns tool."""
    return json.dumps(RETURNS_PAYLOAD)


def _msgs() -> list[ToolMessage]:
    return [
        ToolMessage(content=json.dumps(OHLCV_PAYLOAD), name="get_ohlcv",
                    tool_call_id="a"),
        ToolMessage(content=json.dumps(RETURNS_PAYLOAD), name="get_returns",
                    tool_call_id="b"),
    ]


# ---------------- A1: auth fallback ----------------

def test_make_client_rejects_anthropic_only(monkeypatch):
    """ANTHROPIC_API_KEY alone must NOT build a ChatOpenAI client (it used to
    surface as an opaque auth error against an OpenAI-shaped endpoint)."""
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    with pytest.raises(RuntimeError, match="No LLM API key configured"):
        make_client()


def test_make_client_accepts_openai_alias(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    client = make_client()
    assert client.openai_api_key.get_secret_value() == "sk-fake"  # noqa: S105


# ---------------- A2: answer-aware data_through ----------------

def test_data_through_prefers_payload_the_answer_quotes(tmp_path):
    agent = Agent(session=None, client=None, db_path=str(tmp_path / "x.db"))
    msgs = _msgs()
    assert agent._data_through(msgs, "the 20-day return is 2.2483%") == "2025-12-19"
    assert agent._data_through(msgs, "RELIANCE closed at Rs 1,570.40") == "2025-12-31"
    # nothing quoted -> fall back to the first payload that has data_through
    assert agent._data_through(msgs, "it went up, honestly 999.99") == "2025-12-31"
    # no payloads -> None
    assert agent._data_through([], "any answer 1.5") is None


def test_multi_tool_turn_reports_quoted_payloads_data_through(tmp_path):
    """Through the full graph: an answer quoting the get_returns payload must
    store get_returns' data_through, not get_ohlcv's."""
    db = str(tmp_path / "dt.db")
    scripted = [
        AIMessage(content="", tool_calls=[
            {"name": "get_ohlcv", "args": {"symbol": "TCS.NS"}, "id": "t1",
             "type": "tool_call"},
            {"name": "get_returns", "args": {"symbol": "TCS.NS"}, "id": "t2",
             "type": "tool_call"},
        ]),
        AIMessage(content="TCS returned 2.2483% over 20 sessions."),
    ]
    agent = Agent(session=None, client=FakeChatModel(scripted=scripted), db_path=db,
                  tools=[get_ohlcv, get_returns])
    answer = asyncio.run(agent.ask("TCS 20-day return as of 2025-12-31?"))
    assert "2.2483%" in answer
    row = mem.recent_queries(1, db)[0]
    assert row["data_through"] == "2025-12-19"


# ---------------- A3: async guard path ----------------

def test_guard_repair_uses_ainvoke_not_sync_invoke(tmp_path):
    """The guard's repair call must go through ainvoke: the guard's model here
    exposes ONLY the async path, so a sync .invoke() inside the guard would
    raise NotImplementedError and fail the graph. (The agent's own model is a
    normal fake — langgraph's model node legitimately drives it synchronously.)"""
    from middleware import GroundingGuard

    db = str(tmp_path / "async_guard.db")
    agent_model = FakeChatModel(scripted=[
        tool_call("get_ohlcv", {"symbol": "RELIANCE.NS"}, "t1"),
        AIMessage(content="RELIANCE closed at 1,999.99 (data_through 2025-12-31)."),
    ])
    guard_model = AsyncOnlyFakeChatModel(scripted=[
        AIMessage(content="RELIANCE closed at Rs 1,570.40 (data_through 2025-12-31)."),
    ])
    guard = GroundingGuard(model=guard_model, db_path=db)
    graph = create_agent(
        model=agent_model,
        tools=[get_ohlcv],
        system_prompt="P",
        middleware=[guard],
    )
    result = asyncio.run(graph.ainvoke(
        {"messages": [HumanMessage(content="q")]},
        config={"recursion_limit": 40},
    ))
    assert result["messages"][-1].content.startswith("RELIANCE closed at Rs 1,570.40")
    assert guard.last_result["grounded"] is True and guard.last_result["repaired"]


def test_sync_and_async_guard_paths_share_logic(tmp_path):
    """Same state, same verdict: after_model and aafter_model agree."""
    from middleware import GroundingGuard
    payload = json.dumps({"current_price": 1570.4, "data_through": "2025-12-31"})
    turn = [
        HumanMessage(content="q"),
        AIMessage(content="", tool_calls=[
            {"name": "get_ohlcv", "args": {}, "id": "t1", "type": "tool_call"}]),
        ToolMessage(content=payload, name="get_ohlcv", tool_call_id="t1"),
        AIMessage(content="price is 1,999.99"),
    ]
    repair = "price is Rs 1,570.40"
    sync_guard = GroundingGuard(
        model=FakeChatModel(scripted=[AIMessage(content=repair)]))
    update = sync_guard.after_model({"messages": list(turn)}, runtime=None)
    sync_result = sync_guard.last_result

    async_guard = GroundingGuard(
        model=FakeChatModel(scripted=[AIMessage(content=repair)]))
    update2 = asyncio.run(
        async_guard.aafter_model({"messages": list(turn)}, runtime=None))
    async_result = async_guard.last_result

    assert sync_result == async_result == {"grounded": True, "claims": [
        {"token": "Rs 1,570.40", "value": 1570.4}], "repaired": True}
    assert [m.content for m in update["messages"]] == \
        [m.content for m in update2["messages"]]


# ---------------- A4: REPL error handling ----------------

def test_handle_error_one_line_messages():
    from chat import _handle_error
    from langgraph.errors import GraphRecursionError
    msg = _handle_error(GraphRecursionError("Recursion limit of 30 reached"))
    assert msg == ("hit the step limit: the agent stopped early; "
                   "try a narrower question.")
    assert _handle_error(RuntimeError("HTTP 429 Too Many Requests")) == \
        "provider error: HTTP 429 Too Many Requests"


def test_recursion_error_actually_raised_from_fake_model(tmp_path):
    """A scripted model that keeps calling tools loops until the graph hits
    the step limit — the exact failure the REPL one-liner covers."""
    from langgraph.errors import GraphRecursionError
    db = str(tmp_path / "loop.db")
    loop = AIMessage(content="", tool_calls=[
        {"name": "list_watchlist", "args": {}, "id": "x", "type": "tool_call"}])
    agent = Agent(session=None, client=FakeChatModel(scripted=[loop]), db_path=db)
    with pytest.raises(GraphRecursionError):
        asyncio.run(agent.ask("loop forever"))
