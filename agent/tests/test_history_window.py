"""The conversation window handed back to the provider must be wire-valid.

A raw ``messages[-N:]`` slice can start on a ToolMessage whose AIMessage
(tool_calls) fell outside the window, or end an AIMessage(tool_calls) whose
ToolMessages were cut. OpenAI-compatible providers reject both with a 400, so
the failure only shows up in a long live session — exactly what the fake-model
tests would otherwise miss.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app_agent import Agent, _trim_to_turns  # noqa: E402
from fake_model import FakeChatModel, tool_call  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402


def _assert_wire_valid(messages):
    """Every ToolMessage answers a tool_call visible earlier in the window."""
    assert isinstance(messages[0], HumanMessage), (
        f"window must start a turn, got {type(messages[0]).__name__}: "
        f"{[type(m).__name__ for m in messages]}"
    )
    open_calls: set[str] = set()
    for msg in messages:
        if isinstance(msg, AIMessage):
            open_calls |= {call["id"] for call in (msg.tool_calls or [])}
        elif isinstance(msg, ToolMessage):
            assert msg.tool_call_id in open_calls, (
                f"orphan ToolMessage {msg.tool_call_id} — provider would 400"
            )


def test_long_session_never_sends_an_orphan_tool_message(tmp_path):
    """Eight turns with a one-tool call each, window capped at two turns."""
    scripted = [
        tool_call("list_watchlist", {}, "w1"),
        AIMessage(content="answer 1"),
        tool_call("list_watchlist", {}, "w2"),
        AIMessage(content="answer 2"),
    ]
    agent = Agent(
        session=None,
        client=FakeChatModel(scripted=scripted),
        db_path=str(tmp_path / "window.db"),
        history_limit=2,
    )
    for turn in range(8):
        asyncio.run(agent.ask(f"question {turn}"))
        _assert_wire_valid(agent.messages)
        humans = [m for m in agent.messages if isinstance(m, HumanMessage)]
        assert len(humans) <= 2, f"window grew past history_limit: {len(humans)}"
    assert isinstance(agent.messages[-1], AIMessage)


def test_trim_to_turns_keeps_pairs_together():
    msgs = [
        HumanMessage(content="q1", id="h1"),
        AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "c1",
                                          "type": "tool_call"}], id="a1"),
        ToolMessage(content="{}", tool_call_id="c1", name="t", id="t1"),
        AIMessage(content="a1", id="a2"),
        HumanMessage(content="q2", id="h2"),
        AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "c2",
                                          "type": "tool_call"}], id="a3"),
        ToolMessage(content="{}", tool_call_id="c2", name="t", id="t2"),
        AIMessage(content="a2", id="a4"),
    ]
    # One turn: starts at q2, keeps both halves of its tool call.
    one = _trim_to_turns(msgs, 1)
    assert [type(m).__name__ for m in one] == [
        "HumanMessage", "AIMessage", "ToolMessage", "AIMessage"]
    assert one[0].content == "q2"
    # Two turns: the full list, unchanged.
    assert _trim_to_turns(msgs, 2) == msgs
    # Zero/tiny limits degrade safely rather than leaking an orphan.
    assert _trim_to_turns(msgs, 0) == []
