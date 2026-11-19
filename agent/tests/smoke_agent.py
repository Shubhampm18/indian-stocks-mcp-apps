"""Smoke test: the LangChain agent graph against a scripted fake model
(offline, no API key needed).

Proves create_agent completes a multi-step memory task (watchlist add ->
preference set -> preference read -> watchlist list -> answer) and logs the
turn to durable memory. Run with:
  uv run python tests/smoke_agent.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_model import FakeChatModel, tool_call  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402


def main() -> int:
    import memory as mem
    from app_agent import Agent

    db = os.path.join(tempfile.mkdtemp(), "smoke.db")
    scripted = [
        tool_call("add_to_watchlist", {"symbol": "TCS.NS"}, "t1"),
        tool_call("set_preference",
                  {"key": "default_metric", "value": "return_20d"}, "t2"),
        tool_call("get_preference", {"key": "default_metric"}, "t3"),
        tool_call("list_watchlist", {}, "t4"),
        AIMessage(content="TCS.NS is on your watchlist; default metric is return_20d."),
    ]
    agent_obj = Agent(session=None, client=FakeChatModel(scripted=scripted), db_path=db)
    answer = asyncio.run(agent_obj.ask(
        "add TCS.NS to my watchlist, save return_20d as my default metric, then show it"))
    assert "TCS.NS" in answer, answer
    assert mem.list_watchlist(db) == ["TCS.NS"], mem.list_watchlist(db)
    # the preference round-tripped THROUGH the agent graph (t2 set, t3 read)
    assert mem.get_preference("default_metric", db_path=db) == "return_20d"
    assert "default metric is return_20d" in answer, answer
    # the tool message the graph produced carried the stored value back
    tool_msgs = [m for m in agent_obj.messages
                 if type(m).__name__ == "ToolMessage" and m.name == "get_preference"]
    payloads = [json.loads(m.content) for m in tool_msgs]
    assert payloads and payloads[0]["value"] == "return_20d", payloads
    history = mem.recent_queries(5, db)
    assert len(history) == 1, history
    assert "set_preference" in history[0]["tools_called"]
    assert "get_preference" in history[0]["tools_called"]
    print("SMOKE OK: agent graph + memory round-trip + preference round-trip (fake model).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
