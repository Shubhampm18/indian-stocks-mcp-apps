"""Smoke test: agent loop against a stub LLM (offline, no API key needed).

Proves the tool-calling loop completes a multi-step task (memory tool
then stock tool then answer) and logs the turn to memory. Run with:
  uv run python tests/smoke_agent.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class StubLLM:
    """Fake LLM: add to watchlist, set a preference, get it back, answer."""

    def __init__(self):
        self.calls = 0

    def complete(self, messages, tools):
        self.calls += 1
        if self.calls == 1:
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "t1", "name": "add_to_watchlist", "args": {"symbol": "TCS.NS"}}
                ],
            }
        if self.calls == 2:
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "t2", "name": "set_preference",
                     "args": {"key": "default_metric", "value": "return_20d"}}
                ],
            }
        if self.calls == 3:
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "t3", "name": "get_preference", "args": {"key": "default_metric"}}
                ],
            }
        if self.calls == 4:
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "t4", "name": "list_watchlist", "args": {}}],
            }
        return {"role": "assistant",
                "content": "TCS.NS is on your watchlist; default metric is return_20d.",
                "tool_calls": []}


def main() -> int:
    import json

    import memory as mem
    from agent import Agent

    db = os.path.join(tempfile.mkdtemp(), "smoke.db")
    agent_obj = Agent(session=None, client=StubLLM(), db_path=db)
    answer = asyncio.run(agent_obj.ask(
        "add TCS.NS to my watchlist, save return_20d as my default metric, then show it"))
    assert "TCS.NS" in answer
    assert mem.list_watchlist(db) == ["TCS.NS"], mem.list_watchlist(db)
    # the preference round-tripped THROUGH the agent loop (t2 set, t3 read)
    assert mem.get_preference("default_metric", db_path=db) == "return_20d"
    assert "default metric is return_20d" in answer
    # the tool message the agent received carried the stored value back
    tool_msgs = [m for m in agent_obj.messages if m.get("role") == "tool"]
    pref_payloads = [json.loads(m["content"]) for m in tool_msgs
                     if m.get("name") == "get_preference"]
    assert pref_payloads and pref_payloads[0]["value"] == "return_20d", pref_payloads
    history = mem.recent_queries(5, db)
    assert len(history) == 1 and "set_preference" in history[0]["tools_called"]
    assert "get_preference" in history[0]["tools_called"]
    print("SMOKE OK: agent loop + memory round-trip + preference round-trip (stub LLM).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
