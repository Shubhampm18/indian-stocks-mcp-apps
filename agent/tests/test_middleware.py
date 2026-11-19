"""Phase 3 tests: middleware (memory context, defaults, grounding guard).

All offline: fake chat model, local fake "server" tool (named exactly like a
server tool so the guard's name check fires), tmp SQLite DBs.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import memory as mem
from app_agent import Agent
from fake_model import FakeChatModel, tool_call
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from middleware import (  # noqa: E402
    GroundingGuard,
    MemoryContext,
    PreferenceDefaults,
    extract_claims,
)

# The fake "server tool" — same NAME as a real server tool so the guard
# treats its ToolMessage as server output.
TOOL_PAYLOAD = {"current_price": 1570.4, "data_through": "2025-12-31",
                "currency": "INR"}


@tool
def get_technical_indicators(symbol: str, as_of: str | None = None) -> str:
    """Fake server tool returning the canned payload."""
    return json.dumps(TOOL_PAYLOAD)


def build(db_path, scripted, middleware) -> object:
    return create_agent(
        model=FakeChatModel(scripted=scripted),
        tools=[get_technical_indicators] + build_memory_tools(db_path),
        system_prompt="P",
        middleware=middleware,
    )


def build_memory_tools(db_path):
    from tools_memory import build_memory_tools as b
    return b(db_path)


def run(graph, question="q"):
    return asyncio.run(graph.ainvoke(
        {"messages": [HumanMessage(content=question)]},
        config={"recursion_limit": 40},
    ))


# ---------------- claim extraction / matching ----------------

def test_claims_decimal_and_currency_and_percent():
    claims = extract_claims("Price Rs 1,570.40, up 2.5%, volume 12%")
    values = [c["value"] for c in claims]
    assert 1570.4 in values and 2.5 in values and 12.0 in values
    tokens = [c["token"] for c in claims]
    assert "Rs 1,570.40" in tokens


def test_years_and_small_integers_ignored():
    assert extract_claims("as of 2025-12-31, 7 tools, step 3 of 9") == []
    # 2025.5 is not a year (not integral) and carries a decimal point
    claims = extract_claims("price hit 2025.5")
    assert [c["value"] for c in claims] == [2025.5]


def test_rs_decimal_matches_plain_float():
    """'Rs 1,570.40' must match tool number 1570.4 (2-decimal equality)."""
    claims = extract_claims("closed at Rs 1,570.40")
    from middleware import _matches
    assert all(_matches(c["value"], [1570.4]) for c in claims)


def test_relative_tolerance():
    from middleware import _matches
    assert _matches(1000.0, [1004.0])       # 0.4% off -> within 0.5%
    assert not _matches(1000.0, [1006.0])   # 0.6% off -> not matched


# ---------------- GroundingGuard through the graph ----------------

def test_invented_number_caught_and_repaired(tmp_path):
    db = str(tmp_path / "g1.db")
    guard = GroundingGuard(model=None, db_path=db)
    scripted = [
        tool_call("get_technical_indicators", {"symbol": "RELIANCE.NS"}, "t1"),
        AIMessage(content="RELIANCE closed at 1,999.99 (data_through 2025-12-31)."),
        AIMessage(content="RELIANCE closed at Rs 1,570.40 (data_through 2025-12-31)."),
    ]
    guard.model = FakeChatModel(scripted=[AIMessage(content=scripted[2].content)])
    graph = build(db, list(scripted), [guard])
    result = run(graph)
    assert result["messages"][-1].content.startswith("RELIANCE closed at Rs 1,570.40")
    assert "1,999.99" not in result["messages"][-1].content
    assert guard.last_result["grounded"] is True and guard.last_result["repaired"]


def test_repair_that_still_fails_gets_unverified_prefix(tmp_path):
    db = str(tmp_path / "g2.db")
    guard = GroundingGuard(model=None, db_path=db)
    scripted = [
        tool_call("get_technical_indicators", {"symbol": "RELIANCE.NS"}, "t1"),
        AIMessage(content="RELIANCE closed at 1,999.99."),
        AIMessage(content="RELIANCE closed at 1,888.88."),
    ]
    guard.model = FakeChatModel(scripted=[AIMessage(content=scripted[2].content)])
    graph = build(db, list(scripted), [guard])
    result = run(graph)
    assert result["messages"][-1].content.startswith("(unverified: 1 figures)")
    assert guard.last_result["grounded"] is False


def test_grounded_answer_passes_untouched(tmp_path):
    db = str(tmp_path / "g3.db")
    guard = GroundingGuard(model=None, db_path=db)
    scripted = [
        tool_call("get_technical_indicators", {"symbol": "RELIANCE.NS"}, "t1"),
        AIMessage(content="RELIANCE closed at Rs 1,570.40 (data_through 2025-12-31)."),
    ]
    graph = build(db, scripted, [guard])
    result = run(graph)
    assert result["messages"][-1].content == scripted[-1].content
    assert guard.last_result["grounded"] is True


def test_memory_only_turn_not_guarded(tmp_path):
    db = str(tmp_path / "g4.db")
    guard = GroundingGuard(model=None, db_path=db)
    scripted = [
        tool_call("add_to_watchlist", {"symbol": "TCS.NS"}, "t1"),
        AIMessage(content="Added. Your lucky number is 999.99."),
    ]
    graph = build(db, list(scripted), [guard])
    result = run(graph)
    # no repair happened: the invented figure stays, guard skipped the turn
    assert result["messages"][-1].content == scripted[-1].content
    assert guard.last_result == {"grounded": None, "claims": []}


# ---------------- PreferenceDefaults ----------------

def test_default_metric_injected_via_graph(tmp_path):
    """End-to-end: the model omits metric; the middleware fills it and notes it."""
    db = str(tmp_path / "pd2.db")
    mem.set_preference("default_metric", "return_20d", db)
    seen_args: list[dict] = []

    @tool
    def get_peer_comparison(symbols: list[str], as_of: str | None = None,
                            metric: str | None = None) -> str:
        """Fake peer comparison recording what actually arrived."""
        seen_args.append(dict(metric=metric))
        return json.dumps({"metric": metric, "results": [], "ranked": []})

    from middleware import PreferenceDefaults as PD
    scripted = [
        tool_call("get_peer_comparison", {"symbols": ["RELIANCE.NS", "TCS.NS"]}, "t1"),
        AIMessage(content="TCS beat RELIANCE."),
    ]
    graph = create_agent(
        model=FakeChatModel(scripted=scripted),
        tools=[get_peer_comparison] + build_memory_tools(db),
        system_prompt="P",
        middleware=[PD(db_path=db)],
    )
    run(graph)
    assert seen_args == [{"metric": "return_20d"}]


def test_explicit_metric_not_overridden(tmp_path):
    db = str(tmp_path / "pd3.db")
    mem.set_preference("default_metric", "return_20d", db)
    seen_args: list[dict] = []

    @tool
    def get_peer_comparison(symbols: list[str], as_of: str | None = None,
                            metric: str | None = None) -> str:
        """Fake peer comparison."""
        seen_args.append(dict(metric=metric))
        return json.dumps({"metric": metric, "results": [], "ranked": []})

    scripted = [
        tool_call("get_peer_comparison",
                  {"symbols": ["RELIANCE.NS"], "metric": "vol_60d"}, "t1"),
        AIMessage(content="done."),
    ]
    graph = create_agent(
        model=FakeChatModel(scripted=scripted),
        tools=[get_peer_comparison] + build_memory_tools(db),
        system_prompt="P",
        middleware=[PreferenceDefaults(db_path=db)],
    )
    run(graph)
    assert seen_args == [{"metric": "vol_60d"}]


# ---------------- MemoryContext ----------------

def test_memory_context_appended_to_system_message(tmp_path):
    db = str(tmp_path / "mc.db")
    mem.add_to_watchlist("TCS.NS", db)
    mem.set_preference("default_metric", "return_20d", db)
    mem.log_query("who beat whom?", ["t"], "TCS", db)
    captured: list[str] = []

    class Spy(FakeChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            captured.append(messages[0].content)
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    scripted = [AIMessage(content="ok")]
    graph = create_agent(
        model=Spy(scripted=scripted),
        tools=build_memory_tools(db),
        system_prompt="STATIC PROMPT",
        middleware=[MemoryContext(db_path=db)],
    )
    run(graph, "hello")
    # this-turn-only: static prompt untouched in principle, context appended
    assert captured[0].startswith("STATIC PROMPT")
    assert "Watchlist: TCS.NS" in captured[0]
    assert "default_metric=return_20d" in captured[0]
    assert "asked: who beat whom?" in captured[0]
    assert len(captured[0]) <= 600 + len("STATIC PROMPT") + 50


# ---------------- history migration ----------------

def test_history_roundtrips_new_columns(tmp_path):
    db = str(tmp_path / "hist.db")
    mem.log_query("q", ["get_ohlcv"], "answer", db,
                  as_of_used="2025-12-31", data_through="2025-12-31",
                  key_numbers='["1570.4"]', grounded=True)
    row = mem.recent_queries(1, db)[0]
    assert row["as_of_used"] == "2025-12-31"
    assert row["data_through"] == "2025-12-31"
    assert row["key_numbers"] == '["1570.4"]'
    assert row["grounded"] == 1


def test_log_query_still_callable_with_four_positional_args(tmp_path):
    db = str(tmp_path / "hist2.db")
    mem.log_query("q", [], "a", db)  # old signature
    row = mem.recent_queries(1, db)[0]
    assert row["question"] == "q" and row["grounded"] is None


def test_migration_from_old_schema(tmp_path):
    """A pre-migration DB (old history schema) opens and gains the columns."""
    db = tmp_path / "old.db"
    conn = sqlite3.connect(str(db))
    conn.executescript("""
        CREATE TABLE history (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            ts             TEXT NOT NULL,
            question       TEXT NOT NULL,
            tools_called   TEXT NOT NULL DEFAULT '[]',
            answer_summary TEXT NOT NULL DEFAULT ''
        );
        INSERT INTO history(ts, question, tools_called, answer_summary)
        VALUES ('2025-01-01T00:00:00+00:00', 'old question', '[]', 'old answer');
    """)
    conn.commit()
    conn.close()

    # opening through memory migrates in place
    mem.log_query("new question", [], "new answer", str(db), grounded=False)
    rows = mem.recent_queries(10, str(db))
    assert rows[0]["question"] == "new question"
    old = [r for r in rows if r["question"] == "old question"][0]
    assert old["as_of_used"] is None and old["grounded"] is None


def test_guard_load_bearing_through_agent(tmp_path):
    """The Agent's own wiring includes the guard: an invented figure gets
    repaired end-to-end (this is the test that fails if the guard is
    removed from app_agent.py's middleware list)."""
    db = str(tmp_path / "agent_guard.db")
    # One shared fake model = agent model + guard repair model:
    # call 1 -> tool call, call 2 -> draft with the invented figure,
    # call 3 (the guard's repair invoke) -> corrected text.
    scripted = [
        tool_call("get_technical_indicators", {"symbol": "RELIANCE.NS", "as_of": "2025-12-31"}, "t1"),
        AIMessage(content="RELIANCE closed at 1,999.99 (data_through 2025-12-31)."),
        AIMessage(content="RELIANCE closed at Rs 1,570.40 (data_through 2025-12-31)."),
    ]
    agent_obj = Agent(
        session=None,
        client=FakeChatModel(scripted=scripted),
        db_path=db,
        tools=[get_technical_indicators],
    )
    answer = asyncio.run(agent_obj.ask("RELIANCE price as of 2025-12-31?"))
    assert answer.startswith("RELIANCE closed at Rs 1,570.40"), answer
    row = mem.recent_queries(1, db)[0]
    assert row["grounded"] == 1  # SQLite INTEGER (bools stored as 0/1)
    assert row["as_of_used"] == "2025-12-31"
    assert row["data_through"] == "2025-12-31"


def test_three_parallel_calls_in_one_graph_step(tmp_path):
    """One AIMessage with three tool_calls runs in a single tool-node step;
    all three effects persist (WAL + write lock hold under the graph's
    parallel execution)."""
    db = str(tmp_path / "par3.db")
    scripted = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "add_to_watchlist", "args": {"symbol": "TCS.NS"},
                 "id": "p1", "type": "tool_call"},
                {"name": "add_to_watchlist", "args": {"symbol": "RELIANCE.NS"},
                 "id": "p2", "type": "tool_call"},
                {"name": "set_preference",
                 "args": {"key": "default_metric", "value": "return_20d"},
                 "id": "p3", "type": "tool_call"},
            ],
        ),
        AIMessage(content="Added both symbols and saved the metric."),
    ]
    agent_obj = Agent(session=None, client=FakeChatModel(scripted=scripted), db_path=db)
    answer = asyncio.run(agent_obj.ask("track TCS and RELIANCE, default to return_20d"))
    assert "Added both symbols" in answer
    assert mem.list_watchlist(db) == ["RELIANCE.NS", "TCS.NS"]
    assert mem.get_preference("default_metric", db_path=db) == "return_20d"
    row = mem.recent_queries(1, db)[0]
    assert sorted(json.loads(row["tools_called"])) == ["add_to_watchlist", "add_to_watchlist", "set_preference"]
