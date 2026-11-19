"""Phase C tests: grounding-guard decision table, mixed-turn policy, telemetry.

C1 is a table of (draft, payload, expected) driven end-to-end through
GroundingGuard.after_model: pass rows must not invoke the repair model at
all; fail rows must invoke it exactly once (the scripted repair repeats the
same draft, so the verdict stays 'flagged').

Mixed-turn policy is STRICT (see middleware.py module docstring): memory
payloads never validate claims — this file pins that behaviour.
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
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from middleware import GroundingGuard, extract_claims

# ---------------- C1: the decision table ----------------
# (name, draft, payload dict, expected grounded verdict on the DRAFT)
GUARD_CASES = [
    ("year_ignored",
     "As of 2025 the trend turned up.", {"current_price": 1570.4}, True),
    ("currency_2dp_matches",
     "Closed at Rs 1,570.40.", {"current_price": 1570.4}, True),
    ("percent_matches",
     "RSI is at 6.59%.", {"rsi_14": 6.59}, True),
    ("negative_percent_matches_unsigned_payload",
     "It fell -19.8% from the peak.", {"drawdown_60d": 19.8}, True),
    ("invented_thousands_fail",
     "It jumped to 1,999.99.", {"current_price": 1570.4}, False),
    ("ratio_absent_fail",
     "Volume was 1570.40x its average.", {"rsi_14": 61.2}, False),
    ("ratio_unit_blind_match_documented",
     # unit-blind matching: 1570.40x parses to 1570.4 and MATCHES a payload
     # that merely contains 1570.4 — documented limitation, see module doc.
     "Volume was 1570.40x its average.", {"current_price": 1570.4}, True),
    ("multiple_x_matches_only_if_base_present",
     "Traded about 6x its low.", {"lots": 6}, True),
    ("multiple_x_absent_fail",
     "Traded about 6x its low.", {"current_price": 1570.4}, False),
    ("unit_scaling_lakh_fail_documented",
     # "1.57 lakh" is NOT normalised to 157000 — the claim stays 1.57 and
     # fails against a payload holding the scaled number. Documented.
     "Market cap is Rs 1.57 lakh crore.", {"market_cap": 157000.0}, False),
    ("string_field_number_matches",
     "Closed at Rs 1,570.40.", {"price_str": "1,570.40"}, True),
    ("zero_claims_grounded",
     "The trend is plainly bullish.", {"current_price": 1570.4}, True),
]


def _state(draft: str, payload: dict) -> dict:
    return {
        "messages": [
            HumanMessage(content="q"),
            AIMessage(content="", tool_calls=[
                {"name": "get_ohlcv", "args": {}, "id": "t1", "type": "tool_call"}]),
            ToolMessage(content=json.dumps(payload), name="get_ohlcv",
                        tool_call_id="t1"),
            AIMessage(content=draft),
        ]
    }


@pytest.mark.parametrize("name,draft,payload,expected", GUARD_CASES,
                         ids=[c[0] for c in GUARD_CASES])
def test_guard_decision_table(name, draft, payload, expected):
    """after_model on the table; the repair model echoes the draft back so
    fail rows keep their verdict through the re-check."""
    repair_model = FakeChatModel(scripted=[AIMessage(content=draft)])
    guard = GroundingGuard(model=repair_model)
    guard.after_model(_state(draft, payload), runtime=None)
    result = guard.last_result
    assert result is not None, name
    assert result["grounded"] is expected, (name, result)
    if expected:
        assert repair_model.i == 0, f"{name}: grounded draft must not trigger repair"
        assert result["action"] == "pass"
    else:
        assert repair_model.i == 1, f"{name}: ungrounded draft must repair once"
        assert result["action"] == "flagged"


def test_zero_claims_model_called_once():
    """'the model was called once': the guard's repair model is called zero
    times for a claim-free answer — the only model call in this flow is the
    one that produced the answer itself (not visible to the guard)."""
    repair_model = FakeChatModel(scripted=[])
    guard = GroundingGuard(model=repair_model)
    guard.after_model(_state("The trend is plainly bullish.",
                             {"current_price": 1570.4}), runtime=None)
    assert guard.last_result == {"grounded": True,
                                 "claims": [], "action": "pass"}
    assert repair_model.i == 0


# ---------------- C2: strict mixed-turn policy ----------------

@tool
def get_ohlcv(symbol: str, as_of: str | None = None) -> str:
    """Fake server tool."""
    return json.dumps({"current_price": 1570.4, "data_through": "2025-12-31"})


def test_mixed_turn_memory_payload_never_validates(tmp_path):
    """STRICT policy: in a turn that called a server tool AND a memory tool,
    a claim that exists only inside the memory payload is still flagged.
    The scripted repair repeats the same figure, so the verdict stays
    'flagged' — had the memory payload been scanned, "3.0" (id 3) would have
    grounded the draft without any repair call."""
    db = str(tmp_path / "mixed.db")
    mem.log_query("how many queries so far?", [], "3", db)  # history id 3
    recall_payload = json.dumps(mem.recent_queries(5, db))  # contains "id": 3

    mixed_msgs = [
        HumanMessage(content="q"),
        AIMessage(content="", tool_calls=[
            {"name": "get_ohlcv", "args": {}, "id": "t1", "type": "tool_call"},
            {"name": "recall_history", "args": {}, "id": "t2", "type": "tool_call"},
        ]),
        ToolMessage(content=json.dumps({"current_price": 1570.4}),
                    name="get_ohlcv", tool_call_id="t1"),
        ToolMessage(content=recall_payload, name="recall_history",
                    tool_call_id="t2"),
        AIMessage(content="You have logged 3.0 queries so far."),
    ]
    repair_model = FakeChatModel(scripted=[
        AIMessage(content="You have logged 3.0 queries so far.")])
    guard = GroundingGuard(model=repair_model, db_path=db)
    guard.after_model({"messages": mixed_msgs}, runtime=None)
    # "3.0" exists in the MEMORY payload (id 3) but memory never validates:
    assert guard.last_result["grounded"] is False
    assert guard.last_result["action"] == "flagged"
    assert repair_model.i == 1


def test_mixed_turn_server_figure_passes(tmp_path):
    """Same mixed shape, but the quoted figure comes from the server payload:
    grounded ('pass'), the memory presence is irrelevant."""
    db = str(tmp_path / "mixed2.db")
    mixed_msgs = [
        HumanMessage(content="q"),
        AIMessage(content="", tool_calls=[
            {"name": "get_ohlcv", "args": {}, "id": "t1", "type": "tool_call"},
            {"name": "list_watchlist", "args": {}, "id": "t2", "type": "tool_call"},
        ]),
        ToolMessage(content=json.dumps({"current_price": 1570.4}),
                    name="get_ohlcv", tool_call_id="t1"),
        ToolMessage(content=json.dumps(["TCS.NS"]), name="list_watchlist",
                    tool_call_id="t2"),
        AIMessage(content="RELIANCE closed at Rs 1,570.40."),
    ]
    repair_model = FakeChatModel(scripted=[])
    guard = GroundingGuard(model=repair_model, db_path=db)
    guard.after_model({"messages": mixed_msgs}, runtime=None)
    assert guard.last_result["grounded"] is True
    assert guard.last_result["action"] == "pass"
    assert repair_model.i == 0


# ---------------- C3: guard_action telemetry ----------------

def _agent_with(scripted, db, payload):
    @tool
    def get_ohlcv(symbol: str, as_of: str | None = None) -> str:
        """Fake server tool."""
        return json.dumps(payload)

    return Agent(session=None, client=FakeChatModel(scripted=scripted),
                 db_path=db, tools=[get_ohlcv])


def test_repaired_turn_stores_repaired(tmp_path):
    db = str(tmp_path / "tel1.db")
    scripted = [
        tool_call("get_ohlcv", {"symbol": "RELIANCE.NS"}, "t1"),
        AIMessage(content="RELIANCE closed at 1,999.99 (data_through 2025-12-31)."),
        AIMessage(content="RELIANCE closed at Rs 1,570.40 (data_through 2025-12-31)."),
    ]
    agent = _agent_with(scripted, db, {"current_price": 1570.4,
                                       "data_through": "2025-12-31"})
    answer = asyncio.run(agent.ask("RELIANCE price as of 2025-12-31?"))
    assert answer.startswith("RELIANCE closed at Rs 1,570.40")
    row = mem.recent_queries(1, db)[0]
    assert row["guard_action"] == "repaired"


def test_grounded_turn_stores_pass(tmp_path):
    db = str(tmp_path / "tel2.db")
    scripted = [
        tool_call("get_ohlcv", {"symbol": "RELIANCE.NS"}, "t1"),
        AIMessage(content="RELIANCE closed at Rs 1,570.40 (data_through 2025-12-31)."),
    ]
    agent = _agent_with(scripted, db, {"current_price": 1570.4,
                                       "data_through": "2025-12-31"})
    asyncio.run(agent.ask("RELIANCE price as of 2025-12-31?"))
    row = mem.recent_queries(1, db)[0]
    assert row["guard_action"] == "pass"


def test_memory_only_turn_stores_null(tmp_path):
    db = str(tmp_path / "tel3.db")
    scripted = [
        tool_call("list_watchlist", {}, "t1"),
        AIMessage(content="Your watchlist is empty; nothing to guard 999.99."),
    ]
    agent = _agent_with(scripted, db, {"current_price": 1570.4})
    asyncio.run(agent.ask("what's on my watchlist?"))
    row = mem.recent_queries(1, db)[0]
    assert row["guard_action"] is None


def test_flagged_turn_stores_flagged(tmp_path):
    db = str(tmp_path / "tel4.db")
    scripted = [
        tool_call("get_ohlcv", {"symbol": "RELIANCE.NS"}, "t1"),
        AIMessage(content="RELIANCE closed at 1,999.99."),
        AIMessage(content="RELIANCE closed at 1,888.88."),
    ]
    agent = _agent_with(scripted, db, {"current_price": 1570.4,
                                       "data_through": "2025-12-31"})
    answer = asyncio.run(agent.ask("RELIANCE price as of 2025-12-31?"))
    assert answer.startswith("(unverified: 1 figures)")
    row = mem.recent_queries(1, db)[0]
    assert row["guard_action"] == "flagged"


def test_guard_action_migration_from_old_schema(tmp_path):
    """A DB created before guard_action gains the column on connect."""
    db = tmp_path / "old_guard.db"
    conn = sqlite3.connect(str(db))
    conn.executescript("""
        CREATE TABLE history (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,
            question TEXT NOT NULL, tools_called TEXT NOT NULL DEFAULT '[]',
            answer_summary TEXT NOT NULL DEFAULT '',
            as_of_used TEXT, data_through TEXT, key_numbers TEXT,
            grounded INTEGER
        );
    """)
    conn.commit()
    conn.close()
    mem.log_query("q", [], "a", str(db), guard_action="pass")
    row = mem.recent_queries(1, str(db))[0]
    assert row["guard_action"] == "pass"
