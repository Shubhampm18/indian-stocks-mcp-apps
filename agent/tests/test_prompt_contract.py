"""Phase D tests: the prompt's honesty rules are contractual.

D1 — every rule below is a product promise (never invent figures, state
data_through, pass as_of, bare ticker = NSE, widen the pattern window). A
later prompt tweak that deletes one must fail here, not in production.

D2 — a no-tool turn quoting figures is flagged outright by the guard.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import memory as mem
from app_agent import STATIC_PROMPT, Agent
from fake_model import FakeChatModel, tool_call
from langchain_core.messages import AIMessage

# ---------------- D1: prompt contract ----------------

HONESTY_RULES = [
    "Never invent numbers",          # figures trace to tool output
    "data_through",                  # staleness always stated
    "as_of",                         # point-in-time discipline
    "bare ticker = NSE",             # symbol convention
    "`window` bars",                 # chart-pattern scan-window caveat
]


@pytest.mark.parametrize("rule", HONESTY_RULES,
                         ids=[r.replace(" ", "_").replace("`", "") for r in HONESTY_RULES])
def test_static_prompt_keeps_honesty_rule(rule):
    assert rule in STATIC_PROMPT, (
        f"STATIC_PROMPT lost the honesty rule {rule!r} — that is a product promise"
    )


def test_static_prompt_has_no_tool_mechanics_regression():
    """The prompt stays rule-only: tool-selection mechanics live in tool
    docstrings, so a tool rename must not require a prompt edit."""
    assert "get_ohlcv" not in STATIC_PROMPT
    assert "get_peer_comparison" not in STATIC_PROMPT


# ---------------- D2: no-tool price answers are flagged ----------------

def test_no_tool_price_answer_is_flagged(tmp_path):
    """A draft answering a price question with NO tool call at all has
    nothing behind it: the guard prefixes it and records 'flagged'."""
    db = str(tmp_path / "notool.db")
    scripted = [AIMessage(content="RELIANCE closed at Rs 1,570.40 yesterday.")]
    agent = Agent(session=None, client=FakeChatModel(scripted=scripted), db_path=db)
    answer = asyncio.run(agent.ask("RELIANCE price?"))
    assert answer.startswith("(unverified: 1 figures)"), answer
    assert "Rs 1,570.40" in answer  # text kept, honesty prefix added
    row = mem.recent_queries(1, db)[0]
    assert row["guard_action"] == "flagged"
    assert row["grounded"] == 0


def test_no_tool_answer_without_figures_passes_through(tmp_path):
    """Pure conversation (no claims, no tools) is untouched — NULL action."""
    db = str(tmp_path / "notool2.db")
    scripted = [AIMessage(content="The trend looks supportive, but check the data.")]
    agent = Agent(session=None, client=FakeChatModel(scripted=scripted), db_path=db)
    answer = asyncio.run(agent.ask("vibes?"))
    assert answer == "The trend looks supportive, but check the data."
    row = mem.recent_queries(1, db)[0]
    assert row["guard_action"] is None


def test_memory_only_turn_does_not_flag_its_own_record(tmp_path):
    """The no-tool flag must not fire on a recall answer whose figures come
    from the user's stored history (that payload is its source)."""
    db = str(tmp_path / "notool3.db")
    mem.log_query("RELIANCE price?", [], "it quoted 1570.40", db)
    scripted = [
        tool_call("recall_history", {}, "t1"),
        AIMessage(content="You asked about RELIANCE; it quoted 1,570.40 then."),
    ]
    agent = Agent(session=None, client=FakeChatModel(scripted=scripted), db_path=db)
    answer = asyncio.run(agent.ask("what did I ask last time?"))
    assert "unverified" not in answer, answer
    row = mem.recent_queries(1, db)[0]
    assert row["guard_action"] == "pass"
