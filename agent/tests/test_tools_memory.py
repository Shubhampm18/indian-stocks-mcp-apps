"""Tests for the LangChain memory tools and SQLite-under-parallel-calls.

Offline: memory tools run in-process; the server-side 7 are checked by
test_schemas.py via the installed package's in-process FastMCP instance.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import memory as mem
from tools_memory import MEMORY_TOOLS, build_memory_tools

EXPECTED_MEMORY_TOOLS = [
    "add_to_watchlist",
    "remove_from_watchlist",
    "list_watchlist",
    "recall_history",
    "set_preference",
    "get_preference",
]


def test_six_memory_tools_registered():
    assert sorted(t.name for t in MEMORY_TOOLS) == sorted(EXPECTED_MEMORY_TOOLS)


def test_thirteen_tools_registered():
    """7 server tools (asserted in detail by test_schemas.py) + 6 memory = 13."""
    server = pytest.importorskip("stock_analyst_mcp.server")
    registered = asyncio.run(server.mcp.list_tools())
    assert len(registered) == 7
    assert len(MEMORY_TOOLS) == 6
    assert len(registered) + len(MEMORY_TOOLS) == 13


def test_memory_tools_roundtrip(tmp_path):
    db = str(tmp_path / "rt.db")
    tools = {t.name: t for t in build_memory_tools(db)}

    out = json.loads(tools["add_to_watchlist"].invoke({"symbol": "tcs"}))
    assert out == ["TCS.NS"]

    tools["set_preference"].invoke({"key": "default_metric", "value": "return_20d"})
    out = json.loads(tools["get_preference"].invoke({"key": "default_metric"}))
    assert out == {"key": "default_metric", "value": "return_20d"}

    out = json.loads(tools["get_preference"].invoke({"key": "nope", "default": "x"}))
    assert out == {"key": "nope", "value": "x"}

    out = json.loads(tools["recall_history"].invoke({"limit": 5}))
    assert out == []  # nothing logged yet

    tools["remove_from_watchlist"].invoke({"symbol": "TCS"})
    assert mem.list_watchlist(db) == []


def test_concurrent_adds_both_persist(tmp_path):
    """Two parallel add_to_watchlist calls (as the LangGraph tool node may
    issue them) must both persist — WAL + write lock."""
    db = str(tmp_path / "conc.db")
    done: list[str] = []

    def add(sym: str) -> None:
        done.append(mem.add_to_watchlist(sym, db)[0])

    threads = [threading.Thread(target=add, args=("TCS.NS",)),
               threading.Thread(target=add, args=("RELIANCE.NS",))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert sorted(done) == ["RELIANCE.NS", "TCS.NS"]
    assert mem.list_watchlist(db) == ["RELIANCE.NS", "TCS.NS"]


def test_concurrent_writes_under_lock(tmp_path):
    """Hammer the same DB from many threads; every write must land."""
    db = str(tmp_path / "hammer.db")
    n = 16

    def write(i: int) -> None:
        mem.set_preference(f"key_{i}", str(i), db)
        mem.log_query(f"q{i}", ["t"], "a", db)

    threads = [threading.Thread(target=write, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert len(mem.recent_queries(100, db)) == n
    assert all(mem.get_preference(f"key_{i}", db_path=db) == str(i) for i in range(n))
