"""Tests for the agent's durable memory (SQLite, offline)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import memory as mem


@pytest.fixture()
def db(tmp_path):
    return str(tmp_path / "mem.db")


def test_watchlist_round_trip(db):
    assert mem.add_to_watchlist("RELIANCE.NS", db) == ["RELIANCE.NS"]
    assert mem.add_to_watchlist("TCS.NS", db) == ["RELIANCE.NS", "TCS.NS"]
    # duplicate add is idempotent
    assert mem.add_to_watchlist("TCS.NS", db) == ["RELIANCE.NS", "TCS.NS"]
    assert mem.remove_from_watchlist("RELIANCE.NS", db) == ["TCS.NS"]
    assert mem.list_watchlist(db) == ["TCS.NS"]


def test_watchlist_rejects_empty(db):
    with pytest.raises(ValueError):
        mem.add_to_watchlist("  ", db)


def test_watchlist_normalises_case_and_suffix(db):
    """'tcs', 'TCS' and 'TCS.NS' are one company — one row, not three."""
    mem.add_to_watchlist("tcs", db)
    mem.add_to_watchlist("TCS", db)
    mem.add_to_watchlist("TCS.NS", db)
    assert mem.list_watchlist(db) == ["TCS.NS"]
    # removal also normalises: 'tcs' removes the TCS.NS row
    assert mem.remove_from_watchlist("tcs", db) == []
    mem.add_to_watchlist("TCS.NS", db)
    assert mem.remove_from_watchlist("TCS", db) == []


def test_recall_history_query_filter(db):
    mem.log_query("who beat whom?", ["get_peer_comparison"], "TCS won", db)
    mem.log_query("RELIANCE trend?", ["get_technical_indicators"], "bullish", db)
    mem.log_query("TCS vs RELIANCE RSI", ["get_peer_comparison"], "TCS", db)
    # filtered: only matching questions, newest first
    hits = mem.recent_queries(10, db, query="reliance")
    assert [h["question"] for h in hits] == ["TCS vs RELIANCE RSI", "RELIANCE trend?"]
    # no match -> empty; no query -> plain recency (existing behaviour)
    assert mem.recent_queries(10, db, query="nonexistent-xyz") == []
    assert len(mem.recent_queries(10, db)) == 3


def test_default_normaliser_index_aliases():
    """Index aliases must survive normalisation (regression: "NIFTY" used to
    become the non-existent "NIFTY.NS")."""
    n = mem._default_normalise_symbol
    assert n("NIFTY") == "^NSEI"
    assert n("SENSEX") == "^BSESN"
    assert n("nifty bank") == "^NSEBANK"
    assert n("NIFTY 50") == "^NSEI"
    assert n("BANKNIFTY") == "^NSEBANK"
    # explicit suffix wins — aliases never fire for "NIFTY.NS"
    assert n("NIFTY.NS") == "NIFTY.NS"
    # plain equities unchanged
    assert n("reliance") == "RELIANCE.NS"


def test_watchlist_alias_dedup(db):
    """'NIFTY' and '^NSEI' are the same instrument — one row."""
    mem.add_to_watchlist("NIFTY", db)
    mem.add_to_watchlist("^NSEI", db)
    assert mem.list_watchlist(db) == ["^NSEI"]
    # and removing via the other spelling clears it
    assert mem.remove_from_watchlist("NIFTY", db) == []


def test_default_normaliser_parity_with_server():
    """memory._default_normalise_symbol must agree with
    stock_analyst_mcp.data.normalise_symbol on every shared input.

    Guarded with importorskip so the app suite still runs in a venv
    without the server package (the dev path-dep is optional to install)."""
    server_data = pytest.importorskip("stock_analyst_mcp.data")
    inputs = [
        "NIFTY", "nifty50", "NIFTY_50", "NIFTY 50", "SENSEX", "nifty bank",
        "BANKNIFTY", "NIFTY.NS", "^NSEI", "reliance", "TCS.BO", "tcs",
        "RELIANCE.NS", "TCS.NS", "HDFCBANK", "ITC", "sbin", "TataMotors.NS",
    ]
    for raw in inputs:
        assert mem._default_normalise_symbol(raw) == server_data.normalise_symbol(raw), raw


def test_history_log_and_recall(db):
    mem.log_query("who beat whom?", ["get_peer_comparison"], "TCS won", db)
    mem.log_query("RELIANCE trend?", ["get_technical_indicators"], "bullish", db)
    recent = mem.recent_queries(5, db)
    assert len(recent) == 2
    assert recent[0]["question"] == "RELIANCE trend?"  # newest first
    assert "get_peer_comparison" in recent[1]["tools_called"]


def test_preferences(db):
    assert mem.get_preference("default_metric", db_path=db) is None
    mem.set_preference("default_metric", "return_20d", db)
    assert mem.get_preference("default_metric", db_path=db) == "return_20d"
    mem.set_preference("default_metric", "vol_60d", db)
    assert mem.get_preference("default_metric", db_path=db) == "vol_60d"
