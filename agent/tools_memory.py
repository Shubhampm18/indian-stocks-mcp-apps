"""Memory tools as LangChain @tool functions (durable SQLite via memory.py).

The server's 7 stock tools come over MCP (``load_mcp_tools``); these six are
the agent's local, durable memory and run in-process. They delegate to
memory.py unchanged — no new storage logic here.

The docstrings below are the model-facing descriptions (LangChain derives the
tool schema from them plus the type hints), so they are written as
instructions to the model, not labels. Tools return JSON strings so tool
result content is deterministic and parseable downstream.

Use ``build_memory_tools(db_path)`` to bind the tools to a specific SQLite
file (tests); ``MEMORY_TOOLS`` is the default set bound to memory.DB_PATH.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import memory as mem
from langchain_core.tools import BaseTool, tool


def build_memory_tools(db_path: str | Path | None = None) -> list[BaseTool]:
    """Build the six memory tools bound to one SQLite file.

    ``db_path=None`` binds them to the default ``memory.DB_PATH``.
    """
    path = str(db_path) if db_path is not None else str(mem.DB_PATH)

    @tool
    def add_to_watchlist(symbol: str) -> str:
        """Add a stock symbol to the user's durable watchlist. Use when the
        user says things like "add TCS to my watchlist". Accepts a bare
        ticker ("tcs"), an NSE/BSE suffixed symbol ("TCS.NS", "TCS.BO") or an
        index alias ("NIFTY", "SENSEX", "BANKNIFTY"); the symbol is
        normalised and duplicates are ignored. Returns the full watchlist
        after the change."""
        return json.dumps(mem.add_to_watchlist(symbol, path), ensure_ascii=False)

    @tool
    def remove_from_watchlist(symbol: str) -> str:
        """Remove a symbol from the user's durable watchlist, using the same
        normalisation as adding ("tcs", "TCS" and "TCS.NS" all remove the
        same row; "NIFTY" removes "^NSEI"). Returns the full watchlist after
        the change."""
        return json.dumps(mem.remove_from_watchlist(symbol, path), ensure_ascii=False)

    @tool
    def list_watchlist() -> str:
        """List every symbol on the user's durable watchlist, in the order
        they were added. Call this before answering "what am I tracking?" —
        never answer from conversation context alone."""
        return json.dumps(mem.list_watchlist(path), ensure_ascii=False)

    @tool
    def recall_history(limit: int = 5, query: str | None = None) -> str:
        """Recall the user's past questions and answers from durable memory,
        newest first. Optional `query` filters rows whose question contains
        the text (case-insensitive); without it the most recent `limit`
        (default 5) turns are returned. Use for "what did I ask last time?"
        or to reuse figures from an earlier turn."""
        return json.dumps(
            mem.recent_queries(int(limit), path, query=query), ensure_ascii=False
        )

    @tool
    def set_preference(key: str, value: str) -> str:
        """Save a durable user preference as a key/value pair, overwriting
        any previous value for that key. Known keys: default_metric (peer
        ranking metric such as return_20d), default_benchmark (e.g.
        ^NSEI), default_symbol, default_index. Use when the user says
        "always rank by X" or "remember my default is X"."""
        mem.set_preference(key, value, path)
        return json.dumps({"key": key, "value": value, "saved": True},
                          ensure_ascii=False)

    @tool
    def get_preference(key: str, default: str | None = None) -> str:
        """Read a durable user preference by key. Returns
        {"key": ..., "value": ...} where value is null when unset (unless a
        `default` is given). Check default_metric/default_benchmark here
        before asking the user for a value they may already have set."""
        value = mem.get_preference(key, default=default, db_path=path)
        return json.dumps({"key": key, "value": value}, ensure_ascii=False)

    return [
        add_to_watchlist,
        remove_from_watchlist,
        list_watchlist,
        recall_history,
        set_preference,
        get_preference,
    ]


MEMORY_TOOLS: list[BaseTool] = build_memory_tools()

__all__ = ["build_memory_tools", "MEMORY_TOOLS"]
