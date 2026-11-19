"""Durable memory for the stock-analyst agent (SQLite, stdlib only).

Three tables:
  watchlist(symbol PK, added_at)      -- symbols the user cares about
  history(id, ts, question,          -- one row per agent turn
           tools_called, answer_summary)
  preferences(key PK, value)          -- default symbol/metric/model etc.

The DB file lives next to this module (gitignored) and survives restarts,
unlike the in-process message list the agent keeps for live conversation.
No third-party deps: Python's `sqlite3` is in the standard library.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

DB_PATH = Path(__file__).with_name("memory.db")

# Writes are serialised process-wide: the LangGraph tool node may run tool
# calls in parallel threads, and SQLite connections are per-call.
_write_lock = threading.Lock()

# --- symbol normalisation (dependency-free by design) -----------------------
#
# Watchlist rows are keyed by symbol, so "tcs", "TCS" and "TCS.NS" must
# collapse to one row. memory.py must NOT import stock_analyst_mcp.data:
# it is imported by agent.py at startup and the app deliberately has no
# dependency on the server package (they talk over stdio). So we ship a
# small default normaliser implementing the same public rule (bare ticker
# = NSE guess, explicit .BO/.NS preserved, ^ indices untouched, index
# aliases mapped) and expose set_symbol_normaliser() so a caller that CAN
# import the server package can inject data.normalise_symbol for exact
# parity instead.

_normaliser: Callable[[str], str] = None  # set below


def _default_normalise_symbol(symbol: str) -> str:
    # alias table kept in sync with data.normalise_symbol — see parity test
    # (tests/test_memory.py::test_default_normaliser_parity_with_server)
    aliases = {"NIFTY": "^NSEI", "NIFTY50": "^NSEI", "NIFTY_50": "^NSEI",
               "NIFTY 50": "^NSEI", "BANKNIFTY": "^NSEBANK",
               "SENSEX": "^BSESN", "NIFTY BANK": "^NSEBANK"}
    s = symbol.strip().upper()
    if s.endswith((".NS", ".BO")) or s.startswith("^"):
        return s  # explicit suffix / index ticker wins — aliases never fire
    if "." in s:
        s = s.split(".")[0]
    if s in aliases:
        return aliases[s]
    return f"{s}.NS" if s else s


_normaliser = _default_normalise_symbol


def set_symbol_normaliser(fn: Callable[[str], str]) -> None:
    """Replace the symbol normaliser (e.g. with
    stock_analyst_mcp.data.normalise_symbol for exact server parity)."""
    global _normaliser
    _normaliser = fn

SCHEMA = """
CREATE TABLE IF NOT EXISTS watchlist (
    symbol   TEXT PRIMARY KEY,
    added_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS history (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts             TEXT NOT NULL,
    question       TEXT NOT NULL,
    tools_called   TEXT NOT NULL DEFAULT '[]',
    answer_summary TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS preferences (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _connect(db_path: str | Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    # WAL lets concurrent readers proceed during a write (parallel tool calls).
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------- watchlist ----------------

def add_to_watchlist(symbol: str, db_path: str | Path = DB_PATH) -> list[str]:
    sym = _normaliser(symbol)
    if not sym:
        raise ValueError("symbol must not be empty")
    with _write_lock, _connect(db_path) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO watchlist(symbol, added_at) VALUES (?, ?)",
            (sym, _now()),
        )
        conn.commit()
        return _watchlist(conn)


def remove_from_watchlist(symbol: str, db_path: str | Path = DB_PATH) -> list[str]:
    with _write_lock, _connect(db_path) as conn:
        conn.execute("DELETE FROM watchlist WHERE symbol = ?", (_normaliser(symbol),))
        conn.commit()
        return _watchlist(conn)


def list_watchlist(db_path: str | Path = DB_PATH) -> list[str]:
    with _connect(db_path) as conn:
        return _watchlist(conn)


def _watchlist(conn: sqlite3.Connection) -> list[str]:
    return [
        row[0]
        for row in conn.execute(
            "SELECT symbol FROM watchlist ORDER BY added_at, symbol"
        )
    ]


# ---------------- history ----------------

def log_query(
    question: str,
    tools_called: list[str],
    answer_summary: str,
    db_path: str | Path = DB_PATH,
) -> int:
    with _write_lock, _connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO history(ts, question, tools_called, answer_summary)"
            " VALUES (?, ?, ?, ?)",
            (_now(), question, json.dumps(list(tools_called)), answer_summary),
        )
        conn.commit()
        return int(cur.lastrowid)


def recent_queries(limit: int = 10, db_path: str | Path = DB_PATH,
                   query: str | None = None) -> list[dict]:
    """Recent agent turns, newest first.

    With ``query`` (optional), only rows whose question contains the text
    (case-insensitive SQL LIKE) are returned; without it, plain recency.
    """
    with _connect(db_path) as conn:
        if query:
            rows = conn.execute(
                "SELECT id, ts, question, tools_called, answer_summary"
                " FROM history WHERE question LIKE ? ORDER BY id DESC LIMIT ?",
                (f"%{query}%", int(limit)),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, ts, question, tools_called, answer_summary"
                " FROM history ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
    return [dict(row) for row in rows]


# ---------------- preferences ----------------

def set_preference(key: str, value: str, db_path: str | Path = DB_PATH) -> None:
    with _write_lock, _connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO preferences(key, value) VALUES (?, ?)",
            (key, str(value)),
        )
        conn.commit()


def get_preference(key: str, default: str | None = None, db_path: str | Path = DB_PATH) -> str | None:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT value FROM preferences WHERE key = ?", (key,)
        ).fetchone()
    return row[0] if row else default
