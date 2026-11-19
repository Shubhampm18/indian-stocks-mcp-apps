"""Agentic loop for indian-stocks-mcp: LLM + MCP tools + durable memory.

The agent talks to a (cloud) LLM that decides which of the 7 MCP stock
tools to call, chains calls (e.g. compare -> rank -> summarize), and
answers in plain English. Durable memory (SQLite via memory.py) tracks
the user's watchlist, past queries, and preferences across restarts;
an in-process message list keeps only the live conversation.

The LLM provider is configurable by environment:
  LLM_API_KEY   required (or a provider-specific key such as OPENAI_API_KEY)
  LLM_BASE_URL  OpenAI-compatible base URL (optional)
  LLM_MODEL     model name (optional, default "gpt-4o-mini")
  LLM_TIMEOUT_SECONDS
                per-request timeout in seconds (optional, default 60).
                A timeout raises RuntimeError (chaining the SDK's
                APITimeoutError) instead of blocking forever.

Without a key the agent refuses to start — the MCP client demos
(`client.py --demo`) remain the way to exercise the server offline.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Protocol

import memory as mem


def _load_dotenv() -> None:
    """Load a local .env file (next to this module) into os.environ.

    Only keys not already set in the real environment are loaded, so the
    shell always wins. No third-party dependency needed.
    """
    env_path = Path(__file__).with_name(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv()

SYSTEM_PROMPT = """You are a stock research agent for India (NSE/BSE) stocks.
You have 7 tools over historical price data. Rules:
- Never invent numbers: compute every figure from tool output.
- Prefer get_ohlcv/get_returns/get_technical_indicators for price questions,
  get_peer_comparison for "which beat which", get_index_history for NIFTY/SENSEX.
- Use get_chart_patterns for 'any hammer/doji/support/resistance?' questions.
  The detector scans only the last `window` bars (default 20) of the returned
  frame — a pattern older than the scan window is invisible BY CONSTRUCTION.
  If the user asks about older dates, raise `window` (e.g. 60) up to the
  lookback length; never assume a pattern is absent without checking the
  window covers the dates in question.
- Always pass `as_of` (YYYY-MM-DD) when the user gives a date. Report
  `data_through` alongside any number you quote.
- Symbols: bare ticker = NSE guess, use .NS/.BO suffixes when known.
- You also have memory tools: add_to_watchlist/remove_from_watchlist/
  list_watchlist/recall_history. Use them when the user
  says things like "add X to my watchlist" or "what did I ask last time?".
- Keep answers short: the key number, the rank/trend, then data_through.
"""

STOCK_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_ohlcv",
            "description": "Daily OHLC + rebuilt adjusted close for a symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "as_of": {"type": "string"},
                    "lookback_days": {"type": "integer"},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_returns",
            "description": "Simple/log returns over windows.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "as_of": {"type": "string"},
                    "windows": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_technical_indicators",
            "description": "SMA/EMA/RSI/MACD/Bollinger/ATR/VWAP + trend.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "as_of": {"type": "string"},
                    "indicators": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_chart_patterns",
            "description": ("Candlestick patterns (hammer, doji, engulfing, stars) and "
                            "support/resistance levels. Only the last `window` bars "
                            "(default 20) are scanned — raise window to look further back."),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "as_of": {"type": "string"},
                    "lookback_days": {"type": "integer"},
                    "window": {"type": "integer",
                               "description": "Trailing-bar scan width for the detector (default 20)."},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_risk_metrics",
            "description": "Sharpe/Sortino/max drawdown/volatility/beta/alpha.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "as_of": {"type": "string"},
                    "benchmark": {"type": "string"},
                    "lookback_days": {"type": "integer"},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_index_history",
            "description": "Index close history (^NSEI, ^BSESN, ^NSEBANK).",
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {"type": "string"},
                    "as_of": {"type": "string"},
                    "lookback_days": {"type": "integer"},
                },
                "required": ["index"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_peer_comparison",
            "description": "Rank peers on one metric (return_20d, return_60d, vol_60d, sharpe_60d, drawdown_60d, rsi_14).",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbols": {"type": "array", "items": {"type": "string"}},
                    "as_of": {"type": "string"},
                    "metric": {"type": "string"},
                },
                "required": ["symbols"],
            },
        },
    },
]

MEMORY_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "add_to_watchlist",
            "description": "Add a symbol to the user's durable watchlist.",
            "parameters": {
                "type": "object",
                "properties": {"symbol": {"type": "string"}},
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_from_watchlist",
            "description": "Remove a symbol from the watchlist.",
            "parameters": {
                "type": "object",
                "properties": {"symbol": {"type": "string"}},
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_watchlist",
            "description": "List the user's watchlist symbols.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_history",
            "description": ("Recall past questions/answers (durable memory). "
                            "Optional `query` filters by question text; without it, "
                            "the most recent turns are returned."),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer"},
                    "query": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_preference",
            "description": ("Save a durable user preference (e.g. key='default_symbol', "
                            "value='RELIANCE.NS'). Overwrites the existing value for the key."),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["key", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_preference",
            "description": "Read a durable user preference by key.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "default": {"type": "string"},
                },
                "required": ["key"],
            },
        },
    },
]

ALL_TOOLS = STOCK_TOOL_SCHEMAS + MEMORY_TOOL_SCHEMAS

MAX_TOOL_STEPS = 6


class LLMClient(Protocol):
    """Minimal OpenAI-compatible chat client used by the agent loop."""

    def complete(self, messages: list[dict[str, Any]], tools: list[dict]) -> dict[str, Any]:
        ...


class OpenAICompatibleClient:
    """Talks to any OpenAI-compatible chat-completions endpoint."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", base_url: str | None = None,
                 timeout: float | None = 60.0):
        from openai import OpenAI

        if timeout is None:
            timeout = 60.0
        timeout = float(timeout)
        if timeout <= 0:
            raise ValueError("LLM timeout must be a positive number of seconds")
        kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self.model = model
        self.timeout = timeout

    def _normalise_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert internal message dicts to the wire format.

        Internal tool calls are stored as {"id", "name", "args"}; NVIDIA
        (and most strict OpenAI-compatible servers) require the wire
        format {"id", "type": "function", "function": {"name", "arguments"}}.
        """
        out = []
        for m in messages:
            m = dict(m)
            calls = m.get("tool_calls") or []
            if calls:
                wire = []
                for c in calls:
                    # Already in wire format? leave untouched.
                    if "function" in c and isinstance(c["function"], dict):
                        wire.append(c)
                    else:
                        wire.append({
                            "id": c.get("id", c.get("name")),
                            "type": "function",
                            "function": {
                                "name": c.get("name"),
                                "arguments": json.dumps(c.get("args") or {}),
                            },
                        })
                m["tool_calls"] = wire
            out.append(m)
        return out

    def complete(self, messages, tools):
        from openai import APITimeoutError

        try:
            resp = self._client.chat.completions.create(
                model=self.model, messages=self._normalise_messages(messages),
                tools=tools, tool_choice="auto"
            )
        except APITimeoutError as exc:
            raise RuntimeError(
                f"LLM request to model '{self.model}' timed out after "
                f"{self.timeout:g}s. The provider may be congested or the "
                f"request is too heavy; retry, or raise LLM_TIMEOUT_SECONDS."
            ) from exc
        msg = resp.choices[0].message
        calls = []
        for tc in msg.tool_calls or []:
            calls.append(
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "args": json.loads(tc.function.arguments or "{}"),
                }
            )
        return {"role": "assistant", "content": msg.content or "", "tool_calls": calls}


def make_client() -> LLMClient:
    """Build the LLM client from env vars. Raises if no key is configured."""
    key = (
        os.environ.get("LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
    )
    if not key:
        raise RuntimeError(
            "No LLM API key configured. Set LLM_API_KEY (or OPENAI_API_KEY) "
            "in your environment, or use `client.py --demo` which needs none."
        )
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    base_url = os.environ.get("LLM_BASE_URL") or None
    try:
        timeout: float = float(os.environ.get("LLM_TIMEOUT_SECONDS", "60"))
    except (TypeError, ValueError):
        raise ValueError(
            f"LLM_TIMEOUT_SECONDS must be a number, got "
            f"{os.environ.get('LLM_TIMEOUT_SECONDS')!r}."
        ) from None
    if timeout <= 0:
        raise ValueError(f"LLM_TIMEOUT_SECONDS must be positive, got {timeout:g}.")
    return OpenAICompatibleClient(api_key=key, model=model, base_url=base_url, timeout=timeout)


def _extract_payload(result) -> dict[str, Any]:
    text = getattr(result.content[0], "text", "") if result.content else ""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {"raw_text": text, "isError": bool(getattr(result, "isError", False))}


MEMORY_TOOL_NAMES = (
    "add_to_watchlist", "remove_from_watchlist", "list_watchlist",
    "recall_history", "set_preference", "get_preference",
)


async def run_memory_tool(name: str, args: dict[str, Any], db_path: str | Path | None = None) -> Any:
    """Execute a memory tool locally (against SQLite)."""
    path = db_path if db_path is not None else mem.DB_PATH
    if name == "add_to_watchlist":
        return mem.add_to_watchlist(str(args["symbol"]), path)
    if name == "remove_from_watchlist":
        return mem.remove_from_watchlist(str(args["symbol"]), path)
    if name == "list_watchlist":
        return mem.list_watchlist(path)
    if name == "recall_history":
        return mem.recent_queries(int(args.get("limit", 5)), db_path=path,
                                   query=args.get("query"))
    if name == "set_preference":
        return mem.set_preference(str(args["key"]), str(args["value"]), path) or {"ok": True}
    if name == "get_preference":
        return {"key": str(args["key"]),
                "value": mem.get_preference(str(args["key"]),
                                           default=args.get("default"), db_path=path)}
    raise ValueError(f"unknown memory tool: {name}")


class Agent:
    """Agentic loop over one live MCP session + durable SQLite memory."""

    def __init__(self, session, client: LLMClient, db_path: str | Path | None = None,
                 history_limit: int = 20):
        self.session = session
        self.client = client
        self.db_path = db_path
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        self.history_limit = history_limit

    async def ask(self, question: str) -> str:
        """Run one user question through the tool-calling loop."""
        self.messages.append({"role": "user", "content": question})
        tools_called: list[str] = []
        final_answer = ""
        for _ in range(MAX_TOOL_STEPS + 1):
            reply = self.client.complete(self.messages, ALL_TOOLS)
            calls = reply.get("tool_calls") or []
            self.messages.append(
                {
                    "role": "assistant",
                    "content": reply.get("content", ""),
                    "tool_calls": reply.get("tool_calls", []),
                }
            )
            if not calls:
                final_answer = reply.get("content", "").strip()
                break
            for call in calls:
                name = call["name"]
                args = dict(call.get("args") or {})
                tools_called.append(name)
                if name.startswith(MEMORY_TOOL_NAMES):
                    outcome = await run_memory_tool(name, args, self.db_path)
                else:
                    # Drop null values (LLM optional params)
                    args = {k: v for k, v in args.items() if v is not None}
                    result = await self.session.call_tool(name, args)
                    outcome = _extract_payload(result)
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", name),
                        "name": name,
                        "content": json.dumps(outcome, ensure_ascii=False)[:8000],
                    }
                )
        if not final_answer:
            final_answer = "(agent stopped after max tool steps)"
        if self.db_path is not None:
            mem.log_query(question, tools_called, final_answer[:500], self.db_path)
        else:
            mem.log_query(question, tools_called, final_answer[:500])
        # Keep only recent turns so the LLM context stays small.
        self.messages = self.messages[:1] + self.messages[-(self.history_limit * 2):]
        return final_answer


__all__ = [
    "Agent",
    "OpenAICompatibleClient",
    "make_client",
    "run_memory_tool",
    "SYSTEM_PROMPT",
]
