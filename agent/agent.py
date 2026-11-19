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
    """Server-tool payloads arrive as JSON text (FastMCP content[0]);
    parse, falling back to a raw envelope for non-JSON output."""
    if isinstance(result, str):
        text = result
    else:
        text = getattr(result.content[0], "text", "") if getattr(result, "content", None) else ""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {"raw_text": text, "isError": False}


class Agent:
    """Agentic loop over one live MCP session + durable SQLite memory.

    Tool schemas come from the server itself (``load_mcp_tools``) plus the
    local memory tools — nothing hand-written here, so a server-side tool
    addition can no longer drift from what the model sees.
    """

    def __init__(self, session, client: LLMClient, db_path: str | Path | None = None,
                 history_limit: int = 20):
        self.session = session
        self.client = client
        self.db_path = db_path
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        self.history_limit = history_limit
        self._tools: list[Any] | None = None  # loaded lazily (needs an async ctx)
        self._schemas: list[dict[str, Any]] | None = None

    async def _ensure_tools(self) -> list[Any]:
        """Load server tools over MCP (once) and bind the memory tools."""
        if self._tools is None:
            from langchain_core.utils.function_calling import convert_to_openai_tool
            from langchain_mcp_adapters.tools import load_mcp_tools
            from tools_memory import build_memory_tools

            server_tools = await load_mcp_tools(self.session) if self.session is not None else []
            self._tools = server_tools + build_memory_tools(self.db_path)
            self._schemas = [convert_to_openai_tool(t) for t in self._tools]
        return self._tools

    async def ask(self, question: str) -> str:
        """Run one user question through the tool-calling loop."""
        tools = await self._ensure_tools()
        by_name = {t.name: t for t in tools}
        self.messages.append({"role": "user", "content": question})
        tools_called: list[str] = []
        final_answer = ""
        for _ in range(MAX_TOOL_STEPS + 1):
            reply = self.client.complete(self.messages, self._schemas)
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
                # Drop null values (LLM optional params)
                args = {k: v for k, v in (call.get("args") or {}).items() if v is not None}
                tools_called.append(name)
                try:
                    raw = await by_name[name].ainvoke(args)
                    outcome = _extract_payload(raw) if isinstance(raw, str) else raw
                except KeyError:
                    outcome = {"isError": True, "error": f"unknown tool: {name}"}
                except Exception as exc:
                    outcome = {"isError": True, "error": str(exc)}
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
    "SYSTEM_PROMPT",
]
