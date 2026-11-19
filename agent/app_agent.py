"""LangChain agent over indian-stocks-mcp: create_agent + MCP tools + memory.

The loop IS ``create_agent`` (LangChain 1.x): one model, one flat tool list,
no sub-agents, no checkpointer. Server tools arrive over MCP via
``load_mcp_tools`` — their schemas are the server's own, never re-written
here. The six memory tools are local ``@tool`` functions (tools_memory.py)
bound to the same SQLite file as memory.py.

The LLM provider is configurable by environment:
  LLM_API_KEY   required (or a provider-specific key such as OPENAI_API_KEY)
  LLM_BASE_URL  OpenAI-compatible base URL (optional)
  LLM_MODEL     model name (optional, default "gpt-4o-mini")
  LLM_TIMEOUT_SECONDS
                per-request timeout in seconds (optional, default 60);
                ChatOpenAI(max_retries=2) retries transient failures first.

Without a key the agent refuses to start — the MCP client demos
(`client.py --demo`) remain the way to exercise the server offline.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import memory as mem
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


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

# Honesty rules only — tool-selection mechanics live in the tools' own
# descriptions now (server docstrings + tools_memory docstrings).
STATIC_PROMPT = """You are a stock research agent for India (NSE/BSE) stocks.
- Never invent numbers: compute every figure from tool output.
- State `data_through` alongside any number you quote.
- Always pass `as_of` (YYYY-MM-DD) when the user gives a date.
- Symbols: bare ticker = NSE guess, use .NS/.BO suffixes when known.
- Chart patterns: the server scans only the last `window` bars (default 20)
  — for pattern questions about older dates, raise `window` until the scan
  covers the dates in question; never call a pattern absent without that.
- Keep answers short: the key number, the rank/trend, then data_through.
"""

# Recursion budget: in langchain 1.5 each middleware hook node
# (before_model / after_model) counts as a step, so one model call costs
# ~3 (hook + model + guard) plus one more per tool round-trip. 12 would
# abort after ~2 tool calls; 30 still bounds the loop hard (~7-8 calls).
RECURSION_LIMIT = 30


def make_client() -> Any:
    """Build the chat model from env vars. Raises if no key is configured."""
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
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model, base_url=base_url, api_key=key,
                      timeout=timeout, max_retries=2)


def _text_of(message: BaseMessage) -> str:
    """AIMessage.content may be str or a list of content blocks; get the text."""
    content = message.content
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, dict):
            parts.append(str(block.get("text", "")))
        else:
            parts.append(str(block))
    return "".join(parts)


class Agent:
    """One LangChain agent over one live MCP session + durable SQLite memory.

    Same construction and ask() signature as before, so chat.py and
    client.py are untouched by the migration.
    """

    def __init__(self, session, client, db_path: str | Path | None = None,
                 history_limit: int = 20, tools: list | None = None):
        self.session = session
        self.client = client  # a BaseChatModel (e.g. ChatOpenAI or a fake)
        self.db_path = db_path
        self.history_limit = history_limit
        # Test seam: extra tools appended to the server's (tests stand in a
        # fake server tool offline; live runs pass None).
        self._extra_tools = list(tools or [])
        # Live conversation (Human/AI/Tool messages); the system prompt is
        # STATIC_PROMPT, applied by the graph at every model call.
        self.messages: list[BaseMessage] = []
        self._graph = None  # built lazily: load_mcp_tools needs an async ctx
        self._guard = None  # GroundingGuard ref (grounding columns for history)

    async def _ensure_graph(self):
        if self._graph is None:
            from langchain.agents import create_agent
            from langchain_mcp_adapters.tools import load_mcp_tools
            from middleware import GroundingGuard, MemoryContext, PreferenceDefaults
            from tools_memory import build_memory_tools

            server_tools = (
                await load_mcp_tools(self.session) if self.session is not None else []
            )
            self._guard = GroundingGuard(model=self.client, db_path=self.db_path)
            self._graph = create_agent(
                model=self.client,
                tools=server_tools + self._extra_tools
                + build_memory_tools(self.db_path),
                system_prompt=STATIC_PROMPT,
                middleware=[
                    MemoryContext(db_path=self.db_path),
                    PreferenceDefaults(db_path=self.db_path),
                    self._guard,
                ],
            )
        return self._graph

    async def ask(self, question: str) -> str:
        """Run one user question through the agent graph."""
        graph = await self._ensure_graph()
        self.messages.append(HumanMessage(content=question))
        input_len = len(self.messages)
        result = await graph.ainvoke(
            {"messages": list(self.messages)},
            config={"recursion_limit": RECURSION_LIMIT},
        )
        all_msgs: list[BaseMessage] = result["messages"]
        new_msgs = all_msgs[input_len:]
        tool_args = [
            call.get("args") or {}
            for m in new_msgs if isinstance(m, AIMessage)
            for call in (m.tool_calls or [])
        ]
        tools_called = [
            call["name"]
            for m in new_msgs if isinstance(m, AIMessage)
            for call in (m.tool_calls or [])
        ]
        answer = next(
            (_text_of(m) for m in reversed(all_msgs) if isinstance(m, AIMessage)),
            "",
        ).strip() or "(agent produced no text)"
        # Keep only recent turns so the model context stays small.
        self.messages = all_msgs[-(self.history_limit * 2):]
        # Grounding columns for history (3.5); None when the guard didn't run.
        as_of_used = ", ".join(
            sorted({str(a["as_of"]) for a in tool_args if a.get("as_of")})
        ) or None
        data_through = self._data_through(new_msgs)
        guard = getattr(self, "_guard", None)
        grounded = guard.last_result["grounded"] if guard and guard.last_result else None
        key_numbers = None
        if guard and guard.last_result:
            claims = guard.last_result.get("claims") or []
            key_numbers = json.dumps([c["token"] for c in claims],
                                     ensure_ascii=False) if claims else None
        self._log(question, tools_called, answer, as_of_used, data_through,
                  key_numbers, grounded)
        return answer

    def _data_through(self, new_msgs: list[BaseMessage]) -> str | None:
        """Pull data_through out of this turn's tool payloads (first hit)."""
        for m in new_msgs:
            if type(m).__name__ == "ToolMessage":
                try:
                    payload = json.loads(_text_of(m))
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
                if isinstance(payload, dict) and payload.get("data_through"):
                    return str(payload["data_through"])
        return None

    def _log(self, question, tools_called, answer, as_of_used, data_through,
             key_numbers, grounded) -> None:
        if self.db_path is not None:
            mem.log_query(question, tools_called, answer[:500], self.db_path,
                          as_of_used=as_of_used, data_through=data_through,
                          key_numbers=key_numbers, grounded=grounded)
        else:
            mem.log_query(question, tools_called, answer[:500],
                          as_of_used=as_of_used, data_through=data_through,
                          key_numbers=key_numbers, grounded=grounded)


__all__ = ["Agent", "STATIC_PROMPT", "make_client"]
