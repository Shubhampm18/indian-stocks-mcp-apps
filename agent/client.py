"""MCP stdio client for indian-stocks-mcp (demo mode + agentic mode).

Modes:
  --demo   Three hardcoded end-to-end demos (peer comparison, technical
           indicators, future-date error path). No LLM needed.
  (default) Agentic mode: free-text questions answered by an LLM agent
           (agent.py) over live tool calls, with durable SQLite memory.
           Requires LLM_API_KEY / OPENAI_API_KEY.

The server's stdout is the MCP JSON-RPC channel: this client never prints
to it. All prints below go to the client's own stdout/stderr.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def server_params() -> StdioServerParameters:
    """Resolve how to launch the server.

    Order:
      1. ``STOCK_ANALYST_MCP_DIR`` -> a local checkout (uv run --project);
      2. the installed package -> ``python -m stock_analyst_mcp.server``
         (this is the normal path: indian-stocks-mcp is a dev dependency);
      3. ``uvx indian-stocks-mcp`` -> the published distribution.
    """
    env_dir = os.environ.get("STOCK_ANALYST_MCP_DIR")
    if env_dir:
        return StdioServerParameters(
            command="uv",
            args=["run", "--project", env_dir, "indian-stocks-mcp"],
        )
    if importlib.util.find_spec("stock_analyst_mcp") is not None:
        return StdioServerParameters(
            command=sys.executable, args=["-m", "stock_analyst_mcp.server"]
        )
    return StdioServerParameters(command="uvx", args=["indian-stocks-mcp"])


EXPECTED_TOOLS = {
    "get_ohlcv",
    "get_returns",
    "get_technical_indicators",
    "get_risk_metrics",
    "get_index_history",
    "get_peer_comparison",
}


def _payload(result) -> dict[str, Any]:
    """Extract the tool's dict payload from a CallToolResult.

    FastMCP returns the tool's dict as JSON text in content[0]. The
    server also signals errors with {"isError": True, ...} *inside* the
    payload, and `result.isError` is True only for protocol-level errors.
    """
    text = getattr(result.content[0], "text", "") if result.content else ""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {"raw_text": text, "isError": bool(getattr(result, "isError", False))}


def _show(title: str, data: dict[str, Any]) -> None:
    print(f"\n{'=' * 72}")
    print(title)
    print("=" * 72)
    print(json.dumps(data, indent=2, ensure_ascii=False))


async def _run_demos(session) -> int:
    # ---- demo 1: peer comparison ----------------------------------------
    r1 = await session.call_tool(
        "get_peer_comparison",
        {
            "symbols": ["RELIANCE.NS", "TCS.NS"],
            "as_of": "2025-12-31",
            "metric": "return_20d",
        },
    )
    _show("Demo 1 — get_peer_comparison(RELIANCE.NS, TCS.NS, return_20d)", _payload(r1))

    # ---- demo 2: technical indicators ------------------------------------
    r2 = await session.call_tool(
        "get_technical_indicators",
        {"symbol": "RELIANCE.NS", "as_of": "2025-12-31"},
    )
    d2 = _payload(r2)
    _show(
        "Demo 2 — get_technical_indicators(RELIANCE.NS)",
        {
            "current_price": d2.get("current_price"),
            "trend": d2.get("trend"),
            "indicators": d2.get("indicators"),
            "source": d2.get("source"),
            "as_of": d2.get("as_of"),
            "data_through": d2.get("data_through"),
            "currency": d2.get("currency"),
            "isError": d2.get("isError", False),
        },
    )

    # ---- demo 3: future-date error path ----------------------------------
    r3 = await session.call_tool(
        "get_ohlcv",
        {"symbol": "RELIANCE.NS", "as_of": "2999-01-01"},
    )
    d3 = _payload(r3)
    _show("Demo 3 — get_ohlcv(RELIANCE.NS, as_of=2999-01-01) [future-date error path]", d3)
    if not d3.get("isError"):
        print("Demo 3 FAILED: expected isError=true for a future as_of.", file=sys.stderr)
        return 1
    print("\nDemo 3 OK: server returned isError=true — the as_of guarantee holds.")
    return 0


async def main() -> int:
    ap = argparse.ArgumentParser(description="indian-stocks-mcp client")
    ap.add_argument("--demo", action="store_true",
                    help="Run the 3 hardcoded demos (no LLM needed).")
    ap.add_argument("question", nargs="?",
                    help="One free-text question for the agentic mode.")
    args = ap.parse_args()

    params = server_params()
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # ---- list tools ------------------------------------------------
            tools = await session.list_tools()
            names = [t.name for t in tools.tools]
            print(f"Server: indian-stocks-mcp — {len(names)} tools listed:")
            for t in tools.tools:
                print(f"  - {t.name}")
            missing = EXPECTED_TOOLS - set(names)
            if missing:
                print(f"Missing expected tools: {sorted(missing)}", file=sys.stderr)
                return 1

            if args.demo:
                return await _run_demos(session)

            # ---- agentic mode ----------------------------------------------
            from agent import Agent, make_client

            try:
                llm_client = make_client()
            except RuntimeError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return 1
            agent = Agent(session, llm_client)
            question = args.question or input("question: ").strip()
            if not question:
                print("No question given.", file=sys.stderr)
                return 1
            print(f"\nQ: {question}")
            print(f"A: {await agent.ask(question)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(__import__("anyio").run(main))
