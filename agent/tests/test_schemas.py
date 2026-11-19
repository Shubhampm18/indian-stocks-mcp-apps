"""Schema-contract test: the server's registered tool names must equal the
seven names the agent expects — catches drift at pytest time, not at runtime.

Uses the installed stock_analyst_mcp package (dev dependency) and its in-process
FastMCP instance: no subprocess, no network. No pytest-asyncio dependency —
asyncio.run drives the coroutine.
"""

from __future__ import annotations

import asyncio

import pytest

# The server package is optional to install (dev group); skip cleanly without it.
mcp_server = pytest.importorskip("stock_analyst_mcp.server")

EXPECTED_TOOL_NAMES = [
    "get_ohlcv",
    "get_returns",
    "get_technical_indicators",
    "get_chart_patterns",
    "get_risk_metrics",
    "get_index_history",
    "get_peer_comparison",
]


def test_server_registers_exactly_the_seven_expected_tools():
    registered = asyncio.run(mcp_server.mcp.list_tools())
    names = [t.name for t in registered]
    assert names == EXPECTED_TOOL_NAMES, (
        f"server tool drift: registered={names}, agent expects={EXPECTED_TOOL_NAMES}"
    )


def test_every_tool_name_list_agrees_with_the_server():
    """Three lists pin the server's tools: this test, the demo client, and the
    grounding guard's SERVER_TOOL_NAMES. If one falls behind, new tools escape
    the guard silently (their payloads are never scanned), so drift must fail
    here rather than in production."""
    import client
    import middleware

    assert set(client.EXPECTED_TOOLS) == set(EXPECTED_TOOL_NAMES), (
        "client.EXPECTED_TOOLS drifted from the server"
    )
    assert set(middleware.SERVER_TOOL_NAMES) == set(EXPECTED_TOOL_NAMES), (
        "middleware.SERVER_TOOL_NAMES drifted from the server: those tools' "
        "payloads would not be checked by the grounding guard"
    )
