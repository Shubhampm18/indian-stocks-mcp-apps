"""Compatibility shim — the implementation now lives in app_agent.py.

chat.py and client.py import ``Agent`` / ``make_client`` from here; that
import path is unchanged by the LangChain migration. The create_agent loop,
STATIC_PROMPT, middleware and model wiring all live in app_agent.py.
"""

from __future__ import annotations

from app_agent import STATIC_PROMPT, Agent, make_client

__all__ = ["Agent", "make_client", "STATIC_PROMPT"]
