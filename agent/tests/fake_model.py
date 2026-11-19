"""Scripted BaseChatModel for offline agent tests — no network, no API key.

Yields a fixed list of AIMessages, one per model call, repeating the last one
when the script runs out (so loops can never crash the test). Tool calls are
expressed as AIMessage.tool_calls; ``bind_tools`` is a no-op returning self,
because create_agent binds the tool list and a scripted model ignores it.
"""

from __future__ import annotations

from typing import Any, List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class FakeChatModel(BaseChatModel):
    """Yield scripted AIMessages, one per model call."""

    scripted: List[BaseMessage] = []
    repeat_last: bool = True
    i: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _next_message(self) -> AIMessage:
        """The next scripted message (shared by the sync and async paths)."""
        if not self.scripted:
            return AIMessage(content="")
        if self.i < len(self.scripted):
            msg = self.scripted[self.i]
        elif self.repeat_last:
            msg = self.scripted[-1]
        else:  # pragma: no cover - defensive
            raise RuntimeError(f"script exhausted after {self.i} calls")
        self.i += 1
        return msg

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next_message())])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next_message())])


class AsyncOnlyFakeChatModel(FakeChatModel):
    """Scripted model exposing only the async path.

    ``_generate`` raises, so any sync ``.invoke()`` through this model fails
    loudly — used to prove the guard's repair call went through
    ``ainvoke`` (A3) instead of blocking the event loop.
    """

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        raise NotImplementedError("async-only model: the sync path must not be used")


def tool_call(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    """One AIMessage carrying a single tool call."""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )
