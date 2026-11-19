"""Phase F tests: the guard's repair must not be blinded by payload size.

A 7-tool turn can exceed any excerpt; slicing the payload text used to cut
off exactly the figures the repair needed, producing a false "(unverified)"
flag. The repair input now leads with the COMPLETE number list collected from
the payloads, so only a bounded excerpt follows.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_model import FakeChatModel, tool_call
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from middleware import GroundingGuard
from tools_memory import build_memory_tools


@tool
def get_technical_indicators(symbol: str, as_of: str | None = None) -> str:
    """Fake server tool with a huge filler field BEFORE the target figure."""
    return json.dumps({
        "notes": "x" * 8000,          # pushes the figures below past any 6000-char cutoff
        "current_price": 1570.4,      # the figure the repair needs
        "data_through": "2025-12-31",
    })


def _turn(payload_text: str, draft: str) -> list:
    return [
        HumanMessage(content="q"),
        AIMessage(content="", tool_calls=[
            {"name": "get_technical_indicators", "args": {}, "id": "t1",
             "type": "tool_call"}]),
        ToolMessage(content=payload_text, name="get_technical_indicators",
                    tool_call_id="t1"),
        AIMessage(content=draft),
    ]


def test_repair_succeeds_when_payload_exceeds_excerpt(tmp_path):
    """Tool output > 6000 chars with the target figure beyond the cutoff:
    the repair still finds 1570.4 because the number list is complete."""
    scripted = [
        tool_call("get_technical_indicators", {"symbol": "RELIANCE.NS"}, "t1"),
        AIMessage(content="RELIANCE closed at 1,999.99 (data_through 2025-12-31)."),
    ]
    repair_model = FakeChatModel(scripted=[
        AIMessage(content="RELIANCE closed at Rs 1,570.40 (data_through 2025-12-31)."),
    ])
    guard = GroundingGuard(model=repair_model, db_path=str(tmp_path / "o.db"))
    graph = create_agent(
        model=FakeChatModel(scripted=scripted),
        tools=[get_technical_indicators] + build_memory_tools(None),
        system_prompt="P",
        middleware=[guard],
    )
    result = asyncio.run(graph.ainvoke(
        {"messages": [HumanMessage(content="q")]},
        config={"recursion_limit": 40},
    ))
    answer = result["messages"][-1].content
    assert answer.startswith("RELIANCE closed at Rs 1,570.40"), answer
    assert guard.last_result["grounded"] is True
    assert guard.last_result["action"] == "repaired"
    assert repair_model.i == 1


def test_repair_prompt_carries_complete_number_list_and_stays_bounded():
    """The repair prompt must contain the target figure even though the
    payload excerpt is bounded (and the figure sits past char 6000)."""
    payload_text = json.dumps({"notes": "x" * 8000, "current_price": 1570.4})
    captured: list[str] = []

    class Capture(FakeChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            captured.append(messages[0].content)
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    guard = GroundingGuard(model=Capture(scripted=[
        AIMessage(content="price is Rs 1,570.40")]))
    guard.after_model(
        {"messages": _turn(payload_text, "price is 1,999.99")}, runtime=None)
    assert guard.last_result["action"] == "repaired"
    assert captured, "repair model was never called"
    assert "1570.4" in captured[0]      # complete number list carries the figure
    assert len(captured[0]) < 4000      # yet the repair input stays bounded
