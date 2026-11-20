"""Middleware for the LangChain stock agent: memory context, defaults, grounding.

Three AgentMiddleware subclasses plugged into ``create_agent``:

  MemoryContext        before_model + wrap_model_call — appends a compact
                       durable-memory summary (memory.context_block) to the
                       system message for THIS turn only; the static
                       system prompt itself never changes.
  PreferenceDefaults   wrap_tool_call — fills get_peer_comparison.metric and
                       get_risk_metrics.benchmark from the user's saved
                       preferences when the model omits them, and says so in
                       the tool result ("default applied: ...").
  GroundingGuard       after_model / aafter_model — every quoted figure
                       (decimal point, thousands separator, currency symbol,
                       % or a "6x"-style multiple) must appear in this turn's
                       server-tool output (0.5% relative or 2-decimal
                       equality, sign-insensitive: payloads store drawdowns
                       unsigned while prose adds the sign). One repair call;
                       if it still fails, the answer is prefixed
                       "(unverified: N figures)". A turn that called no
                       price-server tool is checked against its memory
                       payloads instead: figures traceable to the user's own
                       stored record pass (grounded stays NULL — memory is not
                       market data), figures traceable to nothing are flagged
                       outright, since there is no tool output to repair from.

Mixed-turn policy (chosen: STRICT). A turn that calls server tools AND
memory tools is checked against server payloads only — memory payloads never
validate a claim. Memory is context, never authority: a legitimate figure
that only exists in a memory payload costs one repair call, which is the
acceptable price for never passing off recalled context as fresh tool data.
(The permissive alternative — stamping memory-sourced claims source="memory"
— was rejected because it lets the model dress context up as data.)

Matching is unit-blind: "1570.40x" matches a payload 1570.4 numerically, and
"1.57 lakh" matches only a payload that itself writes "1.57 lakh" (string
scan), not one holding 157000.0. Scaling and unit normalisation are out of
scope; the flagged-figures prefix keeps such answers honest.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import memory as mem
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage

# The server's registered tool names (pinned by tests/test_schemas.py).
SERVER_TOOL_NAMES = frozenset({
    "get_ohlcv", "get_returns", "get_technical_indicators",
    "get_chart_patterns", "get_risk_metrics", "get_index_history",
    "get_peer_comparison",
})

# Numbers carrying a marker: thousands separator, decimal point, currency
# prefix, percent suffix or a "6x"-style multiple suffix. Bare integers match
# via the last alternative but are dropped by the marker filter below.
_CLAIM_RE = re.compile(
    r"(?:(?P<cur>Rs\.?|INR|₹|[$€£])\s*)?"
    r"(?P<num>-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+\.\d+|-?\d+)"
    r"(?P<pct>%)?(?P<mult>[xX])?"
)


def extract_claims(text: str) -> list[dict[str, Any]]:
    """Figures the model quoted that need grounding.

    A claim carries a decimal point, a thousands separator, a currency symbol,
    a percent sign or a multiple suffix ("6x"). Years 1900-2100 (bare,
    undecorated) and bare integers 0-9 are ignored. Matching is unit-blind
    (see module docstring).
    """
    claims: list[dict[str, Any]] = []
    for match in _CLAIM_RE.finditer(text):
        token = match.group("num")
        cur = match.group("cur")
        pct = match.group("pct")
        mult = match.group("mult")
        marked = bool(cur or pct or mult or "." in token or "," in token)
        if not marked:
            continue
        value = float(token.replace(",", ""))
        # A year: undecorated whole number 1900-2100.
        if not cur and not pct and not mult and value.is_integer() and 1900 <= value <= 2100:
            continue
        claims.append({"token": match.group(0).strip(), "value": value})
    return claims


def collect_numbers(obj: Any, out: list[float]) -> None:
    """Walk a JSON-ish payload collecting every number (and numbers embedded
    in strings, e.g. '1,340.6' inside a formatted field)."""
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        out.append(float(obj))
        return
    if isinstance(obj, dict):
        for value in obj.values():
            collect_numbers(value, out)
        return
    if isinstance(obj, list):
        for value in obj:
            collect_numbers(value, out)
        return
    if isinstance(obj, str):
        for match in _CLAIM_RE.finditer(obj):
            out.append(float(match.group("num").replace(",", "")))


def _message_text(content: Any) -> str:
    """ToolMessage.content may be str or a list of content blocks; get text."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, dict):
            parts.append(str(block.get("text", "")))
        else:
            parts.append(str(block))
    return "".join(parts)


def _parse_json(content: str) -> Any | None:
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _matches(value: float, numbers: list[float]) -> bool:
    """2-decimal equality, 0.5% relative tolerance, or the same magnitude
    with flipped sign (payloads store drawdowns unsigned while prose adds
    the sign: "fell -19.8%" grounds against drawdown 19.8)."""
    for candidate in numbers:
        if round(value, 2) == round(candidate, 2):
            return True
        if candidate != 0 and abs(value - candidate) / abs(candidate) <= 0.005:
            return True
        if round(abs(value), 2) == round(abs(candidate), 2):
            return True
    return False


async def _await(awaitable):
    return await awaitable


REPAIR_NUMBER_CAP = 200


def _fmt_number(value: float) -> str:
    """Comma-grouped, never scientific notation (volumes reach 1e7)."""
    if float(value).is_integer():
        return f"{value:,.0f}"
    return f"{value:,.6f}".rstrip("0").rstrip(".")


def _repair_number_list(numbers: list[float], unmatched: list[dict[str, Any]],
                        cap: int = REPAIR_NUMBER_CAP) -> str:
    """Candidates for the repair prompt, compact and complete-enough.

    Every distinct payload number while they fit under ``cap``; beyond that,
    keep the ones nearest the claims being repaired (the true value is almost
    always a near-miss of what the model wrote), then show them sorted.
    """
    unique = sorted({round(n, 4) for n in numbers})
    if len(unique) > cap:
        targets = [abs(c["value"]) for c in unmatched] or [0.0]

        def distance(n: float) -> float:
            return min(abs(abs(n) - t) / max(t, 1e-9) for t in targets)

        unique = sorted(sorted(unique, key=distance)[:cap])
    return ", ".join(_fmt_number(n) for n in unique)


def _with_default_note(result, arg_name: str, default: str):
    """Append a visible note so the model can say which default it used."""
    note = f"default applied: {arg_name}={default}"
    if isinstance(result, ToolMessage):
        content = _message_text(result.content)
        return result.model_copy(update={"content": f"{content}\n\n[{note}]"})
    return result  # Command etc. — pass through untouched


class MemoryContext(AgentMiddleware):
    """Show the model its durable memory for this turn only."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = db_path
        self._context = ""

    def before_model(self, state, runtime):
        self._context = mem.context_block(
            self.db_path if self.db_path is not None else mem.DB_PATH
        )
        return None

    def wrap_model_call(self, request, handler):
        return handler(self._inject(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._inject(request))

    def _inject(self, request):
        if self._context:
            base = request.system_message.content if request.system_message else ""
            return request.override(system_message=SystemMessage(
                content=(f"{base}\n\nUser's durable memory (from past sessions):\n"
                         f"{self._context}") if base else self._context
            ))
        return request


class PreferenceDefaults(AgentMiddleware):
    """Fill saved preferences into tool args the model omitted."""

    DEFAULTS = {
        "get_peer_comparison": ("metric", "default_metric"),
        "get_risk_metrics": ("benchmark", "default_benchmark"),
    }

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = db_path

    def wrap_tool_call(self, request, handler):
        return self._apply(request, handler, lambda h, r: h(r))

    async def awrap_tool_call(self, request, handler):
        return await self._apply(request, handler, lambda h, r: _await(h(r)))

    def _apply(self, request, handler, call):
        spec = self.DEFAULTS.get(request.tool_call.get("name", ""))
        if spec is None:
            return call(handler, request)
        arg_name, pref_key = spec
        args = dict(request.tool_call.get("args") or {})
        if args.get(arg_name):
            return call(handler, request)  # model chose explicitly — never override
        default = mem.get_preference(
            pref_key, db_path=self.db_path if self.db_path is not None else mem.DB_PATH
        )
        if not default:
            return call(handler, request)  # nothing saved — leave the server default
        args[arg_name] = default
        request = request.override(tool_call={**request.tool_call, "args": args})
        result = call(handler, request)
        return _with_default_note(result, arg_name, default)


class GroundingGuard(AgentMiddleware):
    """Every quoted figure must come from this turn's server-tool output."""

    def __init__(self, model, db_path: str | Path | None = None):
        self.model = model
        self.db_path = db_path
        self.last_result: dict[str, Any] | None = None

    # -- state inspection ---------------------------------------------------

    def _this_turn_messages(self, messages: list) -> list:
        """Messages after the last HumanMessage (this turn's exchange)."""
        for idx in range(len(messages) - 1, -1, -1):
            if type(messages[idx]).__name__ == "HumanMessage":
                return messages[idx + 1:]
        return list(messages)

    def _server_tool_texts(self, turn_msgs: list) -> list[str]:
        texts = []
        for m in turn_msgs:
            if isinstance(m, ToolMessage) and getattr(m, "name", "") in SERVER_TOOL_NAMES:
                texts.append(_message_text(m.content))
        return texts

    @staticmethod
    def _memory_tool_texts(turn_msgs: list) -> list[str]:
        """Local (non-price-server) payloads — the memory tools today."""
        return [_message_text(m.content) for m in turn_msgs
                if isinstance(m, ToolMessage)
                and getattr(m, "name", "") not in SERVER_TOOL_NAMES]

    @staticmethod
    def _tool_numbers(texts: list[str]) -> list[float]:
        out: list[float] = []
        for text in texts:
            payload = _parse_json(text)
            if payload is not None:
                collect_numbers(payload, out)
            collect_numbers(text, out)  # numbers embedded in strings too
        return out

    # -- guard --------------------------------------------------------------

    def _analyze(self, messages: list, last: AIMessage) -> dict[str, Any] | None:
        """Shared analysis for the sync and async paths.

        Sets ``last_result`` and returns None when no repair is needed
        (skipped turn / already grounded); otherwise returns the repair plan,
        or — for a figure quoted with no tool call at all — a direct
        ``replacement`` (nothing exists to repair against).
        """
        turn_msgs = self._this_turn_messages(messages)
        texts = self._server_tool_texts(turn_msgs)
        if not texts:
            # No price-server payload this turn. A recall answer's source IS the
            # user's own stored record, so memory payloads may validate its
            # figures — but a figure that appears in neither memory nor the
            # server was invented, and nothing exists to repair it from.
            memory_texts = self._memory_tool_texts(turn_msgs)
            draft = _message_text(last.content)
            claims = extract_claims(draft)
            numbers = self._tool_numbers(memory_texts) if memory_texts else []
            unmatched = [c for c in claims if not _matches(c["value"], numbers)]
            if not unmatched:
                # grounded stays None: these are memory figures, not price data.
                self.last_result = {"grounded": None, "claims": claims,
                                    "action": "pass" if claims else None}
                return None
            self.last_result = {"grounded": False, "claims": claims,
                                "unmatched": unmatched, "action": "flagged"}
            return {"last_id": last.id, "numbers": numbers, "repair_input": None,
                    "replacement":
                        f"(unverified: {len(unmatched)} figures) "
                        f"{draft}".strip()}
        draft = _message_text(last.content)
        claims = extract_claims(draft)
        numbers = self._tool_numbers(texts)
        unmatched = [c for c in claims if not _matches(c["value"], numbers)]
        if not unmatched:
            self.last_result = {"grounded": True, "claims": claims,
                                "action": "pass"}
            return None
        # ONE repair call.
        listing = ", ".join(c["token"] for c in unmatched)
        repair_prompt = (
            f"These figures are not in the tool output: {listing}. "
            "Rewrite using only tool numbers, or say you cannot."
        )
        # A 7-tool turn can exceed any excerpt, and slicing the payload text
        # can cut off exactly the figures the repair needs -> a false
        # "(unverified)" flag. Lead with the numbers collected from the
        # payloads, kept compact: a 400-row payload has thousands of them, so
        # beyond the cap we keep the ones closest to the claims being repaired.
        number_list = _repair_number_list(numbers, unmatched)
        repair_input = (
            f"{repair_prompt}\n\nNumbers present in the tool output:\n"
            f"{number_list}\n\nTool output (excerpt):\n"
            + "\n---\n".join(texts)[:1200]
            + f"\n\nYour draft answer:\n{draft}"
        )
        return {"last_id": last.id, "numbers": numbers, "repair_input": repair_input}

    def _finish(self, plan: dict[str, Any], repaired_text: str) -> dict[str, Any]:
        """Shared repair verdict for the sync and async paths."""
        claims2 = extract_claims(repaired_text)
        unmatched2 = [c for c in claims2 if not _matches(c["value"], plan["numbers"])]
        if not unmatched2:
            self.last_result = {"grounded": True, "claims": claims2,
                                "repaired": True, "action": "repaired"}
            replacement = repaired_text
        else:
            self.last_result = {"grounded": False, "claims": claims2,
                                "unmatched": unmatched2, "action": "flagged"}
            replacement = f"(unverified: {len(unmatched2)} figures) {repaired_text}".strip()
        return {"messages": [RemoveMessage(id=plan["last_id"]),
                             AIMessage(content=replacement)]}

    def after_model(self, state, runtime):
        messages = state["messages"]
        last = messages[-1] if messages else None
        self.last_result = None
        if not isinstance(last, AIMessage) or last.tool_calls:
            return None  # tool calls route to the tool node, not the answer
        plan = self._analyze(messages, last)
        if plan is None:
            return None
        if plan.get("replacement") is not None:  # flagged without repair
            return {"messages": [RemoveMessage(id=plan["last_id"]),
                                 AIMessage(content=plan["replacement"])]}
        repaired = self.model.invoke([HumanMessage(content=plan["repair_input"])])
        repaired_text = _message_text(repaired.content) if isinstance(repaired, AIMessage) \
            else str(repaired)
        return self._finish(plan, repaired_text)

    async def aafter_model(self, state, runtime):
        """Async twin of after_model — the repair call must not block the
        event loop on every ungrounded answer. Same logic, shared helpers."""
        messages = state["messages"]
        last = messages[-1] if messages else None
        self.last_result = None
        if not isinstance(last, AIMessage) or last.tool_calls:
            return None  # tool calls route to the tool node, not the answer
        plan = self._analyze(messages, last)
        if plan is None:
            return None
        if plan.get("replacement") is not None:  # flagged without repair
            return {"messages": [RemoveMessage(id=plan["last_id"]),
                                 AIMessage(content=plan["replacement"])]}
        repaired = await self.model.ainvoke([HumanMessage(content=plan["repair_input"])])
        repaired_text = _message_text(repaired.content) if isinstance(repaired, AIMessage) \
            else str(repaired)
        return self._finish(plan, repaired_text)


__all__ = [
    "MemoryContext",
    "PreferenceDefaults",
    "GroundingGuard",
    "SERVER_TOOL_NAMES",
    "extract_claims",
    "collect_numbers",
]
