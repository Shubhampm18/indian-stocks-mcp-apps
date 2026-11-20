"""Shared helpers for the 4-page dashboard (Overview / Patterns / Peers /
Vintage).

Every page imports this module to get: the data-layer shim, the shared
sidebar (symbol / as_of / lookback — seeded from st.query_params so a
?symbol=...&as_of=...&lookback=... URL is a linkable claim, session_state
so choices persist across page switches), ONE cached full-frame fetch,
in-memory lookback slicing, the cached pattern detector, the header
metrics, and the guarded data assembly.

GUARANTEE: no row with date > as_of is ever returned from get_data() /
vintage frames — the explicit check below mirrors the data layer's
enforce_as_of and cannot be optimised away (unlike a bare assert under
python -O).
"""

from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from typing import Mapping
from pathlib import Path

import pandas as pd
import streamlit as st


# The dashboard consumes the published package (see pyproject dependencies).
# STOCK_ANALYST_MCP_SRC is honoured only as an explicit opt-in override for
# pointing at an uninstalled checkout while developing the server.
_src_override = os.environ.get("STOCK_ANALYST_MCP_SRC")
if _src_override:
    sys.path.insert(0, _src_override)

from stock_analyst_mcp import data as D  # noqa: E402

# The dashboard depends on a PRIVATE server function (_get_frame). Pin it
# loudly: dashboard/tests assert hasattr(D, "_get_frame"), so a server
# refactor that renames it fails in CI, not at runtime on someone's screen.
PRIVATE_API_USED = ("dashboard calls stock_analyst_mcp.data._get_frame "
                    "directly (no MCP hop); pinned by tests/test_common.py")

IST_TODAY = pd.Timestamp.now(tz="Asia/Kolkata").date()

ALL_IND = ["SMA_20", "SMA_50", "EMA_12", "EMA_26", "RSI_14",
           "MACD", "BOLLINGER", "ATR_14", "VWAP"]

METHODOLOGY = (
    f"Adjusted prices rebuilt from corporate actions known as of the chosen "
    f"as_of (factor method: 1 - div/close_before_ex for dividends, with the "
    f"as-filed cash rebased across later splits). Splits are already applied "
    f"by the source and are recorded, not re-applied. Cache TTL 24h "
    f"(pyarrow). yfinance research use only."
)

_DEFAULT_SYMBOL = "RELIANCE.NS"
_DEFAULT_LOOKBACK = 120
_LOOKBACK_MIN, _LOOKBACK_MAX = 10, 400


# ---------------------------------------------------------------------------
# One cached fetch per (symbol, as_of)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def load_full_frame(symbol: str, as_of: str) -> pd.DataFrame:
    """The untrimmed as_of frame (lookback 100000), cached for one hour.

    Calls the PRIVATE data-layer function D._get_frame exactly once per
    (symbol, as_of); every view (lookback slice, pattern window) slices the
    cached result in memory. attrs["rows_used"] — the pre-trim row count at
    the cutoff — is preserved by _get_frame and by our tail/copy slicing.
    """
    df, _d, _history, _actions = D._get_frame(symbol, as_of, 100000)
    return df


def slice_lookback(full: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """In-memory tail slice — never mutates the cached frame."""
    return full.tail(int(lookback)).copy()


# Alias kept for the historical name used across the pages.
frame_for = slice_lookback


def within_as_of(df: pd.DataFrame, as_of: date) -> bool:
    """The point-in-time guarantee, as a checkable predicate (never a bare
    assert): no row in the frame may be dated after as_of."""
    return df.index.max().date() <= as_of


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested offline in dashboard/tests)
# ---------------------------------------------------------------------------

def parse_view_params(params: Mapping[str, str]) -> dict:
    """Validate/normalise the shareable view parameters.

    Bad symbol -> default; bad/future date -> IST_TODAY - 60 days;
    bad lookback -> default, then clamp to 10..400. Pure: takes a mapping
    of strings (st.query_params is one), returns plain values.
    """
    symbol = str(params.get("symbol", "")).strip() or _DEFAULT_SYMBOL

    raw_date = str(params.get("as_of", "")).strip()
    try:
        as_of = date.fromisoformat(raw_date)
    except ValueError:
        as_of = None
    if as_of is None or as_of > IST_TODAY:
        as_of = IST_TODAY - timedelta(days=60)

    try:
        lookback = int(str(params.get("lookback", "")).strip()
                       or _DEFAULT_LOOKBACK)
    except ValueError:
        lookback = _DEFAULT_LOOKBACK
    lookback = max(_LOOKBACK_MIN, min(_LOOKBACK_MAX, lookback))

    return {"symbol": symbol, "as_of": as_of, "lookback": lookback}


def vintage_delta(v_then: pd.Series, v_now: pd.Series) -> pd.DataFrame:
    """Per-date vintage drift on the COMMON dates of two CloseAdj vintages.

    Non-overlapping indexes yield an empty frame, never an exception.
    Columns: then / now / delta_pct = (now - then) / then * 100.
    """
    common = v_then.index.intersection(v_now.index)
    if len(common) == 0:
        return pd.DataFrame(columns=["then", "now", "delta_pct"])
    joined = pd.DataFrame({
        "then": v_then.reindex(common).astype(float),
        "now": v_now.reindex(common).astype(float),
    })
    joined["delta_pct"] = (joined["now"] - joined["then"]) / joined["then"] * 100
    return joined


def rank_peers(results: list[dict], metric: str) -> list[dict]:
    """Re-rank raw peer results by the chosen metric, best first.

    Rows without a numeric value sink to the bottom (sorted last, flag
    value=None). Pure: the server's 'ranked' field is trusted, but this
    makes the ordering explicit and testable.
    """
    def key(row: dict):
        val = row.get("value")
        return (val is None, -(val if isinstance(val, (int, float)) else 0))

    ranked = sorted(
        (r for r in results if isinstance(r, dict)),
        key=key,
    )
    return [{"symbol": r.get("symbol"), "metric": metric, "value": r.get("value")}
            for r in ranked]


# ---------------------------------------------------------------------------
# Query-param state (seeding + write-back)
# ---------------------------------------------------------------------------

def seed_state_from_query() -> None:
    """Seed session_state from st.query_params once per distinct URL.

    A widget edit updates session_state but not the URL signature, so it is
    not re-seeded; a new URL (pasted link / browser back) re-seeds. Bad
    values fall back to the defaults in parse_view_params.
    """
    params = {k: str(st.query_params.get(k, "")) for k in
              ("symbol", "as_of", "lookback")}
    sig = "|".join(f"{k}={params[k]}" for k in params)
    keys = ("symbol_raw", "as_of", "lookback")
    # Re-seed when the URL changed OR the widget-backed state is missing
    # (widget state can be cleared on page switches / fresh sessions).
    if sig == st.session_state.get("_qp_sig") and \
            all(k in st.session_state for k in keys):
        return
    parsed = parse_view_params(params)
    st.session_state["symbol_raw"] = parsed["symbol"]
    st.session_state["as_of"] = parsed["as_of"]
    st.session_state["lookback"] = parsed["lookback"]
    st.session_state["_qp_sig"] = sig


def sync_state_to_query() -> None:
    """Write the current view state back to st.query_params so the URL is
    a linkable claim of what is on screen."""
    as_of = st.session_state.get("as_of")
    target = {
        "symbol": str(st.session_state.get("symbol_raw", "")).strip(),
        "as_of": as_of.isoformat() if isinstance(as_of, date) else "",
        "lookback": str(int(st.session_state.get("lookback", _DEFAULT_LOOKBACK)
                            or _DEFAULT_LOOKBACK)),
    }
    current = {k: str(st.query_params.get(k, "")) for k in target}
    if current != target:
        st.query_params.from_dict(target)
        st.session_state["_qp_sig"] = "|".join(
            f"{k}={target[k]}" for k in target)


# ---------------------------------------------------------------------------
# Sidebar / header / cached pattern work
# ---------------------------------------------------------------------------

def render_sidebar_common() -> None:
    """Shared sidebar inputs, backed by session_state so they persist
    across page switches; state is seeded from / written to query params."""
    seed_state_from_query()
    with st.sidebar:
        st.text_input("Symbol", key="symbol_raw")
        st.date_input("as_of", key="as_of", max_value=IST_TODAY)
        st.slider("Lookback (days)", 10, 400, key="lookback")
        st.caption(METHODOLOGY)
    sync_state_to_query()


def render_header(symbol: str, as_of: date, data_through) -> None:
    c = st.columns(5)
    c[0].metric("Symbol", symbol)
    c[1].metric("as_of", str(as_of))
    c[2].metric("data_through (≤ as_of)", str(data_through))
    c[3].metric("Source", D.SOURCE)
    c[4].metric("Currency", "INR")


@st.cache_data(show_spinner=False)
def patterns_for(symbol: str, as_of: str, lookback: int, window: int) -> dict:
    """Cached chart-pattern scan for one view; never raises.

    The detector runs on the in-memory slice of the cached frame, so it
    adds no fetch. On a detector hiccup returns {"error": ...} with empty
    patterns/support_resistance so pages stay renderable.
    """
    from stock_analyst_mcp import technicals

    df = slice_lookback(load_full_frame(symbol, as_of), lookback)
    try:
        res = technicals.compute_chart_patterns(df, window=window)
    except Exception as exc:  # keep the page renderable on detector hiccups
        return {"patterns": [], "support_resistance": {}, "error": str(exc)}
    res.setdefault("patterns", [])
    res.setdefault("support_resistance", {})
    return res


def get_data() -> tuple | None:
    """Validate sidebar state, fetch data, enforce the as_of guarantee.

    Returns (symbol, as_of, lookback, df, df_full, close_adj, current_price)
    or None after rendering an error (the page should then st.stop()).
    The full frame comes from the one cached fetch; the view frame is an
    in-memory slice of it.
    """
    symbol_raw = st.session_state.get("symbol_raw", "")
    as_of = st.session_state.get("as_of")
    lookback = int(st.session_state.get("lookback", _DEFAULT_LOOKBACK)
                   or _DEFAULT_LOOKBACK)

    if isinstance(as_of, date) and as_of > IST_TODAY:
        st.error("as_of must not be in the future.")
        return None
    if not symbol_raw.strip():
        st.error("Enter a symbol")
        return None

    symbol = D.normalise_symbol(symbol_raw)
    as_of_str = as_of.isoformat() if isinstance(as_of, date) else str(as_of)
    try:
        # Untrimmed frame for the true cutoff (lookback only affects the view).
        df_full = load_full_frame(symbol, as_of_str)
    except (D.UnknownSymbolError, D.FutureDateError, ValueError) as exc:
        st.error(str(exc))
        return None

    if df_full.empty:
        st.error("No data for this symbol/as_of")
        return None

    df = slice_lookback(df_full, lookback)

    # ---- guarantee: nothing past as_of may be drawn ----
    if not within_as_of(df, as_of):
        st.error("as_of guarantee violated: future row")
        return None

    data_through = df_full.index.max().date()
    render_header(symbol, as_of, data_through)
    close_adj = df["CloseAdj"].astype(float)
    current_price = round(float(close_adj.iloc[-1]), 4)
    return symbol, as_of, lookback, df, df_full, close_adj, current_price
