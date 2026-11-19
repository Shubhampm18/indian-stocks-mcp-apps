"""Shared helpers for the multi-page dashboard.

Every page imports this module to get: the data-layer shim, the shared
sidebar (symbol / as_of / lookback — session_state-backed so choices
persist across page switches), the guarded fetch, and the header metrics.

GUARANTEE: no row with date > as_of is ever returned from get_data() —
the explicit check below mirrors the data layer's enforce_as_of and
cannot be optimised away (unlike a bare assert under python -O).
"""

from __future__ import annotations

import os
import sys
from datetime import date, timedelta
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


def render_sidebar_common() -> None:
    """Shared sidebar inputs, backed by session_state so they persist
    across page switches."""
    with st.sidebar:
        st.text_input("Symbol", key="symbol_raw", value="RELIANCE.NS")
        st.date_input("as_of", key="as_of", value=IST_TODAY - timedelta(days=60),
                      max_value=IST_TODAY)
        st.slider("Lookback (days)", 10, 400, 120, key="lookback")
        st.caption(METHODOLOGY)


def render_header(symbol: str, as_of: date, data_through) -> None:
    c = st.columns(5)
    c[0].metric("Symbol", symbol)
    c[1].metric("as_of", str(as_of))
    c[2].metric("data_through (≤ as_of)", str(data_through))
    c[3].metric("Source", D.SOURCE)
    c[4].metric("Currency", "INR")


def get_data() -> tuple | None:
    """Validate sidebar state, fetch data, enforce the as_of guarantee.

    Returns (symbol, as_of, lookback, df, df_full, close_adj, current_price)
    or None after rendering an error (the page should then st.stop()).
    """
    symbol_raw = st.session_state.get("symbol_raw", "")
    as_of = st.session_state.get("as_of")
    lookback = st.session_state.get("lookback", 120)

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
        df_full, *_ = D._get_frame(symbol, as_of_str, 100000)
        df, d_cut, _hist, _actions = D._get_frame(symbol, as_of_str, lookback)
    except (D.UnknownSymbolError, D.FutureDateError, ValueError) as exc:
        st.error(str(exc))
        return None

    if df.empty:
        st.error("No data for this symbol/as_of")
        return None

    # ---- guarantee: nothing past as_of may be drawn ----
    if df.index.max().date() > as_of:
        st.error("as_of guarantee violated: future row")
        return None

    data_through = df_full.index.max().date()
    render_header(symbol, as_of, data_through)
    close_adj = df["CloseAdj"].astype(float)
    current_price = round(float(close_adj.iloc[-1]), 4)
    return symbol, as_of, lookback, df, df_full, close_adj, current_price
