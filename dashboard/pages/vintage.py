"""Vintage page: why as_of matters.

Same symbol, two rebuild dates. The adjusted close for a historical day
is not a fact about that day: it is that day's close multiplied by every
corporate action known AT the rebuild date. A dividend paid later re-bases
the whole series — so the same date reads differently depending on when
you ask. This page overlays both vintages through the same cached fetch
and quantifies the drift.

Mirror of the server's demo/vintage_drift.py logic, expressed through the
dashboard's cached frames (load_full_frame per as_of), not re-derived.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common
from stock_analyst_mcp import data as D  # noqa: E402  (path shim in common)

data = common.get_data()
if data is None:
    st.stop()
symbol, as_of, lookback, df, df_full, close_adj, current_price = data

st.subheader(f"Vintage drift — {symbol}")

cmp_as_of = st.date_input(
    "Comparison as_of (the 'now' vintage)",
    key="vintage_cmp_as_of",
    value=common.IST_TODAY,
    min_value=as_of,
    max_value=common.IST_TODAY,
)

if not isinstance(cmp_as_of, date) or cmp_as_of > common.IST_TODAY:
    st.error("Comparison date must not be in the future.")
    st.stop()
if as_of > common.IST_TODAY:
    st.error("as_of must not be in the future.")
    st.stop()

try:
    v_then_full = common.load_full_frame(symbol, as_of.isoformat())
    v_now_full = common.load_full_frame(symbol, cmp_as_of.isoformat())
except (D.UnknownSymbolError, D.FutureDateError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

# ---- guarantee: each vintage is bounded by its own as_of ----
if not common.within_as_of(v_then_full, as_of) or \
        not common.within_as_of(v_now_full, cmp_as_of):
    st.error("as_of guarantee violated: future row")
    st.stop()

if v_then_full.empty or v_now_full.empty:
    st.error("No data for this symbol/as_of")
    st.stop()

v_then = v_then_full["CloseAdj"].astype(float)
v_now = v_now_full["CloseAdj"].astype(float)

joined = common.vintage_delta(v_then, v_now)
if joined.empty:
    st.info("The two vintages share no common dates — nothing to compare.")
    st.stop()

latest = joined.index[-1]
drift = float(joined["delta_pct"].iloc[-1])
hc = st.columns(4)
hc[0].metric("latest common date", str(latest.date()))
hc[1].metric("as known then", f"{float(joined['then'].iloc[-1]):,.2f}")
hc[2].metric("as adjusted now", f"{float(joined['now'].iloc[-1]):,.2f}")
hc[3].metric("drift at latest common date", f"{drift:+.2f}%")

fig = go.Figure()
fig.add_trace(go.Scatter(x=joined.index, y=joined["then"], mode="lines",
                         name=f"as_of {as_of} (then)"))
fig.add_trace(go.Scatter(x=joined.index, y=joined["now"], mode="lines",
                         name=f"as_of {cmp_as_of} (now)"))
fig.update_layout(
    title=f"{symbol} CloseAdj — same dates, two rebuild dates",
    xaxis_title="date", yaxis_title="INR",
)
st.plotly_chart(fig, use_container_width=True)

recent = joined.tail(10).copy()
recent.insert(0, "date", [str(i.date()) for i in recent.index])
st.dataframe(recent[["date", "then", "now", "delta_pct"]], hide_index=True)

st.caption(
    "Adjusted closes are rebuilt from corporate actions known as of that "
    "date, so later dividends re-base the whole series (factor method; "
    "splits are already applied by the source and recorded, not re-applied)."
)
