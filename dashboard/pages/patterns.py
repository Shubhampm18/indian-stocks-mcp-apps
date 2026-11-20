"""Patterns page: candlestick-pattern workbench.

A window control (SCAN DEPTH) drives the cached detector directly: the
scan inspects only the last `window` bars, so a pattern older than the
window is invisible by construction — the on-screen state (table, empty
message, markers, data_through / rows_used caption) makes that limit
visible instead of hidden in a default.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common

data = common.get_data()
if data is None:
    st.stop()
symbol, as_of, lookback, df, df_full, close_adj, current_price = data
as_of_str = as_of.isoformat() if isinstance(as_of, date) else str(as_of)

st.subheader("Chart Patterns")
window = st.slider("SCAN DEPTH (bars scanned for patterns)", 5, 250, 20,
                   key="pattern_window")
st.caption(
    f"The detector looks only at the last {window} bars — a pattern older "
    "than this is invisible by construction."
)

pres = common.patterns_for(symbol, as_of_str, lookback, window)
if pres.get("error"):
    st.error(pres["error"])
pattern_list = pres.get("patterns", [])
support_resistance = pres.get("support_resistance", {})

if len(df) < window:
    st.info(f"insufficient data (patterns scan the last {window} bars, "
            f"this view has {len(df)})")
elif pattern_list:
    st.dataframe(pd.DataFrame([
        {"date": p.get("date"), "name": p.get("name"),
         "confidence": p.get("confidence")}
        for p in pattern_list
    ]), hide_index=True)
else:
    st.info(f"No patterns in the last {window} bars")

pc = st.columns(2)
if len(df) < window:
    pc[0].metric("nearest_support", "insufficient data")
    pc[1].metric("nearest_resistance", "insufficient data")
else:
    pc[0].metric("nearest_support",
                 support_resistance.get("nearest_support") or "none below price")
    pc[1].metric("nearest_resistance",
                 support_resistance.get("nearest_resistance") or "none above price")

# ---- chart: markers + S/R so the scan window is visible on screen ----
fig = go.Figure()
fig.add_trace(
    go.Candlestick(
        x=df.index,
        open=df["Open"].astype(float),
        high=df["High"].astype(float),
        low=df["Low"].astype(float),
        close=df["CloseAdj"].astype(float),
        name="OHLC (CloseAdj)",
    )
)
if pattern_list:
    marked = [p for p in pattern_list
              if p.get("date") in {str(idx.date()) for idx in df.index}]
    if marked:
        fig.add_scatter(
            x=[p["date"] for p in marked],
            y=[float(df.loc[df.index.strftime("%Y-%m-%d") == p["date"], "Low"].iloc[0])
               for p in marked],
            mode="markers+text",
            text=[p["name"] for p in marked],
            textposition="bottom center",
            marker={"size": 10, "symbol": "diamond"},
            name="pattern",
        )
for level in support_resistance.get("support", []):
    fig.add_hline(y=level, line_dash="dot", line_color="green")
for level in support_resistance.get("resistance", []):
    fig.add_hline(y=level, line_dash="dot", line_color="red")
fig.update_layout(title=f"{symbol} patterns (last {window} bars)",
                  xaxis_title="date", yaxis_title="INR")
st.plotly_chart(fig, width="stretch")
data_through = df_full.index.max().date()
rows_used = df.attrs.get("rows_used", "n/a")
st.caption(f"data_through {data_through} (≤ as_of {as_of}) · "
           f"rows_used {rows_used} (bars admitted by the cutoff before "
           f"the lookback trim) · scan window {window} bars — the same "
           "green/red dotted support/resistance as the Overview chart.")
