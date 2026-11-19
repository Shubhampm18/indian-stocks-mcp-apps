"""Price page: candlestick (or line) chart with pattern markers + S/R levels."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common
from stock_analyst_mcp import technicals  # noqa: E402  (path shim in common)

data = common.get_data()
if data is None:
    st.stop()
symbol, as_of, lookback, df, df_full, close_adj, current_price = data

st.subheader(f"Price — {symbol}")
show_line = st.checkbox("Show line chart", value=False)

try:
    patterns_result = technicals.compute_chart_patterns(df)
    pattern_list = patterns_result.get("patterns", [])
    support_resistance = patterns_result.get("support_resistance", {})
except Exception as exc:  # keep the chart renderable on detector hiccups
    st.error(str(exc))
    pattern_list, support_resistance = [], {}

fig = go.Figure()
if show_line:
    for column in ("CloseAdj", "Close"):
        fig.add_trace(go.Scatter(x=df.index, y=df[column], mode="lines", name=column))
else:
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
fig.update_layout(title=f"{symbol} OHLC (adjusted close)",
                  xaxis_title="date", yaxis_title="INR")
st.plotly_chart(fig, use_container_width=True)
st.caption("Blue diamonds mark candlestick patterns from the last 20 bars; "
           "green dotted = support, red dotted = resistance.")
