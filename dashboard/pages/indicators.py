"""Indicators page: indicator table + trend badge (sidebar multiselect)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import common
from stock_analyst_mcp import technicals  # noqa: E402  (path shim in common)

with st.sidebar:
    inds = st.multiselect("Indicators", options=common.ALL_IND, default=common.ALL_IND)

data = common.get_data()
if data is None:
    st.stop()
symbol, as_of, lookback, df, df_full, close_adj, current_price = data

st.subheader("Indicators")
calc = technicals.compute_all(df, inds)
sma50, sma200 = calc.get("SMA_50"), calc.get("SMA_200")
if sma200 is None:
    trend, trend_note = "insufficient data (need lookback ≥200 for SMA_200)", ""
elif sma50 is None:
    trend, trend_note = "insufficient data (need lookback ≥50 for SMA_50)", ""
else:
    trend = "bullish" if sma50 > sma200 else "bearish"
    trend_note = f" (SMA_50 {'>' if trend == 'bullish' else '≤'} SMA_200)"
tc = st.columns(2)
tc[0].metric("current_price", f"{current_price} INR")
tc[1].metric("trend", trend, trend_note or None)
st.dataframe(pd.DataFrame([{"indicator": k, "value": v} for k, v in calc.items()]),
             hide_index=True)
