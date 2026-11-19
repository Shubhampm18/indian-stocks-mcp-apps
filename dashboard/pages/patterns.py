"""Patterns page: candlestick-pattern table + nearest support/resistance."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import common
from stock_analyst_mcp import technicals  # noqa: E402  (path shim in common)

data = common.get_data()
if data is None:
    st.stop()
symbol, as_of, lookback, df, df_full, close_adj, current_price = data

st.subheader("Chart Patterns")
try:
    patterns_result = technicals.compute_chart_patterns(df)
    pattern_list = patterns_result.get("patterns", [])
    support_resistance = patterns_result.get("support_resistance", {})
except Exception as exc:
    st.error(str(exc))
    st.stop()

if lookback < 20:
    st.info("insufficient data (patterns scan the last 20 bars)")
elif pattern_list:
    st.dataframe(pd.DataFrame(pattern_list), hide_index=True)
else:
    st.info("No patterns in the last 20 bars")

pc = st.columns(2)
if lookback < 20:
    pc[0].metric("nearest_support", "insufficient data")
    pc[1].metric("nearest_resistance", "insufficient data")
else:
    pc[0].metric("nearest_support",
                 support_resistance.get("nearest_support") or "none below price")
    pc[1].metric("nearest_resistance",
                 support_resistance.get("nearest_resistance") or "none above price")
