"""Returns page: simple/log returns over 1/5/20/60-day windows (pandas)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import common

data = common.get_data()
if data is None:
    st.stop()
symbol, as_of, lookback, df, df_full, close_adj, current_price = data

st.subheader("Returns (simple / log from CloseAdj, pandas)")
rows = []
for w in (1, 5, 20, 60):
    if len(close_adj) <= w or close_adj.iloc[-1 - w] == 0:
        rows.append({"window": f"{w}d", "simple": None, "log": None, "pct_simple": None})
        continue
    past, last = close_adj.iloc[-1 - w], close_adj.iloc[-1]
    simple = (last - past) / past
    rows.append({"window": f"{w}d", "simple": round(float(simple), 6),
                 "log": round(float(np.log(last / past)), 6),
                 "pct_simple": round(float(simple) * 100, 4)})
st.dataframe(pd.DataFrame(rows), hide_index=True)
