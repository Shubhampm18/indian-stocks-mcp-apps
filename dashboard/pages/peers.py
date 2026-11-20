"""Peers page: ranked cross-symbol comparison (own sidebar inputs).

The comparison set is hand-entered and the metric is switchable; the
ranking shown is re-ranked locally with common.rank_peers (same
semantics as the server: descending value, None last).
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.express as px
import streamlit as st

import common
from stock_analyst_mcp.server import (  # noqa: E402  (path shim in common)
    get_peer_comparison as server_peer_comparison,
)

_PEER_METRICS = ["return_20d", "return_60d", "vol_60d", "sharpe_60d",
                 "drawdown_60d", "rsi_14"]
_DEFAULT_PEERS = "RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS"

with st.sidebar:
    peers_raw = st.text_input("Peers (comma-separated, 2–4)",
                              value=_DEFAULT_PEERS)
    peer_metric = st.selectbox("Peer metric", _PEER_METRICS,
                               index=_PEER_METRICS.index("return_20d"))

data = common.get_data()
if data is None:
    st.stop()
symbol, as_of, lookback, df, df_full, close_adj, current_price = data
as_of_str = as_of.isoformat() if isinstance(as_of, date) else str(as_of)

st.subheader("Peer comparison")
peers = [p.strip() for p in peers_raw.split(",") if p.strip()][:4]
if not peers:
    peers = _DEFAULT_PEERS.split(",")
try:
    peer = server_peer_comparison(peers, as_of_str, peer_metric)
except Exception as exc:
    peer = {"error": str(exc), "results": [], "ranked": [], "errors": []}
if peer.get("isError") or peer.get("error"):
    st.error(peer["error"])
elif not peer.get("results"):
    st.warning("No valid peer data")
else:
    ranked = common.rank_peers(peer["results"], peer_metric)
    st.plotly_chart(px.bar(ranked, x="symbol", y="value",
                           title=f"{peer_metric} (ranked)"),
                    width="stretch")
    st.dataframe(pd.DataFrame(ranked), hide_index=True)
for e in peer.get("errors", []):
    st.error(f"{e.get('symbol')}: {e.get('error')}")
st.caption("Peer current_price is each peer's last trading day on/before "
           "as_of (NSE/BSE calendars can differ by a day). "
           "Peers are a hand-entered comparison set, not an index "
           "constituent list.")
